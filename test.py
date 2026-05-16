import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import csv
import random
import itertools

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_ROOT = Path("/rds/user/tm681/hpc-work/ICL_locpol")
CHECKPOINT_DIR = CHECKPOINT_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_ROOT / "latest.pt"
TEST_TF = CHECKPOINT_ROOT / "test_TF.csv"
TEST_LP = CHECKPOINT_ROOT / "test_LP.csv"




###### Sampling Utilities
def sample_random_function_values_grouped(
    X,                  # (n_func, m, d)
    beta,               # larger = smoother
    M=256,              # number of random trigonometric features
    freq_scale=3.14,    # smaller = smoother
    outputscale=1.0,
):
    """
    Sample n_func random smooth functions and evaluate each one on its own set of m points.

    Model:
        f_i(x) = (1/sqrt(M)) sum_{r=1}^M a_{i,r} cos(w_{i,r}^T x + phi_{i,r})

    where for each function i:
        w_{i,r} ~ N(0, freq_scale^2 I_d)
        phi_{i,r} ~ Uniform[0, 2pi]
        a_{i,r} ~ N(0, outputscale^2 * (1 + ||w_{i,r}||^2)^(-beta))
    """
    n_func, m, d = X.shape
    device = X.device
    dtype = X.dtype

    omega = freq_scale * torch.randn(n_func, M, d, device=device, dtype=dtype)   # (n_func, M, d)
    phi = 2.0 * math.pi * torch.rand(n_func, M, device=device, dtype=dtype)      # (n_func, M)

    omega_norm_sq = (omega ** 2).sum(dim=-1)                                      # (n_func, M)
    amp_std = outputscale / (1.0 + omega_norm_sq).pow(beta / 2.0)                 # (n_func, M)
    a = amp_std * torch.randn(n_func, M, device=device, dtype=dtype)              # (n_func, M)

    proj = torch.einsum("fmd,fkd->fmk", X, omega) + phi[:, None, :]

    Y = (a[:, None, :] * torch.cos(proj)).sum(dim=-1) / math.sqrt(M)              # (n_func, m)
    return Y


def generate_test_data(
    n_func,
    n_seq_per_func,
    n,
    d,
    d_max,
    beta,
    sigma=0.01,
    outputscale=1.0,
    device=None,
    M=256,
    freq_scale=3.14,
):
    """
    Generate test data:
    - sample n_func latent random functions,
    - for each, generate n_seq_per_func sequences of length n.

    Output:
        prompts: shape (batch_size, n, d_max + 2)
        target_values: shape (batch_size,)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    batch_size = n_func * n_seq_per_func
    m = n_seq_per_func * n

    X = torch.rand(n_func, m, d, dtype=dtype, device=device)                      # (n_func, m, d)

    Y = sample_random_function_values_grouped(
        X,
        beta=beta,
        M=M,
        freq_scale=freq_scale,
        outputscale=outputscale,
    )                                                                              # (n_func, m)

    if sigma > 0:
        Y = Y + sigma * torch.randn_like(Y)

    X = X.view(n_func, n_seq_per_func, n, d)
    Y = Y.view(n_func, n_seq_per_func, n)

    X_flat = X.reshape(batch_size, n, d)
    Y_flat = Y.reshape(batch_size, n)

    prompts = torch.zeros((batch_size, n, d_max + 2), dtype=dtype, device=device)
    target_values = Y_flat[:, n - 1].clone()

    # x-coordinates in channels 0,...,d-1
    prompts[:, :, :d] = X_flat

    # observed y in fixed channel
    prompts[:, :, d_max] = Y_flat

    # mask final query response
    prompts[:, n - 1, d_max] = 0.0

    # query-indicator channel
    prompts[:, n - 1, d_max + 1] = 1.0

    return prompts, target_values





###### Transformer Model Definition
class ModelConfig:
    """Holds model hyperparameters."""
    def __init__(self, d_embd, d_ffn, n_layer, n_head, max_length=None):
        self.d_embd = d_embd
        self.d_ffn = d_ffn
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_length = max_length if max_length is not None else 1024


class MultiHeadAttention(nn.Module):
    """
    Linear self-attention without softmax:
        Y = (Q K^T V) / sqrt(d_head)
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.d_embd, 3 * config.d_embd)
        self.c_proj = nn.Linear(config.d_embd, config.d_embd)
        self.n_head = config.n_head
        self.d_embd = config.d_embd
        self.head_dim = config.d_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.d_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        kv = k.transpose(-2, -1) @ v                                # (B, nh, hs, hs)
        y = (q @ kv) * (1.0 / math.sqrt(self.head_dim))             # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)            # (B, T, C)
        y = self.c_proj(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class Block(nn.Module):
    """A single transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_embd, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RegressionTransformer(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(input_dim, config.d_embd)
        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.output_layer = nn.Linear(config.d_embd, 1)

    def forward(self, x):
        # x: (B, T, input_dim)
        # The last input channel is the query indicator (1.0 at query position, 0.0 elsewhere).
        # Use argmax to find the query token position for each batch element.
        query_idx = x[:, :, -1].argmax(dim=1)          # (B,)

        x = self.input_proj(x)

        for block in self.transformer.h:
            x = block(x)

        # Select the query token for each batch element rather than blindly taking the last token.
        # This is correct regardless of padding or sequence length.
        batch_idx = torch.arange(x.size(0), device=x.device)
        query_token = x[batch_idx, query_idx]           # (B, d_embd)
        y = self.output_layer(query_token)              # (B, 1)
        return y

    

# Model Config
d_embd = 256
d_ffn = 1024
n_layer = 12
n_head = 1
config = ModelConfig(d_embd=d_embd, d_ffn=d_ffn, n_layer=n_layer, n_head=n_head)

criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionTransformer(config, 5).to(device)
ckpt = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
def unwrap_model(model):
    """Return the underlying model when wrapped with DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model
unwrap_model(model).load_state_dict(ckpt["model_state_dict"])




##### Local polynomial regressionm
def polynomial_features_torch(Z, degree=3):
    """
    Polynomial features up to total degree `degree`.

    Parameters
    ----------
    Z : torch.Tensor, shape (..., d)
        Centred covariates X_i - x_query.

    Returns
    -------
    Phi : torch.Tensor, shape (..., p)
        Polynomial feature tensor, including intercept.
        For d=3, degree=3, p = binomial(3+3, 3) = 20.
    """
    d = Z.shape[-1]
    features = [torch.ones_like(Z[..., :1])]

    for deg in range(1, degree + 1):
        for comb in itertools.combinations_with_replacement(range(d), deg):
            term = torch.ones_like(Z[..., :1])
            for j in comb:
                term = term * Z[..., j:j+1]
            features.append(term)

    return torch.cat(features, dim=-1)


def local_poly_predict_torch(
    X_train,
    y_train,
    X_query,
    bandwidths,
    ridge_alphas,
    degree=3,
):
    """
    Batched local polynomial regression on GPU/CPU.

    Parameters
    ----------
    X_train     : torch.Tensor, shape (B, m, d)
    y_train     : torch.Tensor, shape (B, m)
    X_query     : torch.Tensor, shape (B, q, d)
    bandwidths  : torch.Tensor, shape (H,)
    ridge_alphas: torch.Tensor, shape (R,)
    degree      : int

    Returns
    -------
    preds : torch.Tensor, shape (B, H, R, q)
    """
    device = X_train.device
    dtype  = X_train.dtype

    B, m, d = X_train.shape
    q = X_query.shape[1]
    H = bandwidths.numel()
    R = ridge_alphas.numel()

    Z   = X_train[:, None, :, :] - X_query[:, :, None, :]    # (B, q, m, d)
    Phi = polynomial_features_torch(Z, degree=degree)         # (B, q, m, p)
    p   = Phi.shape[-1]

    dist2 = (Z ** 2).sum(dim=-1)                              # (B, q, m)

    h  = bandwidths.to(device=device, dtype=dtype)
    h2 = h.view(1, H, 1, 1) ** 2
    weights = torch.exp(-0.5 * dist2[:, None, :, :] / h2)    # (B, H, q, m)

    Phi_H = Phi[:, None, :, :, :]                             # (B, 1, q, m, p)
    W     = weights[..., None]                                # (B, H, q, m, 1)

    A_base = torch.einsum("bhqmp,bhqmn->bhqpn", Phi_H * W, Phi_H)  # (B, H, q, p, p)
    y      = y_train[:, None, None, :, None]                        # (B, 1, 1, m, 1)
    rhs    = torch.einsum("bhqmp,bhqmo->bhqpo", Phi_H * W, y)       # (B, H, q, p, 1)

    eye = torch.eye(p, device=device, dtype=dtype)
    ra  = ridge_alphas.to(device=device, dtype=dtype)         # (R,)

    # Broadcast ridge dimension: (B, H, R, q, p, p)
    A   = A_base[:, :, None, :, :, :] + ra.view(1, 1, R, 1, 1, 1) * eye
    rhs = rhs[:, :, None, :, :, :]                            # (B, H, R, q, p, 1)

    theta = torch.linalg.solve(A, rhs)                        # (B, H, R, q, p, 1)
    preds = theta[..., 0, 0]                                  # (B, H, R, q)

    return preds
    
    
    
def make_kfold_indices(m, n_splits=5, random_state=2026):
    """
    Deterministic K-fold split of indices {0, ..., m-1}.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(m)
    folds = np.array_split(indices, n_splits)
    return folds


def choose_bandwidth_ridge_5fold_torch(
    X_context,
    y_context,
    bandwidth_grid,
    ridge_grid,
    degree=3,
    n_splits=5,
):
    """
    Select bandwidth and ridge penalty jointly by 5-fold CV.

    Returns
    -------
    best_h       : torch.Tensor, shape (B,)
    best_h_idx   : torch.Tensor, shape (B,)
    best_r       : torch.Tensor, shape (B,)
    best_r_idx   : torch.Tensor, shape (B,)
    cv_mse       : torch.Tensor, shape (B, H, R)
    """
    device = X_context.device
    B, m, d = X_context.shape
    H = bandwidth_grid.numel()
    R = ridge_grid.numel()

    n_splits   = min(n_splits, m)
    folds      = make_kfold_indices(m, n_splits=n_splits)
    all_idx    = np.arange(m)

    cv_sse   = torch.zeros(B, H, R, device=device, dtype=X_context.dtype)
    cv_count = 0

    for val_idx_np in folds:
        train_idx_np = np.setdiff1d(all_idx, val_idx_np)
        train_idx = torch.as_tensor(train_idx_np, device=device, dtype=torch.long)
        val_idx   = torch.as_tensor(val_idx_np,   device=device, dtype=torch.long)

        preds = local_poly_predict_torch(
            X_train      = X_context[:, train_idx, :],
            y_train      = y_context[:, train_idx],
            X_query      = X_context[:, val_idx, :],
            bandwidths   = bandwidth_grid,
            ridge_alphas = ridge_grid,
            degree       = degree,
        )                                                       # (B, H, R, m_val)

        errors    = (preds - y_context[:, None, None, val_idx]) ** 2
        cv_sse   += errors.sum(dim=-1)
        cv_count += len(val_idx_np)

    cv_mse = cv_sse / cv_count                                 # (B, H, R)

    # Joint argmin over (H, R)
    flat_idx   = cv_mse.view(B, H * R).argmin(dim=1)          # (B,)
    best_h_idx = flat_idx // R
    best_r_idx = flat_idx %  R

    best_h = bandwidth_grid.to(device)[best_h_idx]
    best_r = ridge_grid.to(device)[best_r_idx]

    return best_h, best_h_idx, best_r, best_r_idx, cv_mse
    
    

def evaluate_local_cubic_gpu_on_batch(
    prompts,
    targets,
    d,
    d_max,
    bandwidth_grid,
    ridge_grid,
    degree=3,
    n_splits=5,
):
    device = prompts.device
    B, n, _ = prompts.shape

    X_context = prompts[:, :n - 1, :d]
    y_context = prompts[:, :n - 1, d_max]
    X_query   = prompts[:, n - 1:n, :d]

    bandwidth_grid = torch.as_tensor(bandwidth_grid, dtype=prompts.dtype, device=device)
    ridge_grid     = torch.as_tensor(ridge_grid,     dtype=prompts.dtype, device=device)

    _, best_h_idx, _, best_r_idx, _ = choose_bandwidth_ridge_5fold_torch(
        X_context    = X_context,
        y_context    = y_context,
        bandwidth_grid = bandwidth_grid,
        ridge_grid     = ridge_grid,
        degree       = degree,
        n_splits     = n_splits,
    )

    all_preds = local_poly_predict_torch(
        X_train      = X_context,
        y_train      = y_context,
        X_query      = X_query,
        bandwidths   = bandwidth_grid,
        ridge_alphas = ridge_grid,
        degree       = degree,
    ).squeeze(-1)                                               # (B, H, R)

    batch_idx = torch.arange(B, device=device)
    preds = all_preds[batch_idx, best_h_idx, best_r_idx]       # (B,)

    return ((preds - targets) ** 2).detach().cpu().numpy()



degree = 3
n_splits = 5
ridge_alpha = 1e-2

bandwidth_grid = torch.tensor(
    [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.25, 1.50, 2.00],
    dtype=torch.float32, device=device,
)
ridge_grid = torch.tensor(
    [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    dtype=torch.float32, device=device,
)





def set_seed(seed):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2026)



model.eval()
with torch.no_grad():
    for i in range(100):
        loss_TF_1, loss_TF_2, loss_TF_3, loss_TF_4, loss_TF_5 = [], [], [], [], []
        loss_LP_1, loss_LP_2, loss_LP_3, loss_LP_4, loss_LP_5 = [], [], [], [], []
        prompts, targets = generate_test_data(
            n_func=1000, n_seq_per_func=1,
            n=16, d=3, d_max=3, beta=3 + 1.5,
            sigma=0.01, device=device
        )
        outputs = model(prompts)
        loss_TF_1.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)
        loss_LP_1.append(evaluate_local_cubic_gpu_on_batch(
            prompts, targets,
            d=3, d_max=3,
            bandwidth_grid=bandwidth_grid,
            degree=degree,
            n_splits=n_splits,
            ridge_grid=ridge_grid
        ))


        prompts, targets = generate_test_data(
            n_func=1000, n_seq_per_func=1,
            n=21, d=3, d_max=3, beta=3 + 1.5,
            sigma=0.01, device=device
        )
        outputs = model(prompts)
        loss_TF_2.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)
        loss_LP_2.append(evaluate_local_cubic_gpu_on_batch(
            prompts, targets,
            d=3, d_max=3,
            bandwidth_grid=bandwidth_grid,
            degree=degree,
            n_splits=n_splits,
            ridge_grid=ridge_grid
        ))


        prompts, targets = generate_test_data(
            n_func=1000, n_seq_per_func=1,
            n=26, d=3, d_max=3, beta=3 + 1.5,
            sigma=0.01, device=device
        )
        outputs = model(prompts)
        loss_TF_3.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)
        loss_LP_3.append(evaluate_local_cubic_gpu_on_batch(
            prompts, targets,
            d=3, d_max=3,
            bandwidth_grid=bandwidth_grid,
            degree=degree,
            n_splits=n_splits,
            ridge_grid=ridge_grid
        ))
        

        prompts, targets = generate_test_data(
            n_func=1000, n_seq_per_func=1,
            n=31, d=3, d_max=3, beta=3 + 1.5,
            sigma=0.01, device=device
        )
        outputs = model(prompts)
        loss_TF_4.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)
        loss_LP_4.append(evaluate_local_cubic_gpu_on_batch(
            prompts, targets,
            d=3, d_max=3,
            bandwidth_grid=bandwidth_grid,
            degree=degree,
            n_splits=n_splits,
            ridge_grid=ridge_grid
        ))
        

        prompts, targets = generate_test_data(
            n_func=1000, n_seq_per_func=1,   
            n=36, d=3, d_max=3, beta=3 + 1.5,
            sigma=0.01, device=device
        )
        outputs = model(prompts)
        loss_TF_5.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)
        loss_LP_5.append(evaluate_local_cubic_gpu_on_batch(
            prompts, targets,
            d=3, d_max=3,
            bandwidth_grid=bandwidth_grid,
            degree=degree,
            n_splits=n_splits,
            ridge_grid=ridge_grid
        ))
        
        
        
        loss_TF_1 = np.concatenate(loss_TF_1)
        loss_TF_2 = np.concatenate(loss_TF_2)
        loss_TF_3 = np.concatenate(loss_TF_3)
        loss_TF_4 = np.concatenate(loss_TF_4)
        loss_TF_5 = np.concatenate(loss_TF_5)
        
        loss_LP_1 = np.concatenate(loss_LP_1)
        loss_LP_2 = np.concatenate(loss_LP_2)
        loss_LP_3 = np.concatenate(loss_LP_3)
        loss_LP_4 = np.concatenate(loss_LP_4)
        loss_LP_5 = np.concatenate(loss_LP_5)
        
        df_TF = pd.DataFrame({
            "n_15": loss_TF_1,
            "n_20": loss_TF_2,
            "n_25": loss_TF_3,
            "n_30": loss_TF_4,
            "n_35": loss_TF_5
        })
        df_TF.to_csv(
            TEST_TF,
            mode="a",
            header=not TEST_TF.exists(),
            index=False
        )
        
        df_LP = pd.DataFrame({
            "n_15": loss_LP_1,
            "n_20": loss_LP_2,
            "n_25": loss_LP_3,
            "n_30": loss_LP_4,
            "n_35": loss_LP_5
        })
        df_LP.to_csv(
            TEST_LP,
            mode="a",
            header=not TEST_LP.exists(),
            index=False
        )
        

