import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import csv
import random

import numpy as np
import sys
import os
from pathlib import Path

SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_ROOT = Path("/rds/user/tm681/hpc-work/ICL_locpol")
CHECKPOINT_DIR = CHECKPOINT_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_ROOT / "latest.pt"
LOSSES_CSV = CHECKPOINT_ROOT / "losses.csv"


def set_seed(seed, rank=0):
    """Seed python, numpy, and torch (including CUDA) deterministically per-rank."""
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def setup_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    """Return the underlying model when wrapped with DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model


def distributed_mean(value, world_size):
    """Average a scalar tensor across processes if distributed is initialized."""
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


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


def generate_pretrain_data(
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
    Generate pretraining data:
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


def generate_pretrain_data_scheduled(
    epoch,
    curriculum_steps,
    n_func,
    n_seq_per_func,
    alpha,
    sigma=0.01,
    device=None,
    M=256,
    freq_scale=3.14,
    outputscale=1.0,
):
    """
    Returns a list of (prompts, targets) groups.
    Each group has a uniform sequence length — no padding needed.
    The training loop accumulates loss across all groups.
    """
    d_max = 3
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = []

    if epoch + 1 <= 500:
        groups.append(generate_pretrain_data(
            n_func=n_func, n_seq_per_func=n_seq_per_func,
            n=16, d=1, d_max=d_max, beta=alpha + 0.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    elif epoch + 1 <= 1000:
        groups.append(generate_pretrain_data(
            n_func=n_func, n_seq_per_func=n_seq_per_func,
            n=16, d=2, d_max=d_max, beta=alpha + 1,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    elif epoch + 1 <= curriculum_steps:
        groups.append(generate_pretrain_data(
            n_func=n_func, n_seq_per_func=n_seq_per_func,
            n=16, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    elif epoch + 1 <= 2 * curriculum_steps:
        n_func_1 = int(max(1 - (epoch + 1 - curriculum_steps) / curriculum_steps, 0.2) * n_func)
        n_func_2 = max(n_func - n_func_1, 1)

        groups.append(generate_pretrain_data(
            n_func=n_func_1, n_seq_per_func=n_seq_per_func,
            n=16, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_2, n_seq_per_func=n_seq_per_func,
            n=21, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    elif epoch + 1 <= 3 * curriculum_steps:
        n_func_1 = int(0.2 * n_func)
        n_func_2 = int(max(0.8 - (epoch + 1 - 2 * curriculum_steps) / curriculum_steps, 0.2) * n_func)
        n_func_3 = max(n_func - n_func_1 - n_func_2, 1)

        groups.append(generate_pretrain_data(
            n_func=n_func_1, n_seq_per_func=n_seq_per_func,
            n=16, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_2, n_seq_per_func=n_seq_per_func,
            n=21, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_3, n_seq_per_func=n_seq_per_func,
            n=26, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    elif epoch + 1 <= 4 * curriculum_steps:
        n_func_1 = int(0.2 * n_func)
        n_func_2 = int(0.2 * n_func)
        n_func_3 = int(max(0.6 - (epoch + 1 - 3 * curriculum_steps) / curriculum_steps, 0.2) * n_func)
        n_func_4 = max(n_func - n_func_1 - n_func_2 - n_func_3, 1)

        groups.append(generate_pretrain_data(
            n_func=n_func_1, n_seq_per_func=n_seq_per_func,
            n=16, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_2, n_seq_per_func=n_seq_per_func,
            n=21, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_3, n_seq_per_func=n_seq_per_func,
            n=26, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_4, n_seq_per_func=n_seq_per_func,
            n=31, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    else:
        n_func_1 = int(0.2 * n_func)
        n_func_2 = int(0.2 * n_func)
        n_func_3 = int(0.2 * n_func)
        n_func_4 = int(max(0.4 - (epoch + 1 - 4 * curriculum_steps) / curriculum_steps, 0.2) * n_func)
        n_func_5 = max(n_func - n_func_1 - n_func_2 - n_func_3 - n_func_4, 1)

        groups.append(generate_pretrain_data(
            n_func=n_func_1, n_seq_per_func=n_seq_per_func,
            n=16, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_2, n_seq_per_func=n_seq_per_func,
            n=21, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_3, n_seq_per_func=n_seq_per_func,
            n=26, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_4, n_seq_per_func=n_seq_per_func,
            n=31, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))
        groups.append(generate_pretrain_data(
            n_func=n_func_5, n_seq_per_func=n_seq_per_func,
            n=36, d=3, d_max=d_max, beta=alpha + 1.5,
            sigma=sigma, device=device, M=M, freq_scale=freq_scale, outputscale=outputscale,
        ))

    return groups


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


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            return [
                base_lr * step / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * cosine_decay for base_lr in self.base_lrs]


###### Training Config
n_func = 40
n_seq_per_func = 16
sigma = 0.01
alpha = 3.0

lr = 3e-4
weight_decay = 1e-3
num_epochs = 50000
checkpoint_freq = 5000

d_embd = 256
d_ffn = 1024
n_layer = 12
n_head = 1
config = ModelConfig(d_embd=d_embd, d_ffn=d_ffn, n_layer=n_layer, n_head=n_head)

criterion = nn.MSELoss()


def _byte_state(state):
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def save_checkpoint(epoch, model, optimizer, rank, world_size):
    if not is_main_process(rank):
        return

    # Collect per-rank RNG states from all processes
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        all_rng_states = [None] * world_size
        dist.gather_object(rng_state, all_rng_states, dst=0)
    else:
        all_rng_states = [rng_state]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
        "all_rng_states": all_rng_states,
    }

    path_epoch = CHECKPOINT_DIR / f"epoch_{epoch+1:06d}.pt"
    tmp_latest = CHECKPOINT_ROOT / "latest.pt.tmp"
    torch.save(checkpoint, path_epoch)
    torch.save(checkpoint, tmp_latest)
    os.replace(tmp_latest, LATEST_CHECKPOINT)


def load_checkpoint(model, optimizer, device, rank, world_size):
    start_epoch = 0
    resumed = False

    if LATEST_CHECKPOINT.exists():
        ckpt = torch.load(LATEST_CHECKPOINT, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        resumed = True

        all_rng_states = ckpt.get("all_rng_states", None)

        if all_rng_states is not None and rank < len(all_rng_states):
            # Restore each rank's own RNG state so data generation diverges correctly
            rng = all_rng_states[rank]
            torch.set_rng_state(_byte_state(rng["torch"]))
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if torch.cuda.is_available() and "cuda" in rng:
                cuda_states = [_byte_state(s) for s in rng["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
        else:
            # Fallback if checkpoint has no per-rank states
            set_seed(SEED + start_epoch, rank)

    if not resumed:
        set_seed(SEED, rank)

    return start_epoch


def log_losses(epoch, train_loss, val_loss):
    """Append train/val losses to CSV, writing a header only on first creation."""
    LOSSES_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOSSES_CSV.exists()
    with open(LOSSES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerow([epoch, train_loss, val_loss])


def compute_group_loss(model, groups):
    """Run a forward pass per group (each has uniform sequence length) and average losses."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for prompts, targets in groups:
        outputs = model(prompts)
        total_loss = total_loss + criterion(outputs.squeeze(-1), targets)
    return total_loss / len(groups)


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_model = RegressionTransformer(config, 5).to(device)
    if world_size > 1:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
    else:
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # load_checkpoint sets the seed internally (per rank), so no separate set_seed call needed
    start_epoch = load_checkpoint(model, optimizer, device, rank, world_size)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2000, total_steps=num_epochs, last_epoch=start_epoch - 1)

    # Broadcast non-rank-0 weights from rank 0 to ensure all ranks start identically after resume
    if world_size > 1:
        for param in unwrap_model(model).parameters():
            dist.broadcast(param.data, src=0)

    model.train()

    curriculum_steps = 5000

    for epoch in range(start_epoch, num_epochs):

        # Generate data — separate group per sequence length, no padding
        groups = generate_pretrain_data_scheduled(
            epoch=epoch,
            curriculum_steps=curriculum_steps,
            n_func=n_func,
            n_seq_per_func=n_seq_per_func,
            alpha=alpha,
            sigma=sigma,
            device=device,
        )

        # Forward: accumulate loss across all length groups
        train_loss = compute_group_loss(model, groups)
        reduced_train_loss = distributed_mean(train_loss.detach().clone(), world_size)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            val_groups = generate_pretrain_data_scheduled(
                epoch=epoch,
                curriculum_steps=curriculum_steps,
                n_func=n_func,
                n_seq_per_func=n_seq_per_func,
                alpha=alpha,
                sigma=sigma,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                val_loss = compute_group_loss(model, val_groups)
                reduced_val_loss = distributed_mean(val_loss.detach().clone(), world_size)
            model.train()

            if is_main_process(rank):
                log_losses(epoch+1, reduced_train_loss.item(), reduced_val_loss.item())

        if (epoch + 1) % checkpoint_freq == 0:
            # Non-rank-0 processes send their RNG states before rank-0 saves
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                rng_state = {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda"] = torch.cuda.get_rng_state_all()
                dist.gather_object(rng_state, None if rank != 0 else [None] * world_size, dst=0)

            save_checkpoint(epoch, model, optimizer, rank, world_size)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()