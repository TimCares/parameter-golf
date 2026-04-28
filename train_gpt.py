"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.varlen import varlen_attn

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_iters = int(os.environ.get("WARMUP_ITERS", 0))
    warmup_graph_steps = int(os.environ.get("WARMUP_GRAPH_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    quant_n_bits = int(os.environ.get("QUANT_N_BITS", 8))


# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str) -> tuple[Tensor, Tensor]:
    doc_intervals_pattern = "_doc_intervals.".join(pattern.rsplit(".", maxsplit=1))
    # glob's "*" also matches underscores, so the tokens pattern would otherwise pick up
    # the doc_intervals shards too; filter them out explicitly.
    files = [Path(p) for p in sorted(glob.glob(pattern)) if "_doc_intervals." not in p]
    doc_intervals_files = [Path(p) for p in sorted(glob.glob(doc_intervals_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if len(doc_intervals_files) != len(files):
        raise FileNotFoundError(
            f"doc_intervals shard count ({len(doc_intervals_files)}) "
            f"!= token shard count ({len(files)}) for pattern {pattern}"
        )
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    cu_seq_lens: list[Tensor] = [torch.tensor([0], dtype=torch.int32)]
    for file in doc_intervals_files:
        cu_seq = load_data_shard(file, dtype=np.dtype(np.int32))[1:] + cu_seq_lens[-1][-1]
        cu_seq_lens.append(cu_seq)
    cu_seq_lens = torch.cat(cu_seq_lens).contiguous()
    return tokens.pin_memory(), cu_seq_lens

def build_val_regions(
    val_tokens: Tensor,
    val_cu_seq: Tensor,
    n_regions: int,
) -> list[tuple[Tensor, Tensor]]:
    # Split val into n_regions doc-aligned regions. World-size invariant: callers
    # always pass n_regions = world_size * grad_accum_steps so the regions and their
    # micro-batch plans are identical across (ws, ga) configs whose product matches.
    if val_cu_seq.numel() < n_regions + 1:
        raise ValueError(
            f"validation set has only {val_cu_seq.numel() - 1} docs, "
            f"need at least {n_regions} for world_size * grad_accum_steps"
        )

    n_tokens = val_tokens.numel()
    splits = torch.linspace(0, n_tokens, n_regions + 1).long()

    snap_idx = torch.searchsorted(val_cu_seq, splits)
    snap_idx[0] = 0
    snap_idx[-1] = val_cu_seq.numel() - 1
    # Snap-forward to enforce strict monotonicity (no zero-length region).
    for i in range(1, snap_idx.numel()):
        if snap_idx[i] <= snap_idx[i - 1]:
            snap_idx[i] = snap_idx[i - 1] + 1
    if snap_idx[-1] > val_cu_seq.numel() - 1:
        raise ValueError("world_size * grad_accum_steps too large for the validation set")

    regions: list[tuple[Tensor, Tensor]] = []
    for i in range(n_regions):
        rcu = val_cu_seq[snap_idx[i] : snap_idx[i + 1] + 1]
        t0, t1 = rcu[0].item(), rcu[-1].item()
        regions.append((val_tokens[t0:t1], rcu - t0))
    return regions


def doc_microbatches(local_cu: Tensor, max_tokens: int) -> list[tuple[int, int]]:
    doc_lens = (local_cu[1:] - local_cu[:-1]).tolist()
    batches = []
    start = 0
    cur = 0
    for i, L in enumerate(doc_lens):
        if cur + L > max_tokens and cur > 0:
            batches.append((start, i))
            start = i
            cur = 0
        cur += L
    if start < len(doc_lens):
        batches.append((start, len(doc_lens)))
    return batches


def _build_microbatch(
    region_tokens: Tensor,
    region_cu: Tensor,
    d0: int,
    d1: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    # Build (x, y, cu_seqlens, max_doc_len) on `device` for docs [d0, d1) of a region.
    # y[i] = -100 wherever x[i]'s target lives in a different doc.
    t0 = int(region_cu[d0].item())
    t1 = int(region_cu[d1].item())
    mb_cu = region_cu[d0 : d1 + 1] - t0  # CPU int32, starts at 0, ends at t1 - t0

    batch = region_tokens[t0:t1].to(device, non_blocking=True).to(torch.int64)
    x = batch[:-1]
    y = batch[1:].clone()

    boundary_vals = mb_cu[1:]
    in_x = boundary_vals <= (batch.numel() - 1)
    last_of_doc = (boundary_vals[in_x] - 1).long().to(device, non_blocking=True)
    y[last_of_doc] = -100

    cu_cpu = torch.cat([
        mb_cu[mb_cu < x.numel()].to(torch.int32),
        torch.tensor([x.numel()], dtype=torch.int32),
    ])
    max_doc_len = int((cu_cpu[1:] - cu_cpu[:-1]).max().item())
    cu_seqlens = cu_cpu.to(device, non_blocking=True)
    return x, y, cu_seqlens, max_doc_len


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    val_cu_seq: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Symmetric / world-size-invariant layout:
    #   - val is split once into n_regions = world_size * grad_accum_steps doc-aligned
    #     regions; this cut does not depend on which rank you are.
    #   - rank r owns regions [r*ga, (r+1)*ga). For (ws=1, ga=8) that's all 8; for
    #     (ws=8, ga=1) it's exactly one. The 8 micro-batches contributing to one
    #     optimizer step are the same physical tokens across configs.
    #   - For pure eval (no backward) the final metric is order-invariant so the
    #     symmetric layout matches the simpler one numerically; the value is to keep
    #     the same loop shape ready for TTT.
    n_regions = world_size * grad_accum_steps
    if args.val_batch_size % n_regions != 0:
        raise ValueError(
            f"val_batch_size={args.val_batch_size} must be divisible by "
            f"world_size * grad_accum_steps = {n_regions}"
        )
    tokens_per_microbatch = args.val_batch_size // n_regions

    regions = build_val_regions(val_tokens, val_cu_seq, n_regions)
    owned = list(range(rank * grad_accum_steps, (rank + 1) * grad_accum_steps))
    plans = {r: doc_microbatches(regions[r][1], tokens_per_microbatch) for r in owned}

    # All ranks must run the same number of optimizer steps; cap at the global min so
    # nobody is left without a micro-batch on the final step.
    n_steps_local = min(len(p) for p in plans.values())
    if dist.is_available() and dist.is_initialized():
        n_steps_t = torch.tensor(n_steps_local, device=device, dtype=torch.int64)
        dist.all_reduce(n_steps_t, op=dist.ReduceOp.MIN)
        n_steps = int(n_steps_t.item())
    else:
        n_steps = n_steps_local

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for step in range(n_steps):
            for r in owned:
                d0, d1 = plans[r][step]
                tokens_r, cu_r = regions[r]
                x, y, cu_seqlens, max_doc_len = _build_microbatch(tokens_r, cu_r, d0, d1, device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y, cu_seqlens=cu_seqlens, max_doc_len=max_doc_len).detach()

                # cross_entropy(reduction="mean") averages over non-ignored entries
                # only, so weight by the count of non-ignored targets to recover the
                # global token-mean.
                counted = (y != -100)
                n_counted = counted.sum().to(torch.float64)
                val_loss_sum += batch_loss.to(torch.float64) * n_counted
                val_token_count += n_counted

                # Byte count for BPB: skip masked positions (they're not predicted).
                # Clamp to keep the LUT lookup in-range at -100 slots.
                y_safe = y.clamp(min=0)
                token_bytes = base_bytes_lut[y_safe].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[y_safe] & ~is_boundary_token_lut[x]).to(dtype=torch.int16)
                token_bytes = token_bytes * counted.to(torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing to int{n} & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
QUANT_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "QUANT_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
QUANT_KEEP_FLOAT_MAX_NUMEL = 65_536
QUANT_KEEP_FLOAT_STORE_DTYPE = torch.float16
QUANT_PER_ROW_SCALE_DTYPE = torch.float16
QUANT_CLIP_PERCENTILE = 99.99984
QUANT_CLIP_Q = QUANT_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in QUANT_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=QUANT_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def pack_n_bits(t: Tensor, n_bits: int) -> Tensor:
    assert t.dtype == torch.int8

    min_val = -(1 << (n_bits - 1))
    max_val = (1 << (n_bits - 1)) - 1

    t_min = int(t.min().item())
    t_max = int(t.max().item())
    assert t_min >= min_val and t_max <= max_val

    x = t.flatten()
    num_values = x.numel()
    buffer_size = math.ceil(num_values * n_bits / 8)

    buffer = torch.zeros(buffer_size, dtype=torch.uint8, device=x.device)

    payload = x.to(torch.int32) % (1 << n_bits)

    global_idx = torch.arange(num_values, device=x.device, dtype=torch.int64) * n_bits
    byte_idx = global_idx // 8
    bit_offset = (global_idx % 8).to(torch.int32)

    main = ((payload << bit_offset) & 0xFF).to(torch.uint8)
    buffer.scatter_add_(0, byte_idx, main)

    spill_mask = (bit_offset + n_bits) > 8
    if spill_mask.any():
        spill_idx = byte_idx[spill_mask] + 1
        spill_shift = 8 - bit_offset[spill_mask]
        spill = ((payload[spill_mask] >> spill_shift) & 0xFF).to(torch.uint8)
        buffer.scatter_add_(0, spill_idx, spill)

    return buffer


def unpack_n_bits(buffer: Tensor, n_bits: int, shape: tuple[int, ...]) -> Tensor:
    assert buffer.dtype == torch.uint8

    buffer_size = math.prod(shape)

    global_bit_pos = torch.arange(buffer_size, device=buffer.device, dtype=torch.int64) * n_bits
    byte_idx = global_bit_pos // 8
    bit_offset = (global_bit_pos % 8).to(torch.int32)

    cur = buffer[byte_idx].to(torch.int32)
    next_ = torch.zeros_like(cur)
    next_mask = (byte_idx + 1) < buffer.numel()
    next_[next_mask] = buffer[byte_idx[next_mask] + 1].to(torch.int32)

    raw = ((cur >> bit_offset) | (next_ << (8 - bit_offset))) & ((1 << n_bits) - 1)

    sign_bit = 1 << (n_bits - 1)
    signed = torch.where(raw >= sign_bit, raw - (1 << n_bits), raw)

    return signed.to(torch.int8).reshape(shape)


def quantize_float_tensor(t: Tensor, n_bits: int) -> tuple[Tensor, Tensor]:
    max_val = 2 ** (n_bits - 1) - 1

    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), QUANT_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -int(max_val), int(max_val)).to(torch.int8).contiguous()
        return q, scale.to(dtype=QUANT_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), QUANT_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_val if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -int(max_val), int(max_val)).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict(state_dict: dict[str, Tensor], n_bits: int = 8):
    quantized: dict[str, Tensor] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "quant_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["quant_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= QUANT_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["quant_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t, n_bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = pack_n_bits(q, n_bits=n_bits)
        scales[name] = s
        shapes[name] = t.shape
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["quant_payload_bytes"] += tensor_nbytes(quantized[name]) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": f"int{n_bits}_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "shapes": shapes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict(obj: dict[str, object], n_bits: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        shape = obj["shapes"][name]
        q = unpack_n_bits(q, n_bits=n_bits, shape=shape)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path, dtype: np.dtype = np.dtype(np.uint16)) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_elems = int(header[2])
    bytes_per_item = int(header[3]) or 2  # 0 is falsy -> backward compat, default 2 bytes (uint16)
    requested_bytes_per_item = np.dtype(dtype).itemsize
    if bytes_per_item != requested_bytes_per_item:
        raise ValueError(f"Expected {bytes_per_item} bytes per item, but dtype {dtype} was requested, which is {requested_bytes_per_item} bytes per item")

    expected_size = header_bytes + num_elems * requested_bytes_per_item
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    arr = np.fromfile(file, dtype=np.dtype(dtype).newbyteorder("<"), count=num_elems, offset=header_bytes)
    if arr.size != num_elems:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(arr.astype(dtype, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        doc_intervals_pattern = "_doc_intervals.".join(pattern.rsplit(".", maxsplit=1))
        # glob's "*" also matches underscores, so the tokens pattern would otherwise pick
        # up the doc_intervals shards too; filter them out explicitly.
        self.files = [Path(p) for p in sorted(glob.glob(pattern)) if "_doc_intervals." not in p]
        self.doc_intervals_files = [Path(p) for p in sorted(glob.glob(doc_intervals_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        if len(self.doc_intervals_files) != len(self.files):
            raise FileNotFoundError(
                f"doc_intervals shard count ({len(self.doc_intervals_files)}) "
                f"!= token shard count ({len(self.files)}) for pattern {pattern}"
            )
        self.file_idx = 0

        self._load_file_idx(self.file_idx)

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self._load_file_idx(self.file_idx)
    
    def _load_file_idx(self, file_idx: int) -> None:
        self.tokens = load_data_shard(self.files[file_idx]).pin_memory()
        self.doc_intervals = load_data_shard(self.doc_intervals_files[file_idx], dtype=np.dtype(np.int32))
        self.pos = 0

    def take(self, n: int) -> tuple[Tensor, Tensor]:
        cross_shard_len = 0
        chunks: list[Tensor] = []
        cu_chunks: list[Tensor] = [torch.tensor([0], dtype=torch.int32)]
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                cross_shard_len = cu_chunks[-1][-1].item()
                continue
            k = min(remaining, avail)
            start = self.pos
            end = self.pos + k
            chunks.append(self.tokens[start : end])
            mask = (self.doc_intervals > start) & (self.doc_intervals < end)
            inter = self.doc_intervals[mask] - start
            cu = torch.cat([
                inter.to(torch.int32),
                torch.tensor([end - start], dtype=torch.int32),
            ]) + cross_shard_len
            cu_chunks.append(cu)
            self.pos = end
            remaining -= k
        
        cat_chunks = chunks[0] if len(chunks) == 1 else torch.cat(chunks).pin_memory()
        cu_seqlens = cu_chunks[0] if len(cu_chunks) == 1 else torch.cat(cu_chunks)
        return cat_chunks, cu_seqlens


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, grad_accum_steps: int) -> tuple[Tensor, Tensor, Tensor, int]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk, full_cu_seqlens = self.stream.take(per_rank_span * self.world_size)
        
        rank_start = self.rank * per_rank_span
        rank_end = rank_start + per_rank_span

        local = chunk[rank_start : rank_end]
        x = local[:-1].to(self.device, non_blocking=True).to(torch.int64)
        y = local[1:].to(self.device, non_blocking=True).to(torch.int64)

        after_rank_start_mask = full_cu_seqlens > rank_start

        # compute local cu_seq
        n_input = local_tokens
        interior = full_cu_seqlens[after_rank_start_mask & (full_cu_seqlens < rank_start + n_input)] - rank_start
        local_cu_cpu = torch.cat([
            torch.tensor([0], dtype=torch.int32),
            interior.to(torch.int32),
            torch.tensor([n_input], dtype=torch.int32),
        ])
        # Compute max_doc_len on the CPU side so the model forward stays graph-break free
        # (varlen_attn wants a Python int, and .item() inside torch.compile fullgraph is a
        # hard error). Mirror this in eval's _build_microbatch.
        max_doc_len = int((local_cu_cpu[1:] - local_cu_cpu[:-1]).max().item())
        local_cu_seqlens = local_cu_cpu.to(self.device, non_blocking=True)

        # compute loss mask for BOS of doc n+1 for doc n
        boundary_vals = full_cu_seqlens[
            after_rank_start_mask & (full_cu_seqlens <= rank_start + n_input)
        ]
        doc_end_in_x = (boundary_vals - rank_start - 1).long().to(self.device, non_blocking=True)
        y[doc_end_in_x] = -100

        return (
            x,
            local_cu_seqlens,
            y,
            max_doc_len,
        )

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps) * self.scale


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def prepare_positions(self, n_tokens: int, cu_seqlens: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        repeats = cu_seqlens[1:] - cu_seqlens[:-1]
        positions = torch.arange(n_tokens, device=cu_seqlens.device) - cu_seqlens[:-1].repeat_interleave(repeats)
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)
        return freqs.cos().to(dtype)[:, None, :], freqs.sin().to(dtype)[:, None, :]

    @staticmethod
    def apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

        self.q_rms_norm = RMSNorm()
        self.k_rms_norm = RMSNorm()

    def forward(self, x: Tensor, cu_seqlens: Tensor, rotary_cos: Tensor, rotary_sin: Tensor, max_doc_len: int) -> Tensor:
        n_tokens, dim = x.shape
        q = self.c_q(x).reshape(n_tokens, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(n_tokens, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(n_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        q = Rotary.apply(q, rotary_cos, rotary_sin)
        k = Rotary.apply(k, rotary_cos, rotary_sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None]
        y = varlen_attn(
            q,
            k,
            v,
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_doc_len,
            max_k=max_doc_len,
            window_size=(-1, 0),
        )
        y = y.contiguous().reshape(n_tokens, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        qk_gain_init: float,
    ):
        super().__init__()
        
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, qk_gain_init)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        
        self.mlp = MLP(dim, mlp_mult)
        self.mlp_norm = RMSNorm()
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, cu_seqlens: Tensor, rotary_cos: Tensor, rotary_sin: Tensor, max_doc_len: int) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, :] * x + mix[1][None, :] * x0

        attn_out = self.attn(self.attn_norm(x), cu_seqlens=cu_seqlens, rotary_cos=rotary_cos, rotary_sin=rotary_sin, max_doc_len=max_doc_len)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, :] * attn_out
        
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.tok_norm = RMSNorm()
        self.num_encoder_layers = int(math.ceil(num_layers / 2))
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.rotary = Rotary(dim=model_dim // num_heads, base=rope_base)

        self.num_layers = num_layers
        self.even_layers = self.num_layers % 2 == 0

        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)

        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, cu_seqlens: Tensor, max_doc_len: int) -> Tensor:
        n_tokens = input_ids.numel()

        x = self.tok_emb(input_ids)
        x = self.tok_norm(x)
        x0 = x
        skips: list[Tensor] = []

        cos, sin = self.rotary.prepare_positions(
            n_tokens=n_tokens,
            cu_seqlens=cu_seqlens,
            dtype=x.dtype,  # will be the same as q
        )

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, cu_seqlens=cu_seqlens, rotary_cos=cos, rotary_sin=sin, max_doc_len=max_doc_len)

            if self.even_layers or i < self.num_encoder_layers - 1:
                skips.append(x)
        
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, cu_seqlens=cu_seqlens, rotary_cos=cos, rotary_sin=sin, max_doc_len=max_doc_len)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        logits_proj = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens, val_cu_seq = load_validation_tokens(args.val_files)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # dynamic=None: auto. n_tokens is fixed per step (deterministic packing) so the bulk
    # of the graph stays statically specialized; cu_seqlens length and max_doc_len change
    # per batch, and Dynamo will mark them dynamic on the first recompile and then settle.
    # dynamic=False would recompile every batch; dynamic=True gives up static specialization.
    compiled_model = torch.compile(base_model, dynamic=None, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    scalar_params.append(base_model.tok_norm.scale)
    scalar_params.append(base_model.final_norm.scale)

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    
    optimizer_scalar = torch.optim.Adam([
            {"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())

    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"embed_lr:{args.embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_graph_steps:{args.warmup_graph_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul_iters(step: int) -> tuple[float, float]:
        if args.warmup_iters > 0 and step < args.warmup_iters:
            return min(max(step / args.warmup_iters, 1e-3), 1.0), 1.0
        
        if args.warmdown_iters <= 0:
            return 1.0, 1.0

        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        warmdown_lr_mul = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)

        if warmdown_start <= step < args.iterations:
            return warmdown_lr_mul, warmdown_lr_mul
        else:
            return 1.0, 1.0

    def lr_mul_wallclock(step: int, elapsed_ms: float) -> tuple[float, float]:
        assert max_wallclock_ms is not None

        step_ms = elapsed_ms / max(step, 1)
        warmup_ms = args.warmup_iters * step_ms
        is_warmup = args.warmup_iters > 0 and warmup_ms > 0 and elapsed_ms <= warmup_ms

        if is_warmup:
            return min(max(elapsed_ms / warmup_ms, 1e-3), 1.0), 1.0
        
        if args.warmdown_iters <= 0:
            return 1.0, 1.0
        
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        warmdown_lr_mul = remaining_ms / max(warmdown_ms, 1e-9)

        if remaining_ms <= warmdown_ms:
            return warmdown_lr_mul, warmdown_lr_mul
        else:
            return 1.0, 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_graph_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_graph_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, cu_seqlens, y, max_doc_len = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y, cu_seqlens=cu_seqlens, max_doc_len=max_doc_len)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_graph_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_graph_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_graph_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                val_cu_seq,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale_adam, scale_muon = lr_mul_wallclock(step, elapsed_ms) if max_wallclock_ms is not None else lr_mul_iters(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, cu_seqlens, y, max_doc_len = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, cu_seqlens=cu_seqlens, max_doc_len=max_doc_len)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * (scale_adam if isinstance(opt, torch.optim.Adam) else scale_muon)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed quantized+zlib artifact and validate the round-tripped weights.

    n_bits = args.quant_n_bits
    quant_filename = f"final_model.int{n_bits}.ptz"

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict(base_model.state_dict(), n_bits=n_bits)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open(quant_filename, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_filename)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["quant_payload_bytes"], 1)
        log0(
            f"Serialized model int{n_bits}+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['quant_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int{n_bits}+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(quant_filename, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state, n_bits=n_bits), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        val_cu_seq,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int{n_bits}_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int{n_bits}_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()