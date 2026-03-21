"""
Depth Recurrence + XSA + EMA + LoRA TTT
- 6 unique blocks x 2 passes = 12 effective layers (U-Net over passes)
- Per-virtual-layer params: resid_mix, attn_scale, mlp_scale (tiny, unshared)
- XSA on last 4 virtual layers
- EMA (decay=0.997)
- LoRA TTT eval (causal per-document, adapts Q/K/V + lm_head)
- FP16 tok_emb passthrough, BigramHash(10240)
- Int6 for all block weights + zstd-22
- SmearGate, orthogonal init + muP scaling
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

try:
    import zstandard

    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_unique_blocks = int(os.environ.get("NUM_UNIQUE_BLOCKS", 6))
    num_passes = int(os.environ.get("NUM_PASSES", 2))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd = float(os.environ.get("MUON_WD", 0.02))
    adam_wd = float(os.environ.get("ADAM_WD", 0.01))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 64))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # LoRA TTT
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 16))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
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
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
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
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------


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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def run_validation(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.train(False)
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            local = val_tokens[
                bs * args.train_seq_len : be * args.train_seq_len + 1
            ].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(
                -1, args.train_seq_len
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n
            val_token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = val_loss_sum / val_token_count
    model.train(was_training)
    return float(vl.item()), float(
        vl.item() / math.log(2.0) * (val_token_count.item() / val_byte_count.item())
    )


# -----------------------------
# INT6 QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p
    for p in "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_weights,smear,bigram.scale".split(
        ","
    )
    if p
)
FP16_KEEP_NAME_PATTERNS = ("tok_emb",)


def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(
            torch.int8
        )
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def quantize_state_dict(state_dict: dict[str, Tensor]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 8192:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(p in name for p in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        q, s = quantize_int6_per_row(t)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = {"type": "int6"}
    return result, meta


def dequantize_state_dict(
    result: dict[str, Tensor], meta: dict[str, object], template_sd: dict[str, Tensor]
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            if _HAS_FA3:
                # FA3 layout: [B, T, H, D] → cos/sin [1, T, 1, half_D]
                self._cos_cached = freqs.cos()[None, :, None, :]
                self._sin_cached = freqs.sin()[None, :, None, :]
            else:
                # SDPA layout: [B, H, T, D] → cos/sin [1, 1, T, half_D]
                self._cos_cached = freqs.cos()[None, None, :, :]
                self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, q_delta=None, k_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x) + (k_delta if k_delta is not None else 0)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if _HAS_FA3:
            # FA3: [B, T, H, D] layout
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
            y = flash_attn_3_func(q, k, v, causal=True)
            if self.use_xsa:
                y = self._xsa(y, v)
            y = y.reshape(bsz, seqlen, dim)
        else:
            # PyTorch SDPA: [B, H, T, D] layout
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
            if self.use_xsa:
                y_t = y.transpose(1, 2)
                v_t = v.transpose(1, 2)
                y_t = self._xsa(y_t, v_t)
                y = y_t.transpose(1, 2)
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (
            CastedLinear(bigram_dim, model_dim, bias=False)
            if bigram_dim != model_dim
            else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class SharedBlock(nn.Module):
    """Heavy weights shared across virtual layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        resid_mix: Tensor,
        attn_scale: Tensor,
        mlp_scale: Tensor,
        q_delta_fn=None,
        k_delta_fn=None,
        v_delta_fn=None,
    ) -> Tensor:
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        kd = k_delta_fn(n) if k_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        attn_out = self.attn(n, qd, kd, vd)
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_unique_blocks: int,
        num_passes: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_unique_blocks = num_unique_blocks
        self.num_passes = num_passes
        self.num_virtual_layers = num_unique_blocks * num_passes  # e.g. 6*2=12

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = (
            BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
            if bigram_vocab_size > 0
            else None
        )
        self.smear = SmearGate(model_dim)

        # Shared heavy blocks
        self.blocks = nn.ModuleList(
            [
                SharedBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_unique_blocks)
            ]
        )

        # Per-virtual-layer lightweight params (unshared)
        nvl = self.num_virtual_layers
        self.layer_resid_mix = nn.ParameterList(
            [
                nn.Parameter(
                    torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float()
                )
                for _ in range(nvl)
            ]
        )
        self.layer_attn_scale = nn.ParameterList(
            [
                nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
                for _ in range(nvl)
            ]
        )
        self.layer_mlp_scale = nn.ParameterList(
            [
                nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
                for _ in range(nvl)
            ]
        )

        # U-Net: encoder = first pass, decoder = second pass
        self.num_encoder_layers = num_unique_blocks  # pass 1
        self.num_decoder_layers = num_unique_blocks  # pass 2
        self.num_skip_weights = num_unique_blocks
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )

        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # XSA on last N virtual layers
        if xsa_last_n > 0:
            for vl in range(max(0, nvl - xsa_last_n), nvl):
                self.blocks[vl % num_unique_blocks].attn.use_xsa = True

        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        # Orthogonal init with muP scaling (scale by effective depth)
        effective_layers = self.num_virtual_layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * effective_layers))

    def _body(self, input_ids: Tensor, lora=None):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        K = self.num_unique_blocks

        # Pass 1 (encoder): virtual layers 0..K-1, using blocks 0..K-1
        for i in range(K):
            qd = lora.q_loras[i] if lora else None
            kd = lora.k_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](
                x,
                x0,
                self.layer_resid_mix[i],
                self.layer_attn_scale[i],
                self.layer_mlp_scale[i],
                qd,
                kd,
                vd,
            )
            skips.append(x)

        # Pass 2 (decoder): virtual layers K..2K-1, reusing blocks 0..K-1
        for i in range(K):
            vl = K + i  # virtual layer index
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            qd = lora.q_loras[vl] if lora else None
            kd = lora.k_loras[vl] if lora else None
            vd = lora.v_loras[vl] if lora else None
            x = self.blocks[i](
                x,
                x0,
                self.layer_resid_mix[vl],
                self.layer_attn_scale[vl],
                self.layer_mlp_scale[vl],
                qd,
                kd,
                vd,
            )

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        logits = self._body(input_ids, lora)
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
            ).reshape(bsz, sl)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self._body(input_ids)


# -----------------------------
# LoRA TTT (causal per-document)
# -----------------------------

BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        with torch.no_grad():
            self.A.uniform_(
                -1.0 / math.sqrt(self.in_features), 1.0 / math.sqrt(self.in_features)
            )
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        nvl = model.num_virtual_layers
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        # One LoRA per virtual layer (shares the underlying block weights but adapts differently)
        self.q_loras = nn.ModuleList(
            [
                BatchedLinearLoRA(
                    bsz,
                    dim,
                    model.blocks[i % model.num_unique_blocks].attn.c_q.weight.shape[0],
                    rank,
                )
                for i in range(nvl)
            ]
        )
        self.k_loras = nn.ModuleList(
            [
                BatchedLinearLoRA(
                    bsz,
                    dim,
                    model.blocks[i % model.num_unique_blocks].attn.c_k.weight.shape[0],
                    rank,
                )
                for i in range(nvl)
            ]
        )
        self.v_loras = nn.ModuleList(
            [
                BatchedLinearLoRA(
                    bsz,
                    dim,
                    model.blocks[i % model.num_unique_blocks].attn.c_v.weight.shape[0],
                    rank,
                )
                for i in range(nvl)
            ]
        )

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if s:
                s["exp_avg"].zero_()
                s["exp_avg_sq"].zero_()
                s["step"].fill_(0)


def _find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1]) + 1
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _chunk_window(ci, pred_len, num_chunks, chunk_size, seq_len):
    cs = ci * chunk_size
    ce = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    ws = max(0, ce - seq_len)
    return ws, ce - ws, cs - ws, ce - cs


def run_ttt_lora(
    args: Hyperparameters,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[
        (len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size
    ]
    cs, esl, bs, lr_rank = (
        args.ttt_chunk_size,
        args.ttt_eval_seq_len,
        args.ttt_batch_size,
        args.ttt_lora_rank,
    )
    rank_docs.sort(key=lambda d: (d[1] - 2) // cs)
    base_model.train(False)
    for p in base_model.parameters():
        p.requires_grad_(False)
    lora = BatchedTTTLoRA(bs, base_model, lr_rank).to(device)
    opt = torch.optim.Adam(
        lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-4
    )
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    for bi in range(0, len(rank_docs), bs):
        batch = rank_docs[bi : bi + bs]
        bsz = len(batch)
        if bsz == bs:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lr_rank).to(device)
            cur_opt = torch.optim.Adam(
                cur_lora.parameters(),
                lr=args.ttt_lora_lr,
                betas=(args.beta1, args.beta2),
                eps=1e-4,
            )
        pred_lens = [dl - 1 for _, dl in batch]
        ncs = [(pl + cs - 1) // cs for pl in pred_lens]
        for ci in range(max(ncs)):
            stats0 = _chunk_window(ci, (ci + 1) * cs, ci + 1, cs, esl)
            ctx_sz, co0 = stats0[1], stats0[2]
            active = [ci < nc for nc in ncs]
            needs_train = any(ci < nc - 1 for nc in ncs)
            x = torch.zeros(bsz, ctx_sz, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, ctx_sz, dtype=torch.int64, device=device)
            di = []
            for b in range(bsz):
                if not active[b]:
                    di.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _chunk_window(ci, pred_lens[b], ncs[b], cs, esl)
                chunk = all_tokens[ds + ws : ds + ws + wl + 1].to(
                    dtype=torch.int64, device=device
                )
                x[b, :wl] = chunk[:-1]
                y[b, :wl] = chunk[1:]
                di.append((co, cl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    ptl = base_model(x, y, lora=cur_lora)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = di[b]
                    lbl = ptl[b, co : co + cl].to(torch.float64)
                    tgt = y[b, co : co + cl]
                    prev = x[b, co : co + cl]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
                    loss_sum += lbl.sum()
                    byte_sum += tb.sum()
                    tok_count += cl
            if needs_train:
                mask = torch.tensor(
                    [float(ci < ncs[b] - 1) for b in range(bsz)], device=device
                )
                cur_opt.zero_grad()
                (ptl[:, co0 : co0 + cs].mean(dim=-1) * mask).sum().backward()
                cur_opt.step()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, byte_sum, tok_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = float(loss_sum.item() / tok_count.item())
    return vl, float((loss_sum.item() / math.log(2.0)) / byte_sum.item())


# -----------------------------
# SLIDING WINDOW EVAL
# -----------------------------


def run_sliding_window(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 64,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    n = len(window_starts)
    my_s, my_e = (n * rank) // world_size, (n * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.train(False)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64) + (
                    has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
                ).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / token_count).item()
    return vl, vl / math.log(2.0) * (token_count.item() / byte_count.item())


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size > 0 and 8 % world_size == 0
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

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
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # MODEL
    base_model = (
        GPT(
            vocab_size=args.vocab_size,
            num_unique_blocks=args.num_unique_blocks,
            num_passes=args.num_passes,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            bigram_vocab_size=args.bigram_vocab_size,
            bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # OPTIMIZER SETUP
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for n, p in block_named_params
        if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for n, p in block_named_params
        if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # Per-virtual-layer params are scalar (Adam)
    for pl in (
        list(base_model.layer_resid_mix)
        + list(base_model.layer_attn_scale)
        + list(base_model.layer_mlp_scale)
    ):
        scalar_params.append(pl)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram:
        scalar_params.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [
        {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
    ]
    if base_model.bigram:
        tok_params.append(
            {
                "params": [base_model.bigram.embed.weight],
                "lr": token_lr,
                "base_lr": token_lr,
            }
        )
        if base_model.bigram.proj:
            matrix_params.append(base_model.bigram.proj.weight)

    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(
            1,
            torch.optim.Adam(
                [
                    {
                        "params": [base_model.lm_head.weight],
                        "lr": args.head_lr,
                        "base_lr": args.head_lr,
                    }
                ],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=True,
            ),
        )

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(
        f"model_params:{n_params} unique_blocks:{args.num_unique_blocks} passes:{args.num_passes} virtual_layers:{base_model.num_virtual_layers}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} seed:{args.seed}"
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if wd_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        remaining = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining / max(args.warmdown_iters * step_ms, 1e-9)
            if remaining <= args.warmdown_iters * step_ms
            else 1.0
        )

    # WARMUP
    if args.warmup_steps > 0:
        init_sd = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opts, strict=True):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # EMA
    ema_state = None
    if args.ema_enabled:
        ema_state = {
            name: t.detach().float().clone()
            for name, t in base_model.state_dict().items()
        }

    # MAIN LOOP
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = run_validation(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        for group in optimizer_muon.param_groups:
            group["momentum"] = (
                1 - frac
            ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        if args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        ):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # Apply EMA
    if ema_state is not None:
        log0("ema:applying EMA weights")
        avg_state = {
            name: t.to(dtype=base_model.state_dict()[name].dtype)
            for name, t in ema_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)

    # SERIALIZATION
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(
            f"Serialized model: {os.path.getsize('final_model.pt')} bytes  Code: {len(code.encode('utf-8'))} bytes"
        )

    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = quantize_state_dict(sd_cpu)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = (
        zstandard.ZstdCompressor(level=22).compress(quant_raw)
        if _COMPRESSOR == "zstd"
        else zlib.compress(quant_raw, 9)
    )
    if master_process:
        with open("final_model.ptz", "wb") as f:
            f.write(quant_blob)
        qfb = os.path.getsize("final_model.ptz")
        log0(
            f"Serialized model int6+{_COMPRESSOR}: {qfb} bytes  Total: {qfb + len(code.encode('utf-8'))} bytes"
        )

    if distributed:
        dist.barrier()
    with open("final_model.ptz", "rb") as f:
        qbd = f.read()
    dec = (
        zstandard.ZstdDecompressor().decompress(qbd)
        if _COMPRESSOR == "zstd"
        else zlib.decompress(qbd)
    )
    qs = torch.load(io.BytesIO(dec), map_location="cpu")
    base_model.load_state_dict(
        dequantize_state_dict(qs["w"], qs["m"], sd_cpu), strict=True
    )

    # EVAL: LoRA TTT first (competition score)
    torch._dynamo.reset()
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_loss, ttt_bpb = run_ttt_lora(
        args,
        base_model,
        rank,
        world_size,
        device,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_ttt_lora val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_ttt):.0f}ms"
    )
    log0(f"final_ttt_lora_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    # EVAL: Sliding window (comparison)
    torch.cuda.synchronize()
    t_sw = time.perf_counter()
    if args.eval_stride > 0:
        sw_loss, sw_bpb = run_sliding_window(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_sw):.0f}ms"
        )
        log0(f"final_sliding_window_exact val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
