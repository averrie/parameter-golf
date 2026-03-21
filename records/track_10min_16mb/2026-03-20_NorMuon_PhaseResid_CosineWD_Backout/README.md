# NorMuon + PhaseResid + CosineWD + Backout

**val_bpb:** (pending 8xH100 run)

## Changes from baseline `train_gpt.py`

### Hyperparameter Changes

| Parameter | Baseline | Submission | Rationale |
|-----------|----------|------|-----------|
| NUM_LAYERS | 9 | 10 | Extra layer within 16MB budget |
| MLP_MULT | 2 | 3.0 | 3x MLP expansion (biggest quality win) |
| TRAIN_SEQ_LEN | 1024 | 2048 | Longer context for training and eval |
| ROPE_BASE | 10,000 | 50,000 | Better long-context utilization at seq_len=2048 |
| LOGIT_SOFTCAP | 30 | 20 | Tighter logit clipping |
| WARMDOWN_ITERS | 1200 | 3000 | Longer warmdown phase |
| TIED_EMBED_LR | 0.05 | 0.03 | Tuned for 10L architecture |
| MATRIX_LR | 0.04 | 0.02 | Tuned for 10L architecture |
| SCALAR_LR | 0.04 | 0.02 | Tuned for 10L architecture |
| MUON_MOMENTUM | 0.95 | 0.99 | Higher momentum with warmup 0.92 to 0.99 |
| MUON_MOMENTUM_WARMUP | 0.85 to 0.95 over 500 steps | 0.92 to 0.99 over 1500 steps | 3x longer ramp to higher final momentum |
| GRAD_CLIP_NORM | 0.0 | 0.3 | Gradient clipping for stability |

### New Architecture Components

- **SmearGate**: Learned per-dimension gate blending each token with the previous token's embedding. Applied after RMS norm, before transformer blocks.

- **BigramHashEmbedding** (vocab=10240, dim=128): XOR-based hash of consecutive token pairs into a 10240-bucket embedding table, projected to model_dim. Captures bigram statistics additively.

- **Backout mechanism**: Learnable subtraction of mid-layer residual from final output: `x = x - backout_lambda * x_midlayer` (init 0.2). Removes stale low-level features from the residual stream.

- **Orthogonal weight initialization** with muP output scaling: `orthogonal_(gain=1.0)` for large matrices, output projections scaled by `1/sqrt(2*num_layers)`.

### New Training Techniques

- **NorMuon**: Per-neuron second-moment EMA in Muon optimizer. After Newton-Schulz orthogonalization, scales updates by `rsqrt(EMA(g^2))` per row. Interacts with cautious WD and phase-transition resid_mix.

- **Phase-transition resid_mix initialization**: Sigmoid-scheduled init — early layers blend more of x0 (initial embedding), late layers trust the residual stream. `phase = sigmoid(3.0 * (i/(L-1) - 0.5))`.

- **Cosine weight decay schedule**: `WD(t) = WD_base * 0.5 * (1 + cos(pi * t/T))`, decaying from 0.04 to 0 over training.

- **Cautious weight decay in Muon**: Only applies weight decay when the update and weight agree in sign: `mask = (g * p) >= 0`.

- **Momentum cooldown**: During warmdown, Muon momentum decays from 0.99 toward 0.90 proportional to LR decay.

- **SWA** (Stochastic Weight Averaging): Collects checkpoints every 50 steps when LR scale < 0.4, averages at end of training.

### New Quantization Scheme

- **Mixed int5-MLP / int6-attn**: MLP weights quantized to 5-bit range (clip=15), attention to 6-bit range (clip=31). Both stored as int8 for zstd compression.
- **FP16 passthrough** for tied embeddings and last-layer c_k (quantization-sensitive).
- **Magnitude pruning 5%**: Zeros out bottom 5th percentile of large weight matrices before quantization for better compression.
- **zstd level 22** compression (falls back to zlib-9 if zstandard unavailable).

### Evaluation Methods

- **LoRA TTT** (competition score): Per-document rank-8 LoRA adapters on Q, V projections and lm_head. Single Adam step per 256-token chunk, reset between documents. Causal: each chunk is scored before being used for training, so no tokens are trained on before evaluation.

- **Sliding window eval** (stride=64): Each token scored with near-full context instead of 0-2047 average.

### Reports two metrics
- `final_sliding_window` — sliding window on quantized model (comparison)
- `final_ttt_lora` — LoRA TTT (competition score)

## Run Command
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
