# sa3.cpp — Tensor map (stable-audio-3-medium)

Derived from the real `model.safetensors` header (997 tensors, **all F32**). Full dump:
[tensors-medium.txt](tensors-medium.txt). This is the spec the GGUF converter implements.
Three top-level groups: `model.*` (DiT, 522), `pretransform.*` (AE, 472), `conditioner.*` (3).
T5Gemma weights are **not** here — they live in the repo subfolder `t5gemma-b-b-ul2`.

## Shared building blocks (appear in both AE and DiT)
- **Differential attention** — fused `to_qkv` is **5×d** (double-Q + double-K + single-V); cross-attn
  splits as `to_q` 2×d + `to_kv` 3×d. Per-head `q_norm`/`k_norm` over `dim_head=64` (RMS, `qk_norm`).
- **RoPE** — `rope.inv_freq [16]` per attention (→ 32 rotary dims of the 64-d head).
- **GLU feed-forward** — `ff.ff.0.proj` (→ N), GLU gate halves it, `ff.ff.2` back to dim.
- **Norms differ by component** (confirm exact math in `models/transformer.py` when implementing):
  - AE blocks: `pre_norm`/`ff_norm` carry **`alpha[1]` + `beta[D]` + `gamma[D]`**.
  - DiT blocks: `pre_norm`/`ff_norm`/`cross_attend_norm` carry **`gamma[D]` only** (RMSNorm), with
    modulation supplied externally by adaLN.

## DiT (`model.model.*`, embed_dim 1536, depth 24, heads 24, dim_head 64)
Structural:
- `project_in [1536,256]`, `project_out [256,1536]`; residual 1×1 `preprocess_conv`/`postprocess_conv [256,256,1]`.
- `to_timestep_embed.{0[1536,256],2[1536,1536]}` (256 = expo Fourier feature dim).
- `to_cond_embed.{0[1536,768],2}` (cross-attn cond: T5Gemma+seconds, 768→1536).
- `to_global_embed.{0[1536,768],2}` (global cond: seconds_total → 1536).
- `transformer.global_cond_embedder.{0[1536,1536],2[9216,1536]}` → **adaLN-zero** (9216 = 6×1536:
  scale/shift/gate for attn + ff).
- `transformer.memory_tokens [64,1536]` (64 register tokens); `rotary_pos_emb.inv_freq [16]`.

Per layer `transformer.layers.{0..23}.`:
- `pre_norm.gamma`, `self_attn.to_qkv [7680,1536]` (=5×1536), `self_attn.{q,k}_norm.gamma[64]`, `self_attn.to_out [1536,1536]`.
- `cross_attend_norm.gamma`, `cross_attn.to_q [3072,1536]` (=2×1536), `cross_attn.to_kv [4608,1536]` (=3×1536), `cross_attn.{q,k}_norm.gamma[64]`, `cross_attn.to_out`.
- `ff_norm.gamma`, `ff.ff.0.proj [12288,1536]` (=8×1536, GLU→6144), `ff.ff.2 [1536,6144]`.
- `to_scale_shift_gate [9216]` (per-layer adaLN base/bias).
- `to_local_embed.{0[1536,257],2[1536,1536]}` — **inpainting** `local_add_cond` (257 = 256 masked latent + 1 mask), injected **every layer**.

## AE / SAME-L (`pretransform.model.*`)
- `bottleneck.*` — **SoftNorm**: `scaling_factor[1,256,1]`, `bias[1,256,1]`, `running_std[1]`
  (`noise_scaling_factor[1,0,1]` empty since `noise_augment_dim=0`).
- Encoder/decoder are single-stage. Decoder example: `layers.1 = Linear(256→1536)`, `layers.3 =`
  resampling block holding `transformers.{0..11}` (12 layers, same block as above with α/β/γ norms),
  a weight-normed `mapping` conv (`weight_g`/`weight_v`, fold at inference), and learned
  `new_tokens [1,1,1536]` for the (variable-)stride resample. Encoder mirrors it. Patchify (×256) and
  stride (×16) give total 4096× → 256-d latents at ≈10.77 Hz.

## Conditioner (`conditioner.conditioners.*`)
- `prompt.padding_embedding [768]` — learned pad vector for T5Gemma (`padding_mode="learned"`).
- `seconds_total.embedder.embedding.1.{weight[768,256],bias[768]}` — NumberConditioner:
  expo-Fourier(→256) then `Linear(256→768)`.

## Converter notes
- Everything is F32 → straightforward GGUF pack; quantize later (Phase 7).
- Fuse weight-norm `mapping.weight_g`/`weight_v` → single conv weight at convert time.
- Keep fused `to_qkv`/`to_kv` as-is (GGML can slice) or pre-split — decide during DiT graph work.
- Bundle `model_config.json` scalars into GGUF KV metadata (acestep `convert.py` pattern).
