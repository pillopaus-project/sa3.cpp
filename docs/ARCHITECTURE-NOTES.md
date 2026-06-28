# sa3.cpp — Architecture Notes

Working notes from reading the two reference repos under `refs/`:

- `refs/stable-audio-3/` — Stability AI's official PyTorch implementation (the thing we port).
- `refs/acestep.cpp/` — a mature GGML port of a *very* similar stack (ACE-Step 1.5). Our porting playbook.

Target for this goal: the **`medium`** DiT (`stabilityai/stable-audio-3-medium`, ~1.4B) paired
with the **`SAME-L`** autoencoder (`stabilityai/SAME-L`), stereo 44.1 kHz. The medium model is the
**ARC** (distilled) variant whose default sampler is **ping-pong**.

---

## The stack (three transformers, end to end)

```
text prompt ──> T5Gemma encoder (768-d) ──┐
                                           │ cross_attn_cond
                          ┌────────────────▼─────────────────┐
   noise / init latent ──>│  DiT (continuous transformer)    │──> velocity
                          │  + pingpong sampler loop          │
                          └────────────────┬─────────────────┘
                                           │ latent (256-d @ ~86 Hz)
                          ┌────────────────▼─────────────────┐
                          │  SAME-L decoder (transformer AE) │──> stereo 44.1 kHz
                          └──────────────────────────────────┘
```

The single most important finding: **every component is attention + linear + norm + plain conv1d.**
There is no convolutional Oobleck VAE here. That means we very likely do **not** need acestep's
patched GGML fork (`GGML_OP_SNAKE`, `GGML_OP_COL2IM_1D` exist purely for Oobleck's snake activations
and transposed convs). Stock GGML should cover the whole sa3 stack. This is the biggest de-risking
fact of the project.

---

## ⭐ CONFIRMED config — `stable-audio-3-medium` (read from real `model_config.json`)

These supersede the PyTorch class defaults. Source on disk (the local weights):
`C:\Users\thegr\AppData\Local\Gary4LocalTest\hf-download-hotfix\hub\models--stabilityai--stable-audio-3-medium\snapshots\<hash>\` — `model.safetensors` (8.6 GB, contains AE under `pretransform.*` + DiT) and `model_config.json`. The T5Gemma encoder ships in subfolder `t5gemma-b-b-ul2`. `model_type = diffusion_cond_inpaint`, `sample_rate = 44100`, stereo.

**Autoencoder (= the "SAME-L" AE; class = `SAMEEncoder`/`SAMEDecoder`, config type `taae_v2`):**
- `PatchedPretransform`: patch_size **256**, channels 2 → audio becomes **512-channel** seq at 44100/256 ≈ 172 Hz before the encoder.
- Encoder/decoder: **single** resampling stage — `c_mults=[6]`, `channels=256` (→ inner dim 1536), `strides=[16]`, `transformer_depths=[12]` (12 transformer layers), `latent_dim=256`, `dim_heads=64`, `differential=True`, `variable_stride=True`, `mapping_style="none"`, `conv_bias=False`, `use_snake=False`. Decoder adds `sinusoidal_blocks=[8]`.
- **Total `downsampling_ratio = 4096`** (256 × 16) → latent ≈ **10.77 Hz**, **256-d**.
- Bottleneck: **`SoftNormBottleneck`** (`dim=256`, `noise_regularize`, `auto_scale`) — must replicate at inference.

**DiT (`type=dit`, `diffusion_objective=rf_denoiser` → ARC/pingpong):**
- `io_channels=256`, `embed_dim=1536`, `depth=24`, `num_heads=24` (**dim_head=64**), `ff` mult 4.0.
- `attn_kwargs`: **differential attention** + **`qk_norm=rms`**; `norm_type=rms_norm` with **`force_fp32`**.
- `global_cond_type=adaLN`; `timestep_features_type=expo` (ExpoFourierFeatures); **`num_memory_tokens=64`**.
- `cond_token_dim=768`, `global_cond_dim=768`.

**Conditioning (`cond_dim=768`):**
- `prompt` → **T5Gemma** (`max_length=256`, `padding_mode="learned"`).
- `seconds_total` → **`NumberConditioner`** (duration, `expo` fourier, range 0–384 s).
- `cross_attention_cond_ids=[prompt, seconds_total]`; `global_cond_ids=[seconds_total]` (→ adaLN).
- **Inpainting path:** `local_add_cond_ids=[inpaint_mask, inpaint_masked_input]`, `local_add_cond_dim=257` (256 masked latent + 1 mask). i.e. inpaint/continuation rides in via **`local_add_cond`**, *not* `input_concat_cond`.

Implications for the port: differential attention + RMS qk-norm appear in **both** the AE and the DiT
— implement once, reuse. `force_fp32` norms and fp32 timestep are non-negotiable for fidelity.

---

## Component 1 — SAME-L autoencoder (`models/autoencoders.py`)

Transformer-based, NOT convolutional. Built from `TransformerResamplingBlock` =
`WNConv1d` channel-mapping + a stack of `TransformerBlock`s, with strided down/upsampling.

- **For exact medium dims see the CONFIRMED config section above.** The class defaults in
  `autoencoders.py` (`c_mults=[1,2,4,8]`, `strides=[2,4,8,8]`, `channels=128`, `latent_dim=32`) are
  **NOT** what medium uses — medium is single-stage (`c_mults=[6]`, `strides=[16]`, `channels=256`,
  `transformer_depths=[12]`, `latent_dim=256`) behind a `PatchedPretransform(256)`, total 4096×.
- `factory.py::create_autoencoder_from_config` always builds `SAMEEncoder`/`SAMEDecoder` (the
  `taae_v2` type string is ignored) + `PatchedPretransform` + `SoftNormBottleneck`.
- `differential=True` — **differential attention**, also used in the DiT. Implement once, reuse.
- `WNConv1d` = weight-normed conv1d. At inference fold weight-norm into a plain conv1d weight.
- `decode_audio` / `encode_audio` already do chunked/overlap tiling — mirror that for long audio.

Port order: **decoder first** (it's the output path and the easiest to validate against PyTorch),
then encoder (needed for audio2audio + inpainting).

## Component 2 — DiT (`models/dit.py` + `models/transformer.py`)

`DiffusionTransformer` wrapping a `ContinuousTransformer`. Per-forward inputs:

- `x` (B, C, T) latent; `t` (B,) timestep — **kept in float32** on purpose: the logsnr transform
  `log((1-t)/t)` amplifies bf16 error ~380× near t=1. Honor this in the port.
- Timestep embed: `FourierFeatures` → `Linear→SiLU→Linear`. Optional `timestep_features_logsnr`.
- `cross_attn_cond`: T5Gemma tokens → `to_cond_embed` → cross-attention.
- `global_embed` (+ timestep) and/or `prepend_cond`: global conditioning (adaLN or prepend).
- `local_add_cond`: **the inpainting channel for medium** — `[inpaint_mask, inpaint_masked_input]`
  (dim 257) projected and added to the transformer hidden states locally. (The generic
  `input_concat_cond` path also exists in the code, but medium wires inpainting through
  `local_add_cond` per its config.)
- `preprocess_conv`/`postprocess_conv`: zero-init residual 1×1 convs.
- Objective `rf_denoiser` (ARC) → velocity; `denoised = x - sigma*v`, `sigma = t`.
- CFG is batched (cond+uncond in batch dim) with optional **APG** (adaptive projected guidance) and
  CFG-rescale. For the distilled ARC/pingpong path CFG is typically **off** (scale 1.0) — simplifies
  the first end-to-end milestone.

## Component 3 — T5Gemma text encoder (`models/conditioners.py`)

`T5GemmaConditioner` → `google/t5gemma-b-b-ul2`, output dim **768**. Encoder-only at inference.
This is the one component with no direct acestep analogue (acestep uses Qwen3). Risk: GGML/llama.cpp
support for t5gemma encoder may need a custom graph. Medium effort, isolated.

---

## Sampling — ping-pong (`inference/sampling.py::sample_flow_pingpong`)

Dead simple once the DiT works:

```
for i in steps:
    denoised = x - t_curr * model(x, t_curr, **cond)
    x = (1 - t_next) * denoised + t_next * randn_like(x)
```

- Schedule: `build_schedule` = linear `sigma_max → 0`, warped by `dist_shift` (flux/logsnr shift),
  `include_endpoint=True`. `sigma_max = init_noise_level` when starting from init audio, else 1.0.
- `rf_denoiser` objective defaults `sampler_type="pingpong"`.
- Needs a seeded RNG for `randn_like` — mirror acestep's Philox (`src/philox.h`) for bit-repro.

## Three inference modes (`model.py::generate`)

All funnel through one `sample_diffusion`:

- **text2music**: pure noise init.
- **audio2audio**: `init_audio` encoded → `init_data`; `noise = init*(1-σmax) + noise*σmax` with
  `σmax = init_noise_level`.
- **inpainting/continuation**: `inpaint_audio` + `inpaint_mask` (built from
  `inpaint_mask_start/end_seconds`, supports multiple non-contiguous regions; 0 = regenerate,
  1 = keep). Masked latents + mask ride in via `local_add_cond` (dim 257). See `models/inpainting.py`.

---

## LoRA (`models/lora/`)

PyTorch path uses `LoRAParametrization` with per-LoRA **sigma intervals** and **layer filters**
(see `dit.py` forward: `enable_lora`/`filter_lora_layers` gated on `sigma[0]`).

Porting strategy (IMPLEMENTED, src/lora.h — and NOT the static merge once planned here): the real
adapters are **DoRA** (`dora-rows`), whose forward `W_eff = magnitude·(W + (alpha/rank)·strength·B@A)/‖V‖_row`
is **non-linear in strength** and **non-additive across adapters**, so a static merge can't represent
adjustable strength or multi-adapter blending. Instead we do a **runtime weight-space recompute** of
`W_eff` per targeted weight (chaining adapters in order), as an in-place ggml graph that runs on the GPU
(~0.13s/adapter). All families (lora/dora/bora + -xs) are supported; converters take `.ckpt → safetensors
→ gguf`. dora-rows is A/B'd against trained adapters at cossim 1.0. (sigma-interval gating is not ported;
inference applies adapters once at full strength setting.)

---

## What we inherit from acestep.cpp (the playbook)

Header-driven C++17 + GGML submodule. Directly reusable patterns:

| acestep file | reuse for sa3 |
|---|---|
| `convert.py`, `quantize.sh` | GGUF conversion bundling tokenizer/config; requantize |
| `src/safetensors.h` | parse HF `.safetensors` (weights + LoRA) |
| `src/gguf-weights.h`, `src/weight-ctx.h` | load GGUF tensors into a backend |
| `src/backend.h` | CPU/CUDA/Vulkan/Metal backend selection |
| `src/model-store.*` | one-resident-module VRAM policy |
| `src/dit.h`, `src/dit-graph.h`, `src/dit-sampler.h` | DiT graph scaffolding to adapt |
| `src/solvers/solver-interface.h` (+registry) | pluggable sampler → add `pingpong` |
| `src/adapter-merge.h` | static LoRA merge |
| `src/vae.h`, `src/vae-enc.h` | tiled AE encode/decode structure (ops differ: transformer not conv) |
| `src/philox.h` | seeded noise for repro |
| `src/audio-io.h`, `src/wav.h`, `mp3/` | I/O |
| `CMakeLists.txt`, `build*.sh/.cmd` | multi-backend build |

Likely difference: acestep's GGML is a **patched fork**. We expect to use **vanilla GGML** because
the sa3 stack needs no snake/transposed-conv ops. To confirm in Phase 0/1.

## Open questions for Phase 0 (resolve against HF before coding)

1. Exact `model_config.json` for `stable-audio-3-medium` and `SAME-L`: dims, depths, head counts,
   latent_dim, RoPE/positional scheme, norm type, `differential` attention details.
2. t5gemma encoder: tensor layout + whether any existing GGML graph can be reused.
3. ARC pingpong defaults: step count, `dist_shift` type/params, whether CFG is truly off.
4. Are the HF repos gated (license click / token)? Affects the weight-fetch script.
