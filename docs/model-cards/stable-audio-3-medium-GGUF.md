<!-- DRAFT v1 — review with Stability (Zach) before publishing: confirm license tag, gating
     fields, and whether redistributing the GGUFs needs the same gate as the source repo. -->
---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-medium
tags:
- audio-generation
- music
- sound-effects
- diffusion
- gguf
- sa3.cpp
extra_gated_prompt: >-
  By clicking "Agree", you agree to the Stability AI Community License and acknowledge
  Stability AI's Privacy Policy. This model also includes the T5Gemma text encoder
  redistributed under the Gemma Terms of Use; by proceeding you agree to those terms as well.
extra_gated_fields:
  Name: text
  Email: text
  Country: country
  Organization or Affiliation: text
  What do you intend to use the model for?:
    type: select
    options:
      - Research
      - Personal use
      - Creative Professional
      - Startup
      - Enterprise
---

# Stable Audio 3 Medium — GGUF (for sa3.cpp)

GGUF conversions of [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp) — a portable C++/GGML port of
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, or Vulkan. Every component is validated
against the PyTorch reference at cosine similarity ~1.0.

## Files

This is a multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the
**conditioner**, plus the shared **encoder + tokenizer** from the
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/betweentwomidnights/t5gemma-b-b-ul2-GGUF) repo.

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-medium-dit-1.5B-v1.0-{F16,F32}.gguf` | pick one precision |
| autoencoder (SAME-L) | `stable-audio-3-medium-same-l-v1.0-{F16,F32}.gguf` | pick one precision |
| conditioner | `stable-audio-3-medium-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/betweentwomidnights/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

**F16** is the production path (~3.5s for 12s of audio on an 8GB laptop GPU); **F32** is for CPU
validation. The conditioner + encoder + tokenizer stay F32 (small / quality-critical).

## Usage

```bash
# pip install huggingface_hub
python tools/download_models.py --variant medium --encoding f16   # fetches this set + the shared encoder

sa3-generate --tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf \
    --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
    --cond models/stable-audio-3-medium-conditioner-v1.0-F32.gguf \
    --dit models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
    --same models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
    --prompt "upbeat funk groove with slap bass" --frames 128 --steps 8 --out song.wav
```

## License

Released under the **Stability AI Community License** (see `LICENSE`). Original weights:
[stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium).
Includes the T5Gemma text encoder under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Relationship to the original

These are **format conversions** (weights → GGUF) for inference in sa3.cpp — no retraining, no
architectural changes. See [sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md)
for the naming convention and how the pieces fit together.
