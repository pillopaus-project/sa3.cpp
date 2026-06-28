<!-- DRAFT v1 — review with Stability (Zach) before publishing: confirm license tag, gating
     fields, and whether redistributing the GGUFs needs the same gate as the source repo. -->
---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-music
tags:
- audio-generation
- music
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

# Stable Audio 3 Small-Music — GGUF (for sa3.cpp)

GGUF conversions of [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp) — a portable C++/GGML port of
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, or Vulkan. The small-music model is a
lighter, faster music generator (SAME-S autoencoder, 0.5B DiT). Validated against the PyTorch
reference at cosine similarity ~1.0.

## Files

Multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the **conditioner**, plus
the shared **encoder + tokenizer** from
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/betweentwomidnights/t5gemma-b-b-ul2-GGUF).

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-small-music-dit-0.5B-v1.0-{F16,F32}.gguf` | pick one precision |
| autoencoder (SAME-S) | `stable-audio-3-small-music-same-s-v1.0-{F16,F32}.gguf` | pick one precision |
| conditioner | `stable-audio-3-small-music-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/betweentwomidnights/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

> **note:** SAME-S needs an **even** `--frames` count (the packed sequence must divide the chunk size).

## Usage

```bash
python tools/download_models.py --variant small-music --encoding f16

sa3-generate --tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf \
    --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
    --cond models/stable-audio-3-small-music-conditioner-v1.0-F32.gguf \
    --dit models/stable-audio-3-small-music-dit-0.5B-v1.0-F16.gguf \
    --same models/stable-audio-3-small-music-same-s-v1.0-F16.gguf \
    --prompt "lo-fi hip hop beat, warm vinyl, mellow keys" --frames 128 --steps 8 --out song.wav
```

## License

Released under the **Stability AI Community License** (see `LICENSE`). Original weights:
[stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music).
Includes the T5Gemma text encoder under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Relationship to the original

**Format conversions** (weights → GGUF) for inference in sa3.cpp — no retraining. See
[sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md).
