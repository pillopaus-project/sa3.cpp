<!-- DRAFT v1 — review with Stability (Zach) before publishing. This is the shared text
     encoder used by all three SA3 variants; it is Google's T5Gemma under the Gemma Terms. -->
---
language:
- en
license: gemma
license_link: https://ai.google.dev/gemma/terms
pipeline_tag: text-to-audio
base_model: google/t5gemma-b-b-ul2
tags:
- text-encoder
- gguf
- sa3.cpp
extra_gated_prompt: >-
  To access this model you must agree to the Gemma Terms of Use at
  https://ai.google.dev/gemma/terms, including the use restrictions in Section 3.2.
extra_gated_fields:
  Name: text
  Email: text
  Country: country
  Organization or Affiliation: text
---

# T5Gemma-b-b-ul2 Encoder + Tokenizer — GGUF (for sa3.cpp)

The **shared** text encoder + tokenizer for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp).
GGUF conversion of the frozen [google/t5gemma-b-b-ul2](https://huggingface.co/google/t5gemma-b-b-ul2)
encoder (encoder-only at inference) that Stable Audio 3 uses to embed text prompts.

This component is **identical across all three SA3 variants** (medium, small-music, small-sfx), so it
lives in its own repo and is fetched once — the per-variant conditioner ships separately in each model
repo. Validated against the PyTorch reference at cosine similarity ~1.0.

## Files

| component | file | notes |
|---|---|---|
| text encoder | `t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf` | encoder weights only (no conditioner) |
| tokenizer | `t5gemma-b-b-ul2-v1.0-vocab.gguf` | Gemma byte-fallback BPE |

Pair these with any SA3 variant repo's DiT + SAME + conditioner:
[medium](https://huggingface.co/betweentwomidnights/stable-audio-3-medium-GGUF) ·
[small-music](https://huggingface.co/betweentwomidnights/stable-audio-3-small-music-GGUF) ·
[small-sfx](https://huggingface.co/betweentwomidnights/stable-audio-3-small-sfx-GGUF).
`tools/download_models.py` fetches this repo automatically alongside whichever variant you pick.

## License

Released under the **Gemma Terms of Use** (see `license_link`). Original weights:
[google/t5gemma-b-b-ul2](https://huggingface.co/google/t5gemma-b-b-ul2).

## Relationship to the original

A **format conversion** (weights → GGUF) for inference in sa3.cpp — no retraining. See
[sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md).
