# sa3.cpp

A portable C++17 / [GGML](https://github.com/ggml-org/ggml) implementation of
**Stable Audio 3** — focused on the `medium` DiT + `SAME-L` autoencoder (the ARC ping-pong variant),
with text2music, audio2audio, and inpainting/continuation, plus PyTorch-compatible LoRA loading.

Modeled on [`acestep.cpp`](https://github.com/ServeurpersoCom/acestep.cpp), a GGML port of the closely
related ACE-Step 1.5 stack.

> **Status: exploration / scaffolding.** No code yet — see the design docs below.

## Docs
- [docs/ARCHITECTURE-NOTES.md](docs/ARCHITECTURE-NOTES.md) — how SA3 is built and how it maps to GGML.
- [docs/ROADMAP.md](docs/ROADMAP.md) — the phased plan from spike to working renderer.

## Reference material (`refs/`, git-ignored)
- `refs/stable-audio-3/` — Stability AI's PyTorch implementation (what we port).
- `refs/acestep.cpp/` — the GGML porting playbook.

## Key idea
SA3's whole stack (T5Gemma text encoder → DiT → SAME-L autoencoder) is transformer-based —
attention + linear + norm + plain conv1d. Unlike acestep's convolutional Oobleck VAE, it needs
no custom GGML ops, so we target **vanilla GGML**.
