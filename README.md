# stable-audio-3 in c++

a portable c++/ggml port of stable audio 3 — prompt string → music, no pytorch in the loop.
runs on cpu or cuda; weights in f32 or f16. every component is validated against the pytorch
reference at cosine similarity ~1.0.

what works:

- text2music, audio2audio, inpainting / continuation
- both sizes: medium (same-l) + small-music (same-s)
- lora / dora / bora adapters (+ xs variants) — runtime strength + multi-adapter blending,
  applied in weight space (not a static merge)
- cuda backend + fp16 — medium generation ~3.5s end-to-end on an 8gb laptop 5070, and long-form
  (sliding-window decoder) scales linearly. see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

what's next:

- [x] same-s / stable-audio-3-small-music and sfx
- [x] audio2audio
- [x] inpainting
- [x] loras (lora/dora/bora + xs variants, runtime strength + multi-adapter blending)
- [x] cuda backend + fp16
- [x] benchmark generation times and stuff ([docs/BENCHMARKS.md](docs/BENCHMARKS.md))
- [ ] vulkan / metal backends

> note: still want to do more testing to confirm the adapters are working just like the pytorch version
> (dora-rows is validated end-to-end at cossim 1.0; the other adapter types are formula-validated but
> not yet a/b'd against a trained checkpoint — a bora training run is in progress for that)

credits:

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
