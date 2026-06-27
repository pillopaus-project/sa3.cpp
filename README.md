# stable-audio-3-medium in c++

what's done:

- text2music, cpu-only rn

what's next:

- [x] same-s / stable-audio-3-small-music and sfx
- [x] audio2audio
- [x] inpainting
- [x] loras (lora/dora/bora + xs variants, runtime strength + multi-adapter blending)
- [ ] support cuda/metal/vulkan
- [ ] benchmark generation times and stuff

> note: still want to do more testing to confirm the adapters are working just like the pytorch version
> (dora-rows is validated end-to-end at cossim 1.0; the other adapter types are formula-validated but
> haven't been A/B'd against a trained checkpoint yet)

credits:

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
