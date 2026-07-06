# Non-GPU Training Tests

The canonical training tree keeps a per-feature native CTest suite for the LoRA training path. These tests cover:

- training config parsing and validation failures;
- dataset split parsing, filelist ordering, metadata, missing captions, duplicate basenames, and train/test/evaluation contamination rejection;
- model path resolution;
- MP3 decode smoke coverage via `ffmpeg`;
- SAME, conditioning, DiT, diffusion, and loop compile/graph coverage;
- adapter target inventory, LoRA/DoRA/BoRA/XS shape checks, infeasible-rank rejection, and effective-weight graph shapes;
- AdamW optimizer math;
- GGUF adapter checkpoint write/load round-trip through the inference loader contract;
- held-out generation command construction.

When CMake is configured with testing enabled, the tests are registered with the shared labels `non-gpu;training;lora`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -L training
```

The same executables can still be run directly from `build/bin` or `build-cuda/bin` for targeted debugging.

GPU and full model-asset acceptance remain separate. Real MNESIA fitting and `sa3-generate --lora` held-out WAV generation require local GGUF model files and an appropriate backend; see `docs/TRAINING_ACCEPTANCE.md` for the current acceptance transcript.
