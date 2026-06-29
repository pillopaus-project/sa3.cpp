# hip / rocm backend — UNTESTED, looking for an AMD tester 🙏

status: **scaffolded but never run.** the dev machines have NVIDIA + Apple GPUs, no AMD, so the
ROCm/HIP backend is wired up (`SA3_HIP` cmake option, `./build.sh hip`) but has not been built or
benchmarked on real hardware.

> **if you're on ROCm — could you pretty please test this backend and report gen times?**
> The whole sa3 stack is vanilla ggml with zero custom ops, and it already runs on CUDA/Vulkan/Metal,
> so HIP *should* just work (ggml's HIP backend is the CUDA kernels hipified). We just can't confirm it
> or measure it. **Open an issue or PR** with: did it build, did it generate, the device line it picked,
> and a few gen times. That closes the last backend. Thank you! — and if ROCm is painful, the **Vulkan**
> backend ([docs/VULKAN.md](VULKAN.md)) also runs on AMD with no SDK fuss and is a fine fallback.

## 0. prerequisites (Linux)

- A [ROCm-supported AMD GPU](https://rocm.docs.amd.com/) + a working **ROCm** install (`rocminfo` runs).
- cmake + the HIP toolchain (`hipcc`). ROCm on Windows is very limited — use Linux.

## 1. build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp
./build.sh hip            # -> build-hip/   (SA3_HIP=ON -> GGML_HIP)
```

if the build can't detect your GPU arch, pass it explicitly (find it via `rocminfo | grep gfx`):

```bash
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSA3_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build-hip --config Release -j$(nproc)
# common arches: gfx1100/gfx1101 (RDNA3, RX 7900/7800), gfx1030 (RDNA2, RX 6900), gfx90a (MI200)
```

## 2. get a model set

same as the other backends (the repos are private for now — auth, see [METAL.md](METAL.md) step 2):

```bash
python3 -m pip install huggingface_hub
hf auth login
python3 tools/download_models.py --variant medium --encoding f16
```

## 3. validate (mirror the Vulkan/Metal write-ups)

`make_backend` (src/gguf_model.h) should auto-pick the AMD GPU (ggml registers it as a `ROCm` device);
`SA3_DEVICE=cpu` forces CPU for the A/B. Then:

1. **determinism** — run the same gen twice on HIP, expect byte-identical output (cosine 1.0).
2. **HIP-vs-CPU, same binary** — generate once on HIP, once with `SA3_DEVICE=cpu`, same seed/prompt, and
   compare with **rms-envelope + log-mag-spectrogram cosine, not raw waveform cosine** (see the Vulkan
   lesson — flash/tensor-core matmuls drift at the sample level but stay perceptually identical). The
   ready-to-run python snippet is in [METAL.md](METAL.md) §4.
3. **speed** — time a 12s medium f16 gen (reference: CUDA ~3s, Vulkan ~4s, M4 ~6s on our hardware), and a
   couple of long ones (30s / 60s / 120s) so we can see the decoder stays linear.
4. **flash attention** — HIP reuses the CUDA flash kernels, so `SA3_FLASH_ATTN=1` (full) should build and
   help the decoder (~25%, like CUDA). `SA3_SAME_FLASH_ATTN=local` is auto-downgraded to `full` on
   ROCm/HIP (its compact shape aborts the shared CUDA flash kernel — see [BENCHMARKS.md](BENCHMARKS.md)).

## 4. what to report

backend line it picked, build/gen success, the parity numbers from §3.2, the speeds from §3.3, and any op
that *errors* (HIP coverage should match CUDA since it's the same kernels, but if something's unsupported
the run fails loudly rather than going silently wrong — tell us which op). when it's green we'll write up
results here like VULKAN.md / METAL.md and check off the last backend.
