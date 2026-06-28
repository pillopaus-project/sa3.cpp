# metal backend — validation checklist (for the m4)

status: **NOT YET BUILT.** this is the plan for bringing up + validating the ggml Metal backend
on an Apple-silicon Mac, mirroring how we did Vulkan (see [VULKAN.md](VULKAN.md)). the whole sa3
stack is vanilla ggml with zero custom ops, so — like cuda and vulkan — Metal should "just work"
with no graph changes; the job is to build it and confirm parity + speed on-device.

## 0. prerequisites

- Xcode command-line tools (`xcode-select --install`) — provides clang + the Metal toolchain.
- cmake on PATH (`brew install cmake`).
- Python for the converters/checks (`pip install huggingface_hub numpy`).

## 1. build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp
./build.sh metal        # -> build-metal/   (SA3_METAL=ON; ggml also auto-picks Accelerate BLAS)
```

ggml embeds the Metal shaders at build time, so there's no separate SDK to install (unlike Vulkan's
glslc). if the build can't find the Metal framework, confirm the Xcode CLT are installed.

## 2. get a model set

once the HF repos are up: `python tools/download_models.py --variant medium --encoding f16`.
(until then, convert locally per [DISTRIBUTION.md](DISTRIBUTION.md) if you have the source weights.)

## 3. device + memory notes

- one Apple GPU, so `make_backend` (src/gguf_model.h) picks it with no fuss; `SA3_DEVICE=cpu` forces
  the CPU backend for the A/B below. `SA3_GPU` is there if you ever need to pin a device.
- **unified memory is a big deal here:** M-series shares RAM with the GPU, so the 8GB-discrete VRAM
  thrash we fought on the 5070 (and the f32-medium OOM under Vulkan) should NOT happen — f32 medium
  likely fits. try f16 first (production path) but f32 is worth a shot for an exact-parity check.

## 4. validate (the right way)

**the key lesson from Vulkan:** raw waveform cosine is the WRONG end-to-end metric for a backend whose
matmuls aren't bit-exact. Vulkan's tensor-core path made full-gen raw cosine 0.72 vs CPU while the audio
was perceptually identical (log-mag spectrogram cosine 0.95, rms envelope ~perfect). Metal's matmul may
likewise differ slightly from CPU. so:

1. **determinism** — run the same gen twice on Metal, expect byte-identical output (cosine 1.0). if it
   drifts run-to-run, Metal is using nondeterministic reductions; flag it.
2. **GPU-vs-CPU, same binary** — generate once on Metal, once with `SA3_DEVICE=cpu`, same seed/prompt:
   ```bash
   B=./build-metal/bin/sa3-generate
   ARGS="--tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
         --cond models/stable-audio-3-medium-conditioner-v1.0-F32.gguf \
         --dit models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
         --same models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
         --prompt 'upbeat funk groove with slap bass' --frames 128 --steps 8 --seed 0"
   $B $ARGS --out metal.wav
   SA3_DEVICE=cpu $B $ARGS --out cpu.wav
   ```
   then compare with **envelope + log-mag-spectrogram cosine**, not raw waveform cosine:
   ```python
   import wave, numpy as np
   def rd(p):
       w=wave.open(p,'rb'); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float64); w.close(); return a
   a,b=rd('metal.wav'),rd('cpu.wav'); n=min(len(a),len(b)); a,b=a[:n],b[:n]
   def logmag(x,win=2048,hop=512):
       f=np.array([np.abs(np.fft.rfft(x[i:i+win]*np.hanning(win))) for i in range(0,len(x)-win,hop)])
       return np.log1p(f).ravel()
   def env(x,w=2205): return np.sqrt(np.array([np.mean(x[i:i+w]**2) for i in range(0,len(x)-w,w)]))
   cos=lambda u,v:float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-12))
   print('raw waveform cosine :', cos(a,b), '  <- expected to be low if matmuls differ; not a failure')
   print('rms envelope cosine :', cos(env(a),env(b)))
   print('log-mag spec cosine :', cos(logmag(a),logmag(b)))
   ```
   **pass:** envelope + log-mag cosine ~0.95+ and the two clips sound the same. (if f32 medium fits in
   unified memory, an f32 Metal-vs-CPU run may even land near raw cosine 1.0 — worth recording.)
3. **isolated-forward parity (optional, stronger)** — if you bring over `refdata/` inputs from the
   Windows box, `sa3-dit` / `sa3-codec` give Metal-vs-CPU cosine on a single forward (these read ~1.0
   on CPU/CUDA; Vulkan's coopmat path read 0.9999). not required if the end-to-end perceptual check passes.
4. **speed** — time a 12s medium f16 gen; for reference CUDA ~3.5s, Vulkan ~4.1s on a 5070 laptop.

## 5. risks to watch

same op set as Vulkan, which all worked: cast, rms_norm, soft_max_ext with a 3D mask, 4D batched
mul_mat, the f16 `ggml_cpy`, and the SAME-L local-attn concat/views. ggml-metal is mature, so coverage
should be fine — but if any op is unsupported the run errors (not silently wrong); note which and we
either tweak the graph or fall back. when it's green, write up results here like VULKAN.md and update
the roadmap (Metal = the last backend).
