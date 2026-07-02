# Embedding sa3 in a host app (libsa3)

`libsa3` is a tiny C ABI over the same generation pipeline the CLI and server use, so you can call
sa3 **in-process** from a host — a JUCE / IPlug2 plugin, a game, any C or C++ program — with no CLI
subprocess and no HTTP. The header is [`src/libsa3.h`](../src/libsa3.h); the pipeline it wraps is
[`src/sa3_pipeline.h`](../src/sa3_pipeline.h).

> ⚠️ **Status: the C ABI is tested; the IPlug2 integration below is NOT.**
> `libsa3` itself works end-to-end — [`tools/sa3-libtest.c`](../tools/sa3-libtest.c) drives the full
> `init → generate → write wav` sequence through the pure-C ABI and is part of the build. But **we have
> yet to actually wire it into an IPlug2 plugin and see what goes wrong.** The build/threading steps here
> are the *from-the-outside* plan; the dead-simple demo project (coming) is what will shake out the real
> `.vcxproj` wiring, the DLL-copy step, and the audio-thread hand-off. Treat this as a starting map, not a
> guarantee — and if you hit a snag, fix it here.

## The API

Full contract is in [`src/libsa3.h`](../src/libsa3.h). The shape:

```c
sa3_context* ctx = sa3_init(&cfg, err, sizeof err);        // load the models (blocks ~seconds)
sa3_audio audio = {0};
sa3_generate(ctx, &req, &audio, err, sizeof err);          // prompt -> planar float samples (blocks)
// or: sa3_generate_ex(ctx, &req_ex, &audio, err, sizeof err); // audio2audio / inpaint with raw planar init audio
// ... use audio.samples (planar: samples[ch*n_samp + s]), audio.seed is the resolved seed ...
sa3_free_audio(&audio);
sa3_unload(ctx);   // optional: drop the models (free VRAM), keep ctx; next generate reloads
sa3_free(ctx);     // destroy
```

- **Zero-initialize** `sa3_config` / `sa3_request` / `sa3_request_ex` (`memset 0` or `{0}`); a `0`/`NULL` field means "default".
- **Errors** don't throw — failures return `NULL`/non-zero and fill your `err` buffer.
- **Not reentrant** — serialize `sa3_generate` calls on a given context.
- **Ownership** — `sa3_generate` allocates `audio.samples`; release it with `sa3_free_audio` (same
  library/CRT — don't `free()` it yourself across the DLL boundary).

For audio2audio or inpaint/continuation, use `sa3_request_ex`. Put your normal request fields under
`req_ex.request`, then provide planar float source audio:

```c
sa3_request_ex req = {0};
req.request.prompt = "turn this into a bright synth loop";
req.request.steps = 8;
req.request.seed = -1;
req.request.keep_models = 1;
req.init_audio.mode = SA3_INIT_AUDIO_A2A;       // or SA3_INIT_AUDIO_INPAINT
req.init_audio.samples = planar;                // samples[ch * n_samp + s]
req.init_audio.n_samp = n_samp;
req.init_audio.n_ch = n_ch;
req.init_audio.sample_rate = sample_rate;       // resampled inside libsa3 if needed
req.init_audio.init_noise_level = 0.85f;        // <=0 uses 0.85

// Inpaint/continue only:
req.init_audio.mode = SA3_INIT_AUDIO_INPAINT;
req.init_audio.inpaint_start = 8.0f;
req.init_audio.inpaint_end = 30.0f;             // can extend beyond source length

sa3_generate_ex(ctx, &req, &audio, err, sizeof err);
```

The canonical, *working* usage example is [`tools/sa3-libtest.c`](../tools/sa3-libtest.c) — copy from there.

## What you build and ship

Build any backend (`cpu` / `cuda` / `vulkan` / …) — `libsa3` is a normal target, so the build scripts
produce it alongside the tools:

```bash
./build.cmd cuda      # or: cmake --build build-cuda --target sa3_shared --config Release
```

| Artifact | Role | Location (Windows / MSVC) |
|---|---|---|
| `src/libsa3.h` | the API you `#include` | in the repo |
| `sa3.lib` | import lib you **link** | `build-<backend>/Release/sa3.lib` |
| `sa3.dll` | runtime lib you **ship** | `build-<backend>/bin/Release/sa3.dll` |
| `ggml*.dll` | ggml backends `sa3.dll` needs | same `bin/Release/` (`ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda`, …) |
| the gguf model set | weights loaded at runtime | any folder; pass its **absolute** path |

On Unix/macOS the library is `libsa3.so` / `libsa3.dylib` (link it directly; ship it + the `libggml*` set).

## Wiring into an IPlug2 project (Windows / Visual Studio)

IPlug2 plugins build per-format from a VS solution. In the plugin's `.vcxproj` (or a shared `.props`):

1. **Include** — add `…\sa3.cpp\src` to *Additional Include Directories*.
2. **Link** — add `…\sa3.cpp\build-cuda\Release\sa3.lib` to *Additional Dependencies*.
3. **Ship the DLLs** — a post-build step copying `sa3.dll` + every `ggml*.dll` into the plugin's output
   dir. For a VST3 that's `MyPlugin.vst3\Contents\x86_64-win\`; for the standalone, next to the `.exe`.
   Windows resolves a plugin's dependent DLLs from the binary's own folder, so they must sit beside
   `sa3.dll`.
4. **Models** — the DAW's working directory is unknown, so **don't** rely on the `"models"` default. Pass
   an absolute `sa3_config.models_dir` (bundle the ggufs with your installer, resolve the path at runtime).

## Threading — the part that matters

`sa3_init` and `sa3_generate` **block for seconds**. Two hard rules:

- **Never** call either from `ProcessBlock` (the audio thread) — you'll stall the DAW and get dropouts.
- **Never** call `sa3_init` in the plugin constructor — the host runs that during plugin *scan*, and a
  multi-second load there makes the DAW look hung.

So everything model-related runs on a worker thread you own; the audio thread only ever reads a finished
buffer.

```cpp
// plugin members
sa3_context*        mCtx = nullptr;
std::thread         mWorker;
std::atomic<bool>   mBusy{false};
std::mutex          mSwapMtx;
std::vector<float>  mReady;          // last finished render, planar [ch*n_samp + s]
std::atomic<bool>   mHaveNew{false};

void GenerateAsync(std::string prompt) {
    if (mBusy.exchange(true)) return;                 // one generation at a time
    if (mWorker.joinable()) mWorker.join();
    mWorker = std::thread([this, prompt = std::move(prompt)] {
        char err[512] = {0};
        if (!mCtx) {                                  // lazy init on the worker, not the ctor
            sa3_config cfg{};
            cfg.models_dir = "C:/path/to/models";     // absolute
            cfg.variant = "medium"; cfg.encoding = "f16";
            mCtx = sa3_init(&cfg, err, sizeof err);
            if (!mCtx) { /* surface err to the UI */ mBusy = false; return; }
        }
        sa3_request req{};                            // {0} => defaults
        req.prompt = prompt.c_str();
        req.frames = 128; req.steps = 8; req.seed = -1;
        req.cfg_scale = 1.0f; req.duration_padding_sec = 6.0f; req.keep_models = 1;
        req.on_progress = [](void*, const char* st, int s, int t, float f) { /* post % to UI */ };

        sa3_audio a{};
        if (sa3_generate(mCtx, &req, &a, err, sizeof err) == 0) {
            std::vector<float> buf(a.samples, a.samples + (size_t)a.n_samp * a.n_ch);
            { std::lock_guard<std::mutex> lk(mSwapMtx); mReady.swap(buf); }
            mHaveNew = true;                          // audio thread picks this up
            sa3_free_audio(&a);
        } /* else surface err */
        mBusy = false;
    });
}
```

The audio thread swaps `mReady` into its playback buffer when `mHaveNew` flips (sampler-style), then plays
it in `ProcessBlock`. On teardown: `if (mWorker.joinable()) mWorker.join(); sa3_free(mCtx);`.

## Caveats you'll actually feel

- **VRAM per instance.** Each plugin instance that calls `sa3_init` loads its **own** model (several GB
  for medium). Ten instances = ten models. This host-contention reality is exactly *why* the networked
  path (one shared `sa3-server`, see [SERVER.md](SERVER.md)) exists — libsa3 is for single-instance /
  embedded / no-network cases.
- **Load latency.** `sa3_init` is multi-second — show a "loading model…" state, and consider
  `sa3_unload(ctx)` when the plugin is hidden/idle to give the DAW its VRAM back (the next generate reloads).
- **`keep_models`.** `1` = snappy repeat gens but holds VRAM; `0` = frees after each gen (light, ~1s
  reload next time). In a DAW, resident + explicit `sa3_unload` on close is the usual pick.
- **Which backend to ship.** The CUDA `sa3.dll` needs a CUDA GPU + driver on the user's machine. For a
  portable plugin, ship a **CPU or Vulkan** build instead (slower, runs anywhere) — same `sa3.dll` name,
  different `ggml-*.dll` set.

## When the demo lands

Once the dead-simple IPlug2 project is built and running, update this file: confirm the exact `.vcxproj`
include/lib/copy steps, the VST3 vs standalone output paths, and any threading or DLL gotcha that bit us —
and drop the "not tested" banner.
