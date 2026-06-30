# sa3-server — HTTP over the pipeline

A small local HTTP server that wraps the same generation pipeline the CLI uses. Built by any of the
`build` scripts (it's a normal target). Meant to be spawned by a host app (a JUCE/IPlug2 plugin, a
gary4local-style backend) which then POSTs to it — the host never touches the CLI.

## Run

```bash
./build/bin/Release/sa3-server.exe --model medium --encoding f16 --port 8086
# args: --host (default 127.0.0.1) --port (8086) --model <variant> --encoding f16|f32
#       --models-dir DIR (or SA3_MODELS_DIR) --adapters-dir DIR (or SA3_ADAPTERS_DIR)
```

It binds to `127.0.0.1` by default (local only). The model loads lazily on the first `/generate`.

## Endpoints

| method | path | body / result |
|---|---|---|
| `GET`  | `/health`   | `{status, model, encoding, loaded}` |
| `POST` | `/generate` | JSON request (below) → **`audio/wav`** bytes (or `{error}` + 4xx/5xx) |
| `POST` | `/unload`   | frees the model (full VRAM release) → `{status:"unloaded"}` |

**`/generate` request:**
```json
{
  "prompt": "breakcore 140bpm",
  "seconds": 12,                 // or "frames": 128  (frames win if both given)
  "steps": 8,
  "seed": 0,
  "loras": [{"name": "kev", "strength": 1.0}, {"name": "keygen", "strength": 0.8}],
  "keep_models": false,          // default: frugal (free after each gen, reload next) — vst/daw-safe

  // sampling-schedule distribution shift (optional) — mirrors the official SA3 gradio selector.
  // default "LogSNR" reproduces the prior schedule exactly. dist_shift_params overrides the 4
  // per-type params (meaning depends on type); omit it to use that type's defaults:
  "dist_shift": "LogSNR",        // "LogSNR" | "Flux" | "Full" | "None"
  "dist_shift_params": [2000, -6.2, 0.0, 2.0],  // LogSNR:(anchor_length,anchor_logsnr,rate,logsnr_end)
                                                //  Flux:(min_len,max_len,alpha_min,alpha_max)
                                                //  Full:(base_shift,max_shift,min_len,max_len)

  // audio2audio / inpaint (optional) — init_path is a LOCAL wav (the server is localhost):
  "init_path": "in.wav",         // audio2audio source; output length follows it
  "init_noise_level": 0.5,       // a2a strength (sigma_max); 1.0 == text2music
  "inpaint_start": 4.0,          // continuation/inpaint: regenerate [start,end] seconds, keep the rest
  "inpaint_end": 30.0            // also the total output duration (a short clip can extend)
}
```
LoRA `name` resolves to `<adapters-dir>/lora-<name>-*.gguf`; a full `"path"` also works. Set
`keep_models: true` to keep the model resident between requests (lower latency, more VRAM); the server
reloads a clean DiT only when a request's adapter set changes, so strengths can vary per request either way.

## Calling it (platform-correct)

**PowerShell** — `curl` is an alias for `Invoke-WebRequest`, so use `Invoke-RestMethod`:
```powershell
$body = '{"prompt":"breakcore 140bpm","seconds":12,"loras":[{"name":"kev","strength":1.0}]}'
Invoke-RestMethod http://localhost:8086/generate -Method Post -ContentType application/json -Body $body -OutFile song.wav
```

**Git Bash / cmd / macOS / Linux** — real `curl`:
```bash
curl -s -X POST http://localhost:8086/generate -H "Content-Type: application/json" \
  -d '{"prompt":"breakcore 140bpm","seconds":12}' -o song.wav
```
(To use `curl.exe` *from PowerShell*, pass the body from a file — `-d "@body.json"` — inline JSON quoting
is mangled by PowerShell's native-argument handling.)

## Residency / lifecycle

Default **frugal** (`keep_models:false`): the model is freed after each generation and reloaded on the
next request — keeps host-process memory low (good for an embedded/VST context) and makes per-request LoRA
strength correct for free. For a long-running service that wants lowest latency, send `keep_models:true`
and call `POST /unload` from your orchestrator when you need the VRAM back (model-switch, idle, pressure) —
the same pattern as the PyTorch sa3 service.

## Notes

`/generate` covers the full pipeline — text2music, LoRA, audio2audio, and inpaint/continuation. The init
audio is passed as a **local file path** (`init_path`), which is the simple, correct thing for a localhost
backend; a base64/multipart upload path could be added later if a *remote* or fully in-memory client ever
needs it. (For an in-process C-ABI alternative to the server, a `libsa3` shim is possible but deferred —
HTTP already covers the JUCE/IPlug2/gary4local consumers.)
