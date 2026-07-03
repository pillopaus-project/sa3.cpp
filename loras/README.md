# LoRAs

Adapter files live here. Two kinds:

- **Converted `.gguf` adapters** — what `sa3-generate --lora <name>` loads (it resolves `lora-<name>-*.gguf`
  from the adapters dir, which defaults to the models dir; point it here with `SA3_ADAPTERS_DIR` /
  `--adapters-dir`). Runtime strength and multi-adapter blending are applied in weight space, not a static merge.
- **Source exports** — a trained adapter's `.safetensors` + `.json` (or a raw `.ckpt`). This folder is the
  default `SA3_SOURCE_LORAS_DIR`.

## Convert an export to a gguf (no Python)

```bash
sa3-lora-convert --in loras/kev --out models/lora-kev-f32.gguf     # reads kev.safetensors + kev.json
```

`sa3-lora-convert` (and the `libsa3` `sa3_convert_lora` C ABI) are a C++ port of `tools/convert_lora.py`, so a
host can convert `.safetensors` adapters **in-process with no Python**. A raw `.ckpt` still needs the Python /
PyTorch helper `tools/lora_ckpt_export.py` to produce the `.safetensors`/`.json` pair first (a checkpoint is a
torch artifact).

Adapter types dora-rows and bora are validated end-to-end against trained checkpoints (cossim 1.0); dora-cols and
the `-xs` variants are formula-validated. See the main README.

Adapter files are gitignored; only this README is tracked.
