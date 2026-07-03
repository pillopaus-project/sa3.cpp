# loras

adapter files live here. two kinds:

- **converted `.gguf` adapters** - what `sa3-generate --lora <name>` loads (it resolves `lora-<name>-*.gguf`
  from the adapters dir, which defaults to the models dir; point it here with `SA3_ADAPTERS_DIR` /
  `--adapters-dir`). Runtime strength and multi-adapter blending are applied in weight space, not a static merge.
- **source exports** - a trained adapter's `.safetensors` + `.json` (or a raw `.ckpt`). This folder is the
  default `SA3_SOURCE_LORAS_DIR`.

## convert an export to a gguf (no python)

```bash
sa3-lora-convert --in loras/kev --out models/lora-kev-f32.gguf     # reads kev.safetensors + kev.json
```

`sa3-lora-convert` (and the `libsa3` `sa3_convert_lora` C ABI) are a C++ port of `tools/convert_lora.py`, so a
host can convert `.safetensors` adapters **in-process with no Python**. a raw `.ckpt` still needs the python /
pytorch helper `tools/lora_ckpt_export.py` to produce the `.safetensors`/`.json` pair first (a checkpoint is a
torch artifact).

adapter types dora-rows and bora are validated end-to-end against trained checkpoints (cossim 1.0); dora-cols and
the `-xs` variants are formula-validated. See the main README.

**a note about prompts** sa3 loras really like it when you use prompts from your training data. because of this i use helpers to populate 'dice' buttons in downstream apps. you don't have to take advantage of them, but i find it super useful, especially when blending multiple loras. you'll get a prompt from any one of your loaded loras. if you keep your .safetensors file in the same folder as the .txt files you trained with, the http server and libsa3 will automatically register the prompt pools. 
