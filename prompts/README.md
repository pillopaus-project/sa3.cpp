# prompt pools

i don't really enjoy typing prompts, so i prefer to populate "dice" buttons in downstream apps — this folder
(the default `SA3_PROMPTS_DIR`) is where those pools live:

- `defaults.json` — the fallback pool, used when no lora-specific prompts are found.
- `<lora-name>.json` (e.g. `kev.json`, `keygen.json`) — a per-lora pool, used when that lora is loaded.

with sa3, loras (especially the dora adapters i've trained) seem to really like prompts from their training
distribution. but loras generate just fine with no prompt at all — especially when several are loaded.

that said, `defaults.json` could use some help: this model kind of has its own prompt "language" and i haven't
gotten around to rewriting the defaults to align with it (i end up just playing with my loras instead). the
upstream prompting guide is the place to start:

https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md
