#!/usr/bin/env python3
"""Dump a DiT reference (inputs + velocity) for the GGML port.

Run with the SA3 venv. Builds the DiffusionTransformer from config, loads the
medium `model.model.*` weights, and runs one forward at cfg_scale=1.0 (ARC).
Conditioning is random (the DiT just consumes it as tensors); this isolates the
DiT graph from the conditioning assembly + tokenizer.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

# pure-SDPA path on CPU
import stable_audio_3.models.transformer as _T
for s in ("flash_attn_func","flash_attn_kvpacked_func","flash_attn_varlen_func",
          "index_first_axis","pad_input","unpad_input"):
    setattr(_T, s, None)
from stable_audio_3.models.dit import DiffusionTransformer

PREF = "model.model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inpaint", action="store_true", help="also feed a random local_add_cond (inpaint path)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text())["model"]["diffusion"]
    dmc, obj = cfg["config"], cfg["diffusion_objective"]
    dit = DiffusionTransformer(diffusion_objective=obj, **dmc).eval()
    dit = dit.float()

    sd = {}
    with safe_open(args.src, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PREF):
                sd[k[len(PREF):]] = f.get_tensor(k).float()
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    print(f"loaded DiT: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")

    T, cond_dim = args.frames, dmc["cond_token_dim"]
    io = dmc["io_channels"]
    x = torch.randn(1, io, T)
    t = torch.tensor([args.t])
    cross = torch.randn(1, 257, cond_dim)      # [prompt(256) + seconds(1)]
    glob  = torch.randn(1, cond_dim)
    local_dim = dmc.get("local_add_cond_dim", 0) or 0   # 257 for inpaint
    local = torch.randn(1, local_dim, T) if (local_dim and args.inpaint) else None

    with torch.no_grad():
        tfeat = dit.timestep_features(t[:, None])           # [1, time_dim]
        vel = dit(x, t, cross_attn_cond=cross, global_embed=glob,
                  local_add_cond=local, cfg_scale=1.0)      # [1, io, T]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # raw inputs in ggml layouts
    np.ascontiguousarray(x[0].numpy().T).tofile(out / "dit_x.f32")        # ggml [io, T]
    np.ascontiguousarray(tfeat[0].numpy()).tofile(out / "dit_tfeat.f32")  # [time_dim]
    np.ascontiguousarray(cross[0].numpy()).tofile(out / "dit_cross.f32")  # ggml [cond_dim, 257]
    np.ascontiguousarray(glob[0].numpy()).tofile(out / "dit_global.f32")  # [cond_dim]
    if local is not None:
        np.ascontiguousarray(local[0].numpy().T).tofile(out / "dit_local.f32")  # ggml [local_dim, T]
    np.save(out / "dit_vel.npy", vel[0].numpy())                          # [io, T]
    print(f"velocity shape {tuple(vel.shape)}  range [{vel.min():.3f},{vel.max():.3f}]  t={args.t} T={T}")
    print(f"saved -> {out}/dit_x.f32, dit_tfeat.f32, dit_cross.f32, dit_global.f32, dit_vel.npy")


if __name__ == "__main__":
    sys.exit(main())
