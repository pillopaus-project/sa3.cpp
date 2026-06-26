#!/usr/bin/env python3
"""Convert the SAME-L decoder subgraph of stable-audio-3-medium to GGUF.

Phase 1 scope: the decode path only (bottleneck.running_std + SAMEDecoder + the
patched-pretransform metadata). Encoder + DiT come in later phases.

Reads tensors lazily by key from the 8.6 GB model.safetensors (we only touch the
~470 `pretransform.model.*` tensors), fuses the weight-normed mapping conv, and
renames to compact GGML-friendly names (GGML_MAX_NAME = 64).

Usage:
  python tools/convert.py --src <model.safetensors> --config <model_config.json> \
                          --out models/sa3-same-l-decoder-f32.gguf
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter

SRC_PREFIX = "pretransform.model."


def fuse_weight_norm(g, v):
    """PyTorch weight_norm (dim=0): w = g * v / ||v|| with norm over all dims but 0."""
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=axes, keepdims=True))
    w = g.astype(np.float64) * v.astype(np.float64) / norm
    return w.astype(np.float32)


def _blk(rest):
    return rest.replace("ff.ff.0.proj", "ff.proj").replace("ff.ff.2", "ff.out")


def rename(src_key):
    """Map a `pretransform.model.*` key to a compact GGUF tensor name, or None to skip."""
    k = src_key[len(SRC_PREFIX):]  # strip prefix
    # bottleneck (running_std for decode; scaling_factor/bias for encode)
    if k == "bottleneck.running_std":   return "ae.running_std"
    if k == "bottleneck.scaling_factor":return "ae.scaling_factor"
    if k == "bottleneck.bias":          return "ae.bias"
    if k == "bottleneck.noise_scaling_factor": return None  # empty (noise_augment_dim=0)
    # decoder.layers: 0=Transpose,1=Linear(256->1536),2=Transpose,3=ResamplingBlock
    if k == "decoder.layers.1.weight":  return "ae.in_proj.weight"
    if k == "decoder.layers.1.bias":    return "ae.in_proj.bias"
    if k == "decoder.layers.3.new_tokens": return "ae.dec.new_tokens"
    if k == "decoder.layers.3.mapping.bias": return "ae.dec.mapping.bias"
    if k == "encoder.layers.0.mapping.bias": return "ae.enc.mapping.bias"
    if k.startswith("decoder.layers.3.transformers."):
        return "ae.dec." + _blk(k[len("decoder.layers.3.transformers."):])
    # encoder.layers: 0=ResamplingBlock,1=Transpose,2=Linear(1536->256),3=Transpose
    if k == "encoder.layers.0.new_tokens": return "ae.enc.new_tokens"
    if k == "encoder.layers.2.weight":  return "ae.out_proj.weight"
    if k == "encoder.layers.2.bias":    return "ae.out_proj.bias"
    if k.startswith("encoder.layers.0.transformers."):
        return "ae.enc." + _blk(k[len("encoder.layers.0.transformers."):])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="medium model.safetensors")
    ap.add_argument("--config", required=True, help="model_config.json")
    ap.add_argument("--out", required=True, help="output .gguf")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    ae_cfg = cfg["model"]["pretransform"]["config"]
    dec = ae_cfg["decoder"]["config"]
    patch = ae_cfg["pretransform"]["config"]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-ae")
    w.add_name("stable-audio-3-medium SAME-L decoder")

    # --- config as KV metadata (single source of truth for the C++ graph) ---
    dim = dec["channels"] * dec["c_mults"][0]      # 256 * 6 = 1536
    w.add_uint32("sa3.ae.dim",               dim)
    w.add_uint32("sa3.ae.latent_dim",        ae_cfg["latent_dim"])           # 256
    w.add_uint32("sa3.ae.dim_heads",         dec["dim_heads"])               # 64
    w.add_uint32("sa3.ae.n_heads",           dim // dec["dim_heads"])        # 24
    w.add_uint32("sa3.ae.depth",             dec["transformer_depths"][0])   # 12
    w.add_uint32("sa3.ae.stride",            dec["strides"][0])              # 16
    w.add_uint32("sa3.ae.sub_chunk",         dec["strides"][0] + 1)          # 17
    w.add_uint32("sa3.ae.output_seg",        dec["strides"][0])              # 16
    w.add_uint32("sa3.ae.sliding_window",    dec["strides"][0] + 1)          # 17 (band +/-)
    w.add_uint32("sa3.ae.ff_mult",           3)                              # AE SwiGLU mult
    w.add_uint32("sa3.ae.sinusoidal_blocks", dec.get("sinusoidal_blocks", [0])[0])  # 8
    w.add_uint32("sa3.ae.out_channels",      dec["out_channels"])            # 512 (=2*256)
    w.add_uint32("sa3.ae.patch_size",        patch["patch_size"])            # 256
    w.add_uint32("sa3.ae.audio_channels",    patch["channels"])             # 2
    w.add_uint32("sa3.ae.downsampling_ratio", ae_cfg["downsampling_ratio"])  # 4096
    w.add_uint32("sa3.ae.sample_rate",       cfg["sample_rate"])             # 44100
    w.add_float32("sa3.ae.rope_base",        10000.0)
    w.add_float32("sa3.ae.qk_norm_eps",      1e-3)

    # --- tensors ---
    n_written, skipped = 0, 0
    with safe_open(args.src, framework="numpy") as f:
        keys = [k for k in f.keys() if k.startswith(SRC_PREFIX)]
        # fuse each weight-normed mapping conv (decoder 1536->512, encoder 512->1536)
        for stem, name in (("decoder.layers.3.mapping", "ae.dec.mapping.weight"),
                           ("encoder.layers.0.mapping", "ae.enc.mapping.weight")):
            g_key, v_key = SRC_PREFIX + stem + ".weight_g", SRC_PREFIX + stem + ".weight_v"
            if g_key in keys and v_key in keys:
                fused = fuse_weight_norm(f.get_tensor(g_key), f.get_tensor(v_key))  # [out,in,1]
                w.add_tensor(name, np.ascontiguousarray(fused.squeeze(-1)))         # -> [out,in]
                n_written += 1

        for k in keys:
            if k.endswith("mapping.weight_g") or k.endswith("mapping.weight_v"):
                continue  # already fused
            name = rename(k)
            if name is None:
                skipped += 1
                continue
            t = np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
            # squeeze singleton dims on small param tensors for clean 1-D shapes
            if name in ("ae.dec.new_tokens", "ae.enc.new_tokens",   # (1,1,1536) -> (1536,)
                        "ae.running_std",                            # (1,) -> (1,)
                        "ae.scaling_factor", "ae.bias"):             # (1,256,1) -> (256,)
                t = t.reshape(-1)
            w.add_tensor(name, t)
            n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}  ({n_written} tensors, skipped {skipped} encoder/encode-only)")
    print(f"  dim={dim} latent={ae_cfg['latent_dim']} depth={dec['transformer_depths'][0]} "
          f"heads={dim//dec['dim_heads']}x{dec['dim_heads']} stride={dec['strides'][0]}")


if __name__ == "__main__":
    sys.exit(main())
