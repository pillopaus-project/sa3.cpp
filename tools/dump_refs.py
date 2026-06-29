#!/usr/bin/env python3
"""Dump PyTorch SAME-L decoder reference activations for validating the GGML port.

Run with a PyTorch env that has stable_audio_3:
  python tools/dump_refs.py \
      --src <model.safetensors> --config <model_config.json> --out refdata --frames 8

Builds just the autoencoder (no DiT), disables the two inference noises
(decoder new_tokens mask_noise + SoftNorm decode noise) for determinism, decodes a
fixed-seed latent on CPU/f32, and saves z + intermediates + audio as .npy.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# stable_audio_3 must be importable (it is, in the sa3 service venv)
from stable_audio_3.factory import create_autoencoder_from_config

# Force the pure-PyTorch SDPA path: flash_attn is CUDA-only, and we want a clean
# f32 CPU reference (with the sliding-window band) to diff the CPU C++ build against.
import stable_audio_3.models.transformer as _T
_T.flash_attn_func = None
_T.flash_attn_kvpacked_func = None
_T.flash_attn_varlen_func = None
_T.index_first_axis = None
_T.pad_input = None
_T.unpad_input = None

SRC_PREFIX = "pretransform.model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--frames", type=int, default=8, help="latent length T")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cpu")
    cfg = json.loads(Path(args.config).read_text())

    ae = create_autoencoder_from_config(cfg["model"]["pretransform"]["config"],
                                        cfg["sample_rate"]).to(dev).eval()

    # load the pretransform.model.* weights (strip prefix), encoder included but unused
    sd = {}
    with safe_open(args.src, framework="pt") as f:
        for k in f.keys():
            if k.startswith(SRC_PREFIX):
                sd[k[len(SRC_PREFIX):]] = f.get_tensor(k).float()
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    print(f"loaded AE: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")

    # --- disable the two inference noises so cossim is meaningful ---
    ae.bottleneck.noise_regularize = False
    resampling_block = ae.decoder.layers[3]            # [Transpose, Linear, Transpose, ResamplingBlock]
    resampling_block.mask_noise = 0.0
    print(f"noises off: bottleneck.noise_regularize={ae.bottleneck.noise_regularize}, "
          f"decoder.mask_noise={resampling_block.mask_noise}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    saved = {}

    def save(name, t):
        a = t.detach().cpu().float().numpy()
        np.save(out / f"{name}.npy", a)
        saved[name] = list(a.shape)
        print(f"  saved {name:20s} {list(a.shape)}")

    # capture intermediates via hooks
    inproj = ae.decoder.layers[1]                      # Linear(256->1536)
    caps = {}
    h1 = inproj.register_forward_hook(lambda m, i, o: caps.__setitem__("after_in_proj", o))
    h2 = resampling_block.register_forward_hook(lambda m, i, o: caps.__setitem__("after_resampling", o))

    with torch.no_grad():
        z = torch.randn(1, cfg["model"]["pretransform"]["config"]["latent_dim"], args.frames, device=dev)
        audio = ae.decode(z)

    h1.remove(); h2.remove()

    save("z", z)
    # raw f32 of z in ggml [latent, T] layout (= z[0].T, C-contiguous) for the C++ tool to read
    np.ascontiguousarray(z[0].detach().cpu().float().numpy().T).tofile(out / "z.f32")
    save("after_in_proj", caps["after_in_proj"])
    save("after_resampling", caps["after_resampling"])
    save("audio", audio)

    # ---- ENCODE path (audio -> latent), reference for the SAME-L encoder ----
    enc_block = ae.encoder.layers[0]                   # [ResamplingBlock, Transpose, Linear, Transpose]
    enc_block.mask_noise = 0.0
    ecaps = {}
    he1 = enc_block.register_forward_hook(lambda m, i, o: ecaps.__setitem__("enc_after_resampling", o))
    he2 = ae.encoder.register_forward_hook(lambda m, i, o: ecaps.__setitem__("enc_latent", o))
    with torch.no_grad():
        L = args.frames * cfg["model"]["pretransform"]["config"]["downsampling_ratio"]  # 8*4096
        in_audio = 0.3 * torch.randn(1, cfg["audio_channels"], L, device=dev)
        z_enc = ae.encode(in_audio)                    # patchify -> encoder -> bottleneck.encode
    he1.remove(); he2.remove()
    # raw f32 of input audio in ggml [L, channels] layout (ne0=L time-fastest = audio[0] (ch,L) C-order)
    np.ascontiguousarray(in_audio[0].detach().cpu().float().numpy()).tofile(out / "enc_audio.f32")
    save("enc_after_resampling", ecaps["enc_after_resampling"])
    save("enc_latent", ecaps["enc_latent"])
    save("z_enc", z_enc)

    (out / "manifest.json").write_text(json.dumps({"frames": args.frames, "seed": args.seed,
                                                    "shapes": saved}, indent=2))
    print(f"manifest -> {out/'manifest.json'}")
    print(f"audio range [{audio.min():.4f}, {audio.max():.4f}]  expect ~(-1,1)")


if __name__ == "__main__":
    sys.exit(main())
