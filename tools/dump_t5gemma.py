#!/usr/bin/env python3
"""Dump a T5Gemma-encoder reference (input_ids + last_hidden_state) for the GGML port.

Run with the SA3 venv (has transformers with T5Gemma):
  .../services/sa3/env/Scripts/python.exe tools/dump_t5gemma.py \
      --model <.../t5gemma-b-b-ul2> --out refdata --prompt "..."

f32 on CPU, eager attention (so logit soft-capping is actually applied).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import T5GemmaEncoderModel, AutoTokenizer, AutoConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the t5gemma-b-b-ul2 dir")
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--prompt", default="Upbeat funk groove with slap bass and bright horns")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.is_encoder_decoder = False
    model = T5GemmaEncoderModel.from_pretrained(args.model, config=cfg,
                                                torch_dtype=torch.float32,
                                                attn_implementation="eager").eval()

    enc = tok([args.prompt], truncation=True, max_length=args.max_length,
              padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"]            # [1, 256]
    attn = enc["attention_mask"]            # [1, 256]
    n_tok = int(attn.sum().item())
    print(f"prompt: {args.prompt!r}\nnon-pad tokens: {n_tok} / {args.max_length}")
    print("first ids:", input_ids[0, :min(n_tok, 16)].tolist())

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attn.bool())["last_hidden_state"]  # [1,256,768]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    input_ids[0].to(torch.int32).numpy().tofile(out / "t5g_input_ids.i32")
    attn[0].to(torch.int32).numpy().tofile(out / "t5g_attn.i32")
    np.save(out / "t5g_hidden.npy", hidden[0].float().numpy())   # [256,768]
    (out / "t5g_manifest.json").write_text(json.dumps(
        {"prompt": args.prompt, "n_tok": n_tok, "max_length": args.max_length}, indent=2))
    print(f"hidden shape {tuple(hidden.shape)}  range [{hidden.min():.3f},{hidden.max():.3f}]")
    print(f"saved -> {out}/t5g_input_ids.i32, t5g_hidden.npy")


if __name__ == "__main__":
    sys.exit(main())
