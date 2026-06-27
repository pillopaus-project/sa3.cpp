#!/usr/bin/env python3
"""Quantize an sa3.cpp gguf to F16 (post-hoc, model-agnostic).

Converts the big 2-D weight matrices to F16 (halves VRAM, enables tensor cores) while
keeping everything that must stay F32 in F32:
  - 1-D tensors (norms, biases, magnitudes, scaling factors, the running_std scalar),
  - any tensor whose name contains "tokens" (memory_tokens / new_tokens are concatenated
    with F32 activations, so their dtype must match).
The C++ graph runs F16 weights x F32 activations natively (ggml_mul_mat). KV metadata is
copied verbatim. Run with the converter .venv (has gguf):

  .venv/Scripts/python.exe tools/quantize_gguf.py --in models/sa3-dit-f32.gguf --out models/sa3-dit-f16.gguf
"""
import argparse, sys
from pathlib import Path
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def field_value(field):
    """Reconstruct a KV field's python value + its GGUFValueType from a GGUFReader field."""
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        et = field.types[1]
        vals = [field.parts[i] for i in field.data]
        if et == GGUFValueType.STRING:
            return [bytes(v).decode("utf-8") for v in vals], t, et
        return [np.asarray(v).item() for v in vals], t, et
    if t == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8"), t, None
    return np.asarray(field.parts[field.data[0]]).item(), t, None


def to_f16(name, ndim):
    return ndim == 2 and "tokens" not in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = GGUFReader(args.inp)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    w = GGUFWriter(args.out, arch=arch)

    # copy KV (skip the ones GGUFWriter sets itself from arch)
    for key, field in r.fields.items():
        if key in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        val, t, et = field_value(field)
        if t == GGUFValueType.ARRAY:
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, t)

    n16 = n32 = 0
    for tensor in r.tensors:
        arr = np.array(tensor.data)            # already shaped, current dtype
        if to_f16(tensor.name, arr.ndim) and arr.dtype != np.float16:
            w.add_tensor(tensor.name, arr.astype(np.float16)); n16 += 1
        else:
            w.add_tensor(tensor.name, np.ascontiguousarray(arr.astype(np.float32))); n32 += 1

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}  ({n16} tensors -> F16, {n32} kept F32)")


if __name__ == "__main__":
    sys.exit(main())
