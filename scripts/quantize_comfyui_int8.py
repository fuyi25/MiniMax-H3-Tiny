"""Quantize an H3-Tiny ComfyUI checkpoint to INT8 ConvRot (W8A8), ComfyUI's native ``int8_tensorwise`` format.

    python scripts/quantize_comfyui_int8.py \\
        --src minimax_h3_tiny_17b_ref2va_bf16.safetensors --out minimax_h3_tiny_17b_ref2va_int8_convrot.safetensors

Input is a file written by ``scripts/convert_comfyui.py``. The output loads the same way as the input (the stock
"Load Diffusion Model" node, or the "H3-Tiny Few-Step Loader" for the few-step checkpoint); ComfyUI reads the
per-layer ``comfy_quant`` entries and runs those layers on its INT8 kernels (fastest with a CUDA 13 PyTorch, or
``--enable-triton-backend``).

The four large linears of every DiT block (``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``, ``mlp.fc2``) are
rotated with a 256-wide block Hadamard along the input dimension and stored as INT8 with one scale per output row;
activations are rotated and quantized per row at run time. This is the layout of the INT8 ConvRot files in
Comfy-Org/MiniMax-H3. Everything else (embedders, patch projections, AdaLN, token refiner, final layer, norms) is
copied unchanged.

The FFN width of the 14B (9792) is not a multiple of the group size, so its ``fc2`` input could not be rotated. The
width is zero-padded to the next multiple (9984): zero rows at the end of both halves of ``fc1`` ([gate; value]) and
zero columns in ``fc2``. SwiGLU gives exactly 0 on the padded channels, so the bf16 model is unchanged; ComfyUI reads
the padded width from the tensors. ``--no-pad-ffn`` keeps such ``fc2`` layers in bf16 instead.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANT_RE = re.compile(r"^blocks\.\d+\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.weight$")
FC1_RE = re.compile(r"^blocks\.\d+\.mlp\.fc1\.(weight|bias)$")
FC2_RE = re.compile(r"^blocks\.\d+\.mlp\.fc2\.weight$")


def pad_ffn(key: str, t: torch.Tensor, group: int) -> torch.Tensor:
    """Zero-pad the FFN width to a multiple of ``group`` (both fc1 halves, fc2 input columns)."""
    if FC1_RE.match(key):
        width = t.shape[0] // 2
        extra = -width % group
        if extra:
            gate, value = t.split(width, dim=0)
            pad = t.new_zeros((extra,) + tuple(t.shape[1:]))
            t = torch.cat([gate, pad, value, pad], dim=0)
    elif FC2_RE.match(key):
        extra = -t.shape[1] % group
        if extra:
            t = torch.cat([t, t.new_zeros(t.shape[0], extra)], dim=1)
    return t


def build_hadamard(size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard (Kronecker powers of a 4x4 block), as used by ComfyUI's ConvRot kernels."""
    if size < 4 or math.log(size, 4) % 1:
        raise ValueError(f"group size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def quantize_convrot(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``W @ blockdiag(H)^T`` then symmetric per-row INT8: returns (int8 [out, in], float32 scale [out, 1])."""
    out_f, in_f = weight.shape
    h = build_hadamard(group, weight.device, weight.dtype)
    w = torch.matmul(weight.reshape(out_f, in_f // group, group), h.T).reshape(out_f, in_f)
    scale = (w.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    q = (w / scale.to(w.dtype)).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="bf16 ComfyUI checkpoint from convert_comfyui.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--group", type=int, default=256, help="ConvRot group size (power of 4)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pad-ffn", action="store_true", help="keep fc2 in bf16 instead of padding the FFN width")
    args = ap.parse_args()

    tag = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": args.group}).encode()
    tag = torch.frombuffer(bytearray(tag), dtype=torch.uint8)
    out, kept = {}, []
    with safe_open(args.src, "pt") as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():
            t = f.get_tensor(key)
            if not args.no_pad_ffn:
                t = pad_ffn(key, t, args.group)
            if not QUANT_RE.match(key):
                out[key] = t
                continue
            if t.shape[1] % args.group:
                kept.append(f"{key} {tuple(t.shape)}")
                out[key] = t
                continue
            q, scale = quantize_convrot(t.to(args.device, torch.float32), args.group)
            layer = key[: -len(".weight")]
            out[key] = q.cpu()
            out[f"{layer}.weight_scale"] = scale.cpu()
            out[f"{layer}.comfy_quant"] = tag.clone()

    n = sum(k.endswith(".comfy_quant") for k in out)
    if "h3tiny" in metadata:
        info = json.loads(metadata["h3tiny"])
        info.update(weights="int8_convrot", convrot_groupsize=args.group, int8_linears=n)
        metadata["h3tiny"] = json.dumps(info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=metadata)
    print(f"{args.out}: {n} INT8 linears, {args.out.stat().st_size / 1e9:.2f} GB")
    for k in kept:
        print(f"  kept bf16 (input width not a multiple of {args.group}): {k}")


if __name__ == "__main__":
    main()
