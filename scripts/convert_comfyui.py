"""Convert an H3-Tiny (or official MiniMax-H3) diffusers transformer to a single ComfyUI checkpoint.

    python scripts/convert_comfyui.py --src /models/MiniMax-H3-Tiny-17B-Pruned-Ref2VA \\
        --out /models/comfyui/minimax_h3_tiny_17b_ref2va_bf16.safetensors

The output loads with ComfyUI's stock "Load Diffusion Model" node and the native MiniMax H3 workflows: put it in
``ComfyUI/models/diffusion_models/``. ComfyUI reads the architecture from the tensors (block count, FFN width), so a
pruned depth needs no extra config.

Key layout follows ComfyUI's MiniMax H3 model (``blocks.N.attn.qkv_proj``, ``mlp.fc1``, ``final_layer.*``, ...).
AdaLN is written in ComfyUI's curve form: ``silu(time_embedder(t))`` for t in [0, 1] lies in a low-dimensional affine
subspace, so every AdaLN projection (our rank-8 factorization or the official full one) is re-expressed exactly over
a shared ``--rank``-dim basis of that curve, sampled on ``--grid`` points (``adaln_t_table``). With the default rank 16
the AdaLN outputs match the diffusers model to ~1e-10 relative error; the time embedder is then not needed.
``--adaln full`` instead writes dense ``[96768, 2688]`` projections plus the time embedder (about 0.5 GB more per block).

Few-step students carry an interval branch (``r_time_embedder``): AdaLN then depends on both t and the step endpoint
r, so it cannot be tabulated over t. Those convert with ``--adaln lowrank`` (chosen automatically): the time embedders
and the rank-8 AdaLN factors are kept as they are, and the file loads with the "H3-Tiny Few-Step Loader" node from
``comfyui/`` in this repository instead of the stock loader. A token refiner wider than the pruned block FFN (the 14B keeps 14336 against 9792)
loads as is: ComfyUI assigns the stored tensors, and we checked that its output matches a build at the true widths.
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

TOP_LEVEL = {
    "proj_in": "video_patch_proj",
    "audio_proj_in": "audio_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "norm_out.norm": "final_layer.norm",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "token_refiner.final_norm": "token_refiner.final_norm",
}
IN_BLOCK = {
    "norm1": "norm1",
    "norm2": "norm2",
    "attn.norm_q": "attn.q_norm",
    "attn.norm_k": "attn.k_norm",
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}
BLOCK_RE = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


class Source:
    """Lazy tensor access to a diffusers transformer directory (sharded or single file)."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.config = json.loads((directory / "config.json").read_text())
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            single = directory / "diffusion_pytorch_model.safetensors"
            with safe_open(single, "pt") as f:
                self.where = {k: single.name for k in f.keys()}
        self._open: dict[str, object] = {}

    def keys(self):
        return self.where.keys()

    def __contains__(self, key):
        return key in self.where

    def __getitem__(self, key) -> torch.Tensor:
        name = self.where[key]
        if name not in self._open:
            self._open[name] = safe_open(self.dir / name, "pt")
        return self._open[name].get_tensor(key)


def time_curve(src: Source, grid: int) -> torch.Tensor:
    """``silu(time_embedder(t))`` on ``grid`` points of t in [0, 1], ComfyUI's convention (fp64)."""
    w1, b1 = src["time_embedder.linear_1.weight"].double(), src["time_embedder.linear_1.bias"].double()
    w2, b2 = src["time_embedder.linear_2.weight"].double(), src["time_embedder.linear_2.bias"].double()
    half = w1.shape[1] // 2
    t = torch.linspace(0.0, 1.0, grid, dtype=torch.float64)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    temb = torch.nn.functional.silu(emb @ w1.T + b1) @ w2.T + b2
    return torch.nn.functional.silu(temb)


def rope_inv_freq(config: dict) -> torch.Tensor:
    dim = config.get("rope_freq_dim", 16)
    theta = config.get("rope_theta", 10000.0)
    return 1.0 / (theta ** (torch.arange(0, 2 * dim, 2, dtype=torch.float64) / (2 * dim))).float()


def adaln_affine(src: Source, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """AdaLN projection of ``silu(temb)`` as one affine map (W, b) in fp64, from a full or a rank-r factorized layer."""
    if f"{prefix}.linear.weight" in src:
        return src[f"{prefix}.linear.weight"].double(), src[f"{prefix}.linear.bias"].double()
    w1, b1 = src[f"{prefix}.linear_1.weight"].double(), src[f"{prefix}.linear_1.bias"].double()
    w2, b2 = src[f"{prefix}.linear_2.weight"].double(), src[f"{prefix}.linear_2.bias"].double()
    return w2 @ w1, w2 @ b1 + b2


def adaln_low_rank(src: Source, prefix: str, basis: torch.Tensor, mean: torch.Tensor):
    """(W B^T, W mu + b) without materializing W for factorized layers."""
    if f"{prefix}.linear.weight" in src:
        w, b = src[f"{prefix}.linear.weight"].double(), src[f"{prefix}.linear.bias"].double()
        return w @ basis.T, w @ mean + b
    w1, b1 = src[f"{prefix}.linear_1.weight"].double(), src[f"{prefix}.linear_1.bias"].double()
    w2, b2 = src[f"{prefix}.linear_2.weight"].double(), src[f"{prefix}.linear_2.bias"].double()
    return w2 @ (w1 @ basis.T), w2 @ (w1 @ mean + b1) + b2


def convert(src: Source, adaln: str = "curve", rank: int = 16, grid: int = 1025, log=print) -> dict[str, torch.Tensor]:
    keys = set(src.keys())
    fewstep = any(k.startswith("r_time_embedder.") for k in keys)
    if fewstep and adaln != "lowrank":
        log("interval branch (r_time_embedder) found: writing the low-rank layout for the H3-Tiny few-step loader")
        adaln = "lowrank"
    out: dict[str, torch.Tensor] = {}
    done: set[str] = set()

    curve = None
    if adaln == "curve":
        s = time_curve(src, grid)
        mean = s.mean(0)
        _, sv, vt = torch.linalg.svd(s - mean, full_matrices=False)
        basis = vt[:rank]
        tail = float((sv[rank:] ** 2).sum() / (sv**2).sum())
        log(f"time curve: rank {rank} keeps all but {tail:.1e} of the variance over {grid} points")
        out["adaln_t_table"] = ((s - mean) @ basis.T).float().contiguous()
        curve = (basis, mean)

    def put_adaln(src_prefix: str, dst_prefix: str):
        if adaln == "lowrank" and f"{src_prefix}.linear_1.weight" in src:
            base = dst_prefix.rsplit(".", 1)[0]  # ...adaln_proj
            for lin in ("linear_1", "linear_2"):
                for suffix in ("weight", "bias"):
                    out[f"{base}.{lin}.{suffix}"] = src[f"{src_prefix}.{lin}.{suffix}"]
        elif curve is not None:
            w, b = adaln_low_rank(src, src_prefix, *curve)
            out[f"{dst_prefix}.weight"], out[f"{dst_prefix}.bias"] = w.float().contiguous(), b.float().contiguous()
        else:
            w, b = adaln_affine(src, src_prefix)
            dtype = torch.bfloat16
            out[f"{dst_prefix}.weight"], out[f"{dst_prefix}.bias"] = w.to(dtype).contiguous(), b.to(dtype).contiguous()
        done.update(k for k in keys if k.startswith(src_prefix + "."))

    for src_key, dst in TOP_LEVEL.items():
        if src_key == "norm_out.linear":
            put_adaln("norm_out", "final_layer.adaln_proj.linear")
            continue
        if src_key.startswith("time_embedder") and curve is not None:
            done.update(k for k in keys if k.startswith(src_key + "."))
            continue
        for suffix in ("weight", "bias"):
            k = f"{src_key}.{suffix}"
            if k in keys:
                out[f"{dst}.{suffix}"] = src[k]
                done.add(k)
    if fewstep:
        for lin, dst in (("linear_1", "proj_in"), ("linear_2", "proj_out")):
            for suffix in ("weight", "bias"):
                out[f"r_time_embedder.{dst}.{suffix}"] = src[f"r_time_embedder.{lin}.{suffix}"]
                done.add(f"r_time_embedder.{lin}.{suffix}")

    blocks: dict[tuple[str, int], set[str]] = {}
    for k in keys:
        m = BLOCK_RE.match(k)
        if m:
            blocks.setdefault((m.group(1), int(m.group(2))), set()).add(m.group(3))
    for (group, i), rests in sorted(blocks.items()):
        dst_group = "blocks" if group == "transformer_blocks" else "token_refiner.blocks"
        p, d = f"{group}.{i}", f"{dst_group}.{i}"
        out[f"{d}.attn.qkv_proj.weight"] = torch.cat([src[f"{p}.attn.to_{x}.weight"] for x in "qkv"], 0)
        done.update(f"{p}.attn.to_{x}.weight" for x in "qkv")
        for rest in rests:
            stem, _, suffix = rest.rpartition(".")
            if stem in IN_BLOCK:
                t = src[f"{p}.{rest}"]
                if stem == "ff.net.0.proj":
                    # diffusers SwiGLU projects to [value; gate], ComfyUI's swiglu expects [gate; value]
                    value, gate = t.chunk(2, dim=0)
                    t = torch.cat([gate, value], dim=0)
                out[f"{d}.{IN_BLOCK[stem]}.{suffix}"] = t
                done.add(f"{p}.{rest}")
        if any(r.startswith("adaln_proj.") for r in rests):
            put_adaln(f"{p}.adaln_proj", f"{d}.adaln_proj.linear")

    left = keys - done
    if left:
        raise SystemExit(f"unmapped source tensors: {sorted(left)[:20]}")
    out["rope.inv_freq"] = rope_inv_freq(src.config)

    n_blocks = len({i for g, i in blocks if g == "transformer_blocks"})
    block_ffn = out["blocks.0.mlp.fc1.weight"].shape[0] // 2
    refiner_ffn = out["token_refiner.blocks.0.mlp.fc1.weight"].shape[0] // 2
    log(f"{n_blocks} blocks, block FFN {block_ffn}, refiner FFN {refiner_ffn}, {len(out)} tensors")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="diffusers transformer directory (config.json + safetensors)")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--adaln", choices=["curve", "full", "lowrank"], default="curve",
                    help="AdaLN layout; checkpoints with an interval branch always use lowrank")
    ap.add_argument("--rank", type=int, default=16, help="curve basis size (curve form)")
    ap.add_argument("--grid", type=int, default=1025, help="t grid points of adaln_t_table (curve form)")
    args = ap.parse_args()

    src = Source(Path(args.src))
    state = convert(src, args.adaln, args.rank, args.grid)
    params = sum(t.numel() for k, t in state.items() if k not in ("adaln_t_table", "rope.inv_freq"))
    fewstep = "r_time_embedder.proj_in.weight" in state
    meta = {"h3tiny": json.dumps({"format": 1, "source": Path(args.src).name,
                                   "adaln": "lowrank" if fewstep else args.adaln, "fewstep": fewstep,
                                   "params": params})}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, args.out, metadata=meta)
    print(f"wrote {args.out}: {params / 1e9:.2f}B parameters, {Path(args.out).stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
