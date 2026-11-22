"""Generate video + audio with an H3 Tiny checkpoint.

    # one clip (ref2va, the default workflow)
    python scripts/infer.py --weights ckpt/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1 --base-model /models/MiniMax-H3 \\
        --prompt "<Picture 1> ..." --references ref1.png --num-frames 124 --out outputs/

    # first / last frame to video + audio with the FL2VA checkpoint (either frame may be omitted; neither = t2va)
    python scripts/infer.py --weights ckpt/MiniMax-H3-Tiny-17B-Pruned-FL2VA --base-model /models/MiniMax-H3 \\
        --workflow fl2va --prompt "..." --image first.png --last-image last.png --out outputs/

    # a JSONL of {"sample_id", "prompt", "references" | "image" / "last_image", "num_frames"?, "height"?, "width"?},
    # sharded over GPUs
    python scripts/infer.py --weights ... --base-model ... --cases cases.jsonl --out outputs/ --shard 0 --num-shards 8

Full-step checkpoints (pruned / OKD) default to 50 steps, few-step checkpoints to 6 (4 or more steps work). Prompts
name references as ``<Picture 1>``, ``<Picture 2>``, ... in the order of ``references``. Finished clips are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3tiny.pipeline import detect_endpoint_mode, load_pipeline  # noqa: E402

FPS = 24
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="H3 Tiny checkpoint directory")
    ap.add_argument("--base-model", required=True, help="official MiniMax-H3 (text encoder, VAEs, processors)")
    ap.add_argument("--prompt")
    ap.add_argument("--workflow", default="ref2va", choices=["ref2va", "fl2va", "t2va"],
                    help="ref2va for Ref2VA checkpoints; fl2va / t2va for FL2VA checkpoints")
    ap.add_argument("--references", nargs="*", default=[], help="ref2va: reference images / videos")
    ap.add_argument("--image", help="fl2va: first frame")
    ap.add_argument("--last-image", help="fl2va: last frame")
    ap.add_argument("--cases", help="JSONL of cases instead of --prompt / --references")
    ap.add_argument("--num-frames", type=int, default=124, help="17n + 5 frames at 24 fps (124 ≈ 5.2 s, 243 ≈ 10.1 s)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--steps", type=int, help="default: 50 for full-step, 6 for few-step checkpoints (4 or more)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--offload", action="store_true", help="CPU offload (GPUs below ~110 GB)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attention", default="auto",
                    help="auto (FlashAttention-3 on Hopper when `kernels` is installed, else sdpa), sdpa, fa3, "
                         "or a diffusers attention backend name")
    args = ap.parse_args()

    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    from diffusers.utils.export_utils import encode_video
    from PIL import Image

    if args.cases:
        cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    elif args.prompt:
        cases = [dict(sample_id="sample", prompt=args.prompt, references=args.references, image=args.image,
                      last_image=args.last_image)]
    else:
        raise SystemExit("pass --prompt (and --references) or --cases")
    mode = detect_endpoint_mode(args.weights)
    steps = args.steps or (6 if mode == "meanflow" else 50)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases[args.shard::args.num_shards] if not (out / f"{c['sample_id']}.mp4").exists()]
    print(f"{args.weights}: {mode} checkpoint, {steps} steps, {len(todo)} clips to generate", flush=True)
    if not todo:
        return 0

    t0 = time.perf_counter()
    workflow = args.workflow
    pipe = load_pipeline(args.weights, args.base_model, workflow=workflow, endpoint_mode=mode, device=args.device,
                         offload=args.offload, attention_backend=args.attention)
    print(f"pipeline loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    failed = []
    for case in todo:
        kwargs = dict(num_frames=case.get("num_frames", args.num_frames))
        if workflow == "ref2va":
            kwargs["references"] = [(MiniMaxH3VideoReference if Path(p).suffix.lower() in VIDEO_SUFFIXES
                                     else MiniMaxH3ImageReference).from_file(p) for p in case.get("references", [])]
        elif workflow == "fl2va":
            for key in ("image", "last_image"):
                if case.get(key):
                    kwargs[key] = Image.open(case[key]).convert("RGB")
        for key in ("height", "width"):
            value = case.get(key, getattr(args, key))
            if value:
                kwargs[key] = value
        t0 = time.perf_counter()
        try:
            result = pipe(prompt=case["prompt"], num_inference_steps=steps,
                          generator=torch.Generator().manual_seed(case.get("seed", args.seed)),
                          output=["videos", "audio", "sampling_rate"], **kwargs)
        except Exception as e:  # keep going with the other clips; report at the end
            failed.append(case["sample_id"])
            print(f"{case['sample_id']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        tmp = out / f".{case['sample_id']}.tmp.mp4"
        encode_video(result["videos"][0], fps=FPS, output_path=str(tmp), audio=result["audio"][0],
                     audio_sample_rate=result["sampling_rate"])
        tmp.rename(out / f"{case['sample_id']}.mp4")
        print(f"{case['sample_id']}: {time.perf_counter() - t0:.1f} s", flush=True)
    if failed:
        print(f"{len(failed)} clips failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
