#!/usr/bin/env python3
"""Headline benchmark for **locateanything-batch** — end-to-end MTP throughput.

This is the exact protocol behind the README comparison table: a single clean
grounding detection per image, measured END TO END (vision encode + prefill +
fast-MTP decode, all on GPU), batched at batch 8 (the sweet spot on a 16 GB card).

It self-generates 8 single-object images (one colored rectangle each), drives the
native fast-MTP batched path greedily with the single-detection prompt
"Detect the <obj>." (which yields one box that stops cleanly at EOS — the
"Locate all instances" phrasing instead invites degenerate multi-box output),
and reports images/second end-to-end.

Run:
    python examples/benchmark.py                 # batch 8 (default)
    LA_BATCH=1 python examples/benchmark.py       # any batch via env
"""
import json
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_DIR, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402
import locateanything_batch.engine as engine  # noqa: E402
from locateanything_batch import load, generate_batch, load_pil  # noqa: E402

IMG_DIR = os.environ.get("LA_IMG_DIR", os.path.join(REPO_DIR, "examples", "_benchimgs_single"))
RESULTS = os.path.join(REPO_DIR, "examples", "_bench_results", "benchmark.json")
QUERY = os.environ.get("LA_QUERY", "rectangle")
# "Detect the <obj>." -> one box, stops at EOS. (The model's default
# "Locate all the instances that matches ..." invites degenerate multi-box output.)
SINGLE_DETECT_PROMPT = os.environ.get("LA_MTP_PROMPT", "Detect the ")
BATCH = int(os.environ.get("LA_BATCH", "8"))
N = int(os.environ.get("LA_NIMG", "32"))   # total images processed (tile of 8) for a stable mean

# 8 single-object images: (fill color, rectangle box) on a light-gray canvas.
_W, _H, _BG = 1024, 768, (210, 210, 210)
_SPECS = [
    ((220, 40, 40), (120, 140, 470, 520)), ((40, 160, 60), (560, 100, 880, 360)),
    ((40, 90, 220), (320, 380, 700, 690)), ((240, 140, 20), (700, 420, 960, 700)),
    ((150, 40, 190), (140, 90, 380, 470)), ((20, 170, 190), (430, 220, 760, 540)),
    ((210, 40, 140), (600, 60, 900, 300)), ((180, 170, 30), (200, 300, 560, 660)),
]


def ensure_images():
    """Generate the 8 single-object bench images if they're not already present."""
    os.makedirs(IMG_DIR, exist_ok=True)
    if len([f for f in os.listdir(IMG_DIR) if f.endswith(".png")]) >= 8:
        return
    from PIL import Image, ImageDraw
    for i, (color, box) in enumerate(_SPECS):
        im = Image.new("RGB", (_W, _H), _BG)
        ImageDraw.Draw(im).rectangle(box, fill=color)
        im.save(os.path.join(IMG_DIR, f"img_{i:02d}.png"))


def main():
    ensure_images()
    engine._PROMPT = SINGLE_DETECT_PROMPT          # single-detection prompt -> clean 1-box output
    print(f"[bench] loading model ...", flush=True)
    tok, _, _ = load()
    base = [load_pil(os.path.join(IMG_DIR, f"img_{i:02d}.png")) for i in range(8)]
    pairs = [(base[i % 8], QUERY) for i in range(N)]

    generate_batch(pairs[:BATCH], temperature=0.0)   # warmup (absorbs one-time kernel/shape init)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    outs = []
    for i in range(0, N, BATCH):
        outs.extend(generate_batch(pairs[i:i + BATCH], temperature=0.0))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen = sum(len(tok(o, add_special_tokens=False)["input_ids"]) for o in outs)
    res = {
        "engine": "fast-MTP (multi-token) + batched, bf16",
        "batch": BATCH, "n_images": N, "query": QUERY,
        "wall_s": dt, "ms_per_image": dt / N * 1e3, "images_per_s": N / dt,
        "gen_tokens_total": gen, "gen_tokens_per_image": gen / N,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "sample_output": outs[0],
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[bench] batch={BATCH}  {res['ms_per_image']:.0f} ms/img  "
          f"{res['images_per_s']:.2f} img/s   "
          f"({N} imgs in {dt:.2f}s, {gen/N:.0f} tok/img, {res['max_vram_gb']:.1f} GB)")
    print(f"[bench] sample: {outs[0][:120]}")
    print(f"[bench] wrote {RESULTS}")


if __name__ == "__main__":
    main()
