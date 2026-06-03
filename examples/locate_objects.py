"""locate_objects — batched LocateAnything-3B detection over a folder of images.

Demonstrates locateanything-batch on an arbitrary image folder. For every image it runs one or
more text queries, parses the ``<box>`` coordinates out of the model's answer, draws them, and
writes an overlay PNG per image plus a summary.json.

With a SINGLE prompt it uses ``generate_batch`` (one (image, query) pair per row). With MULTIPLE
prompts it uses ``generate_batch_grouped`` so each image's [image + instruction] prefix is
vision-encoded + prefilled ONCE and forked across that image's queries (the headline speedup).

Usage:
    python examples/locate_objects.py <image_dir> "<prompt>" ["<prompt2>" ...] [--batch 32] [--out DIR]

Examples:
    python examples/locate_objects.py ./photos "a dog"
    python examples/locate_objects.py ./photos "a dog" "a person riding a bicycle" --batch 32

Drawing needs the [example] extra (opencv):  pip install "locateanything-batch[example]"
"""
import sys, re, json, time, argparse
from pathlib import Path

import numpy as np
import cv2

from locateanything_batch import load, generate_batch, generate_batch_grouped, load_pil

BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")   # LocateAnything-3B box format (coords 0..1000)
PALETTE = [(0, 200, 0), (0, 140, 255), (255, 80, 0), (200, 0, 200), (0, 215, 255), (180, 180, 0)]  # BGR per query
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse(answer):
    """Pull every <box><x1><y1><x2><y2></box> out of the model answer -> normalized [x1,y1,x2,y2]
    in 0..1, filtered to 0 < area < 0.95 (drops degenerate / whole-image boxes)."""
    out = []
    for m in BOX.finditer(answer):
        b = [int(g) / 1000 for g in m.groups()]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if 0 < area < 0.95:
            out.append(b)
    return out


def draw(im, per_query, labels, out_path):
    """Draw each query's boxes in its palette color. per_query[i] = list of normalized boxes for
    labels[i]. imencode().tofile() keeps unicode paths safe on Windows."""
    w, h = im.size
    ov = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    for qi, boxes in enumerate(per_query):
        color = PALETTE[qi % len(PALETTE)]
        for b in boxes:
            x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, 2)
            cv2.putText(ov, labels[qi], (x1 + 2, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imencode(".png", ov)[1].tofile(str(out_path))


def run_chunk(group_chunk, nq):
    """Run one micro-batch of (image, prompts) groups -> list (per image) of nq answer strings.
    nq==1 has no per-image tail to share, so it goes through generate_batch; nq>1 reuses the prefix."""
    if nq == 1:
        outs = generate_batch([(im, qs[0]) for im, qs in group_chunk])
        return [[o] for o in outs]
    return generate_batch_grouped(group_chunk)


def main():
    ap = argparse.ArgumentParser(description="Batched LocateAnything-3B detection over an image folder.")
    ap.add_argument("image_dir", help="folder of images to scan")
    ap.add_argument("prompts", nargs="+", help="one or more text queries to locate")
    ap.add_argument("--batch", type=int, default=32, help="approx rows per batch (images x prompts)")
    ap.add_argument("--out", default=None, help="output dir (default <image_dir>/locate_out)")
    args = ap.parse_args()

    src = Path(args.image_dir)
    out = Path(args.out) if args.out else src / "locate_out"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        sys.exit("no images found in %s" % src)

    prompts = args.prompts
    nq = len(prompts)
    print("loading model ...", flush=True); load()
    imgs = [load_pil(p) for p in files]
    groups = [(im, prompts) for im in imgs]
    gper = max(1, args.batch // nq)                        # images/batch -> ~batch rows (nq rows per image)
    print("images=%d  prompts=%d  ~%d rows/batch (%d imgs/batch)  KV/vision prefix reuse=%s"
          % (len(files), nq, gper * nq, gper, "on" if nq > 1 else "n/a (single prompt)"), flush=True)

    run_chunk([groups[0]], nq)                             # warmup (not timed)

    results = []                                           # results[k] = nq answer strings for image k
    t0 = time.perf_counter()
    nb = (len(groups) + gper - 1) // gper
    for bi, i in enumerate(range(0, len(groups), gper), 1):
        results += run_chunk(groups[i:i + gper], nq)
        print("  batch %2d/%d  (%d/%d imgs)" % (bi, nb, min(i + gper, len(groups)), len(groups)), flush=True)
    dt = time.perf_counter() - t0

    rows = []
    for k, f in enumerate(files):
        per_query = [parse(results[k][qi]) for qi in range(nq)]
        draw(imgs[k], per_query, prompts, out / (f.stem + "_box.png"))
        rows.append({"file": f.name, **{"q%d" % qi: len(per_query[qi]) for qi in range(nq)}})
    (out / "summary.json").write_text(
        json.dumps({"prompts": prompts, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(files); npairs = n * nq
    print("\n" + "=" * 56)
    print("done %d images (%d pairs) in %.1fs" % (n, npairs, dt))
    print("  %.1f ms/img  |  %.1f ms/pair  |  %.2f img/s" % (dt / n * 1000, dt / npairs * 1000, n / dt))
    for qi, p in enumerate(prompts):
        hit = sum(1 for r in rows if r["q%d" % qi] > 0)
        print("  hit[%r] : %d/%d" % (p, hit, n))
    print("  output: %s" % out)


if __name__ == "__main__":
    main()
