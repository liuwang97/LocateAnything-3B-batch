"""bench_equivalence — correctness + throughput gate for the batched fast-MTP engine.

Correctness is gated in tiers (each isolates one failure mode). The contract is:
under GREEDY decoding (temperature=0), batched fast-MTP must be TOKEN-IDENTICAL to
(a) the stock model.generate in fast mode and (b) running each pair singly.
  Tier A  B=1 new path         == stock generate (fast, greedy)
  Tier B  B=2 identical rows    == Tier A (no cross-row contamination)
  Tier C  B=2 same img / diff prompts, both orders == each pair's single run
  Tier D  B=2 diff imgs / diff lengths            == each single run
  Tier E  >=3 mixed frames (ragged accept counts) == each single run
All tiers run with repetition_penalty 1.0 then 1.15.

Throughput: do_sample (temp 0.5), batch in {1,2,4,8}, ms/img.

Usage:
    python examples/bench_equivalence.py <image_dir> [prompt1] [prompt2]
    (needs >=4 distinct images in <image_dir>; defaults: prompt1="a dog", prompt2="a cat")
"""
import sys, time, re
from pathlib import Path
import torch

from locateanything_batch import load, generate_batch, load_pil
from locateanything_batch.engine import _proc_full, MNT, DT

BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)></box>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
Q1 = sys.argv[2] if len(sys.argv) > 2 else "a dog"
Q2 = sys.argv[3] if len(sys.argv) > 3 else "a cat"


def stock_text(im, q, temperature=0.0, repetition_penalty=1.0, top_p=None):
    """Drive the STOCK custom generate in fast mode (greedy unless temperature>0) — the reference."""
    tok, proc, model = load()
    inp = _proc_full(im, q)
    gk = dict(max_new_tokens=MNT, use_cache=True, generation_mode="fast",
              repetition_penalty=repetition_penalty, verbose=False)
    if temperature and temperature > 0:
        gk["temperature"] = temperature; gk["do_sample"] = True
    if top_p is not None:
        gk["top_p"] = top_p
    out = model.generate(pixel_values=inp["pixel_values"].to(DT), input_ids=inp["input_ids"],
                         attention_mask=inp["attention_mask"], image_grid_hws=inp["image_grid_hws"],
                         tokenizer=tok, **gk)
    return out[0] if isinstance(out, tuple) else out


def _imgs(n):
    files = sorted(p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)[:n]
    return [(load_pil(p), p.name) for p in files]


def _short(s, n=70):
    return (s[:n] + "...") if len(s) > n else s


def parse_boxes(s):
    """All boxes as int tuples in 0-1000 (handles 2- and 4-coord)."""
    out = []
    for m in BOX.finditer(s):
        g = m.groups()
        out.append(tuple(int(x) for x in (g[3:7] if g[3] is not None else g[0:3])))
    return out


def _box_delta(a, b):
    """Max abs coordinate delta between two box lists (inf if counts differ)."""
    ba, bb = parse_boxes(a), parse_boxes(b)
    if len(ba) != len(bb):
        return float("inf")
    if not ba:
        return 0
    return max(abs(x - y) for pa, pb in zip(ba, bb) for x, y in zip(pa, pb))


def _tiers(rp):
    """Return list of (name, got, want) for the 5 tiers at repetition_penalty rp."""
    pool = _imgs(6)
    if len(pool) < 4:
        sys.exit("need >=4 distinct images in %s (got %d)" % (SRC, len(pool)))
    (imA, _), (imB, _), (imC, _), (imD, _) = pool[0], pool[1], pool[2], pool[3]
    g = lambda pairs: generate_batch(pairs, temperature=0.0, repetition_penalty=rp)
    checks = []
    a_single = g([(imA, Q1)])[0]
    checks.append(("A  B=1 new == stock generate", a_single,
                   stock_text(imA, Q1, temperature=0.0, repetition_penalty=rp)))
    bb = g([(imA, Q1), (imA, Q1)])
    checks.append(("B  B=2 identical row0 == single", bb[0], a_single))
    checks.append(("B  B=2 identical row1 == single", bb[1], a_single))
    c = g([(imA, Q1), (imA, Q2)]); c_rev = g([(imA, Q2), (imA, Q1)])
    s_c1 = g([(imA, Q2)])[0]
    checks.append(("C  same-img diff-prompt row0", c[0], a_single))
    checks.append(("C  same-img diff-prompt row1", c[1], s_c1))
    checks.append(("C  order-invariant (swapped)", c_rev[1], a_single))
    d = g([(imB, Q1), (imC, Q2)])
    checks.append(("D  diff-img diff-len row0", d[0], g([(imB, Q1)])[0]))
    checks.append(("D  diff-img diff-len row1", d[1], g([(imC, Q2)])[0]))
    quad = [(imA, Q1), (imB, Q2), (imC, Q1), (imD, Q2)]
    e = g(quad)
    for i, p in enumerate(quad):
        checks.append(("E  mixed-batch row%d (ragged)" % i, e[i], g([p])[0]))
    return checks


def correctness():
    # rp=1.0 is the HARD gate: batched fast-MTP must be token-exact vs stock & vs single.
    # rp=1.15 is a numerical-robustness report: the repetition penalty compresses logit margins
    # so bf16 batched-GEMM nonassociativity (B!=1 -> different reduction order, |delta|<=~0.4) can
    # flip a tight argmax. Expected; report box-level agreement.
    print("=" * 78); print("CORRECTNESS  (Q1=%r  Q2=%r)" % (Q1, Q2)); print("=" * 78)
    print("-- rp=1.00 : HARD token-exact gate --")
    npass = nfail = 0
    for name, got, want in _tiers(1.0):
        ok = (got == want); npass += ok; nfail += (not ok)
        print(("  [PASS] " if ok else "  [FAIL] ") + name)
        if not ok:
            print("         got : %s" % _short(got)); print("         want: %s" % _short(want))

    print("-- rp=1.15 : numerical-robustness report (token-exact OR box delta<=8/1000) --")
    TOL = 8
    rpass = rfail = 0
    for name, got, want in _tiers(1.15):
        exact = (got == want); delta = 0 if exact else _box_delta(got, want)
        ok = exact or delta <= TOL
        rpass += ok; rfail += (not ok)
        tail = "exact" if exact else ("box delta=%s/1000" % (delta if delta != float("inf") else "boxcount!"))
        print(("  [ ok ] " if ok else "  [WARN] ") + name + "   (%s)" % tail)

    print("-" * 78)
    print("HARD GATE (rp=1.0): %d passed, %d failed" % (npass, nfail))
    print("ROBUSTNESS (rp=1.15, fp-nondet): %d within tol, %d beyond" % (rpass, rfail))
    return nfail == 0


def throughput():
    print("=" * 78); print("THROUGHPUT (do_sample, temp=0.5)"); print("=" * 78)
    pool = _imgs(32)
    imgs = [im for (im, _) in pool]
    generate_batch([(imgs[0], Q1)], temperature=0.5, top_p=0.9, repetition_penalty=1.15)  # warmup, not timed
    for Bsz in (1, 2, 4, 8):
        batch = [(im, Q1) for im in imgs]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(0, len(batch), Bsz):
            generate_batch(batch[i:i + Bsz], temperature=0.5, top_p=0.9, repetition_penalty=1.15)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        n = len(batch)
        print("  batch=%-2d : %6.1f ms/img  (%5.2f img/s)  total %.1fs for %d imgs"
              % (Bsz, dt / n * 1000, n / dt, dt, n))


if __name__ == "__main__":
    if SRC is None or not SRC.is_dir():
        sys.exit("usage: python examples/bench_equivalence.py <image_dir> [prompt1] [prompt2]")
    print("loading model ...", flush=True); load()
    ok = correctness()
    if ok:
        throughput()
    else:
        print("\n!! correctness failed -- skipping throughput. Fix equivalence first.")
    print("\nDONE")
