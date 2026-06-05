#!/usr/bin/env python3
"""
End-to-end batch benchmark for LocateAnything-3B on the **llama.cpp** AR port
(yuuko-eth `mtmd-grounders` fork) — the AR counterpart to this repo's native MTP
`benchmark.py`. Fires BATCH concurrent requests at a llama-server launched with
parallel slots; each request does vision-encode + prefill + decode server-side, so
the wall over the concurrent batch IS end-to-end img/s.

Start the server first (use the NON-quantized BF16 GGUF for precision parity with
vLLM fp16 / MTP bf16):
  llama-server.exe -m <BF16.gguf> --mmproj <mmproj.gguf> -ngl 99 --special \
    --host 127.0.0.1 --port 8080 --no-webui -np 8 --cont-batching -c 16384

Single-detection workload (matches benchmark.py) via env vars:
  set LA_IMG_DIR=examples\\_benchimgs_single & set LA_QUERY=rectangle
  set LA_PROMPT=Detect the rectangle. & set LA_RESULTS_NAME=llamacpp_single_bf16_b8.json
  set LA_QUANT_LABEL=BF16 (non-quantized) + BF16 mmproj
  python examples\\bench_compare_llamacpp.py --batch 8

Two measurements (both fire BATCH concurrent):
  (B) HEADLINE E2E: greedy, natural EOS  -> img/s (vision + prefill + decode).
  (A) raw decode capacity: greedy, ignore_eos, exactly DECODE_N tokens -> agg tok/s.
"""

import argparse
import base64
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ------------------------- CONFIG -------------------------
HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}"

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.environ.get("LA_IMG_DIR", os.path.join(REPO_DIR, "examples", "_benchimgs"))
RESULTS_DIR = os.path.join(REPO_DIR, "examples", "_bench_results")
RESULTS_JSON = os.path.join(RESULTS_DIR, os.environ.get("LA_RESULTS_NAME", "llamacpp_batch32.json"))

QUERY = os.environ.get("LA_QUERY", "person")
DECODE_N = 128
GEN_MAX = 256
N_IMAGES = 8
REQUEST_TIMEOUT = 600
PROMPT_TEXT = os.environ.get("LA_PROMPT", f"Detect the {QUERY} in the image.")
# ----------------------------------------------------------


def img_path(i):
    return os.path.join(IMG_DIR, f"img_{i:02d}.png")


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def http_post(endpoint, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + endpoint, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(BASE_URL + "/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    if body.get("status") == "ok":
                        return True
        except Exception:
            pass
        time.sleep(1)
    return False


def one_request(b64, n_predict, ignore_eos):
    payload = {
        "model": "locateanything",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEXT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": n_predict,
        "ignore_eos": ignore_eos,
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = http_post("/v1/chat/completions", payload)
    wall_ms = (time.perf_counter() - t0) * 1e3
    timings = resp.get("timings", {}) or {}
    text = resp["choices"][0]["message"]["content"]
    return {
        "wall_ms": wall_ms,
        "gen_n": timings.get("predicted_n"),
        "gen_ms": timings.get("predicted_ms"),
        "gen_tok_s": timings.get("predicted_per_second"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "text": text,
    }


def run_batch(embeds, n_predict, ignore_eos, batch):
    """Fire `batch` requests concurrently; return (per_req, wall_s)."""
    items = [(embeds[i % len(embeds)]) for i in range(batch)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch) as ex:
        per = list(ex.map(lambda b: one_request(b, n_predict, ignore_eos), items))
    wall_s = time.perf_counter() - t0
    return per, wall_s


def pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
    return vals[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    batch = args.batch

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not wait_for_server(20):
        raise SystemExit(
            f"server not reachable at {BASE_URL}. Launch llama-server with "
            f"-np {batch} --cont-batching -c 32768 first.")

    embeds = [b64_image(img_path(i)) for i in range(N_IMAGES)]
    print(f"[bench] server up. batch={batch} decode_n={DECODE_N} images={N_IMAGES}")

    # Warmup batch (discarded).
    print("[bench] warmup batch ...")
    run_batch(embeds, 16, True, batch)

    # (A) decode throughput at batch
    perA, wallA = run_batch(embeds, DECODE_N, True, batch)
    genA = sum(r["gen_n"] for r in perA if r["gen_n"])
    aggA = genA / wallA
    latA = [r["wall_ms"] for r in perA]

    # (B) realistic at batch
    perB, wallB = run_batch(embeds, GEN_MAX, False, batch)
    genB = sum((r["gen_n"] or 0) for r in perB)
    aggB = genB / wallB
    latB = [r["wall_ms"] for r in perB]

    summary = {
        "batch": batch,
        "decode_n_forced": DECODE_N,
        # ---- HEADLINE: E2E throughput (each request does vision-encode + prefill +
        # decode server-side, so wall over the concurrent batch IS end-to-end) ----
        "B_E2E_img_per_s": batch / wallB,          # realistic natural-EOS E2E throughput
        "B_E2E_ms_per_img": wallB / batch * 1000.0,
        "B_wall_s": wallB,
        "B_total_gen_tokens": genB,
        "B_req_latency_ms_mean": statistics.mean(latB),
        "B_sample_text": perB[0]["text"][:160],
        # (A) decode-stress (forced 128 tok) — raw decode capacity + its own E2E
        "A_E2E_img_per_s": batch / wallA,
        "A_wall_s": wallA,
        "A_total_gen_tokens": genA,
        "A_aggregate_decode_tok_s": aggA,
        "A_req_latency_ms_mean": statistics.mean(latA),
        "A_req_latency_ms_p50": pct(latA, 50),
        "A_req_latency_ms_p99": pct(latA, 99),
    }

    result = {
        "track": "llamacpp-windows-native",
        "mode": f"batch-{batch} continuous batching (-np {batch})",
        "model": "LocateAnything-3B",
        "decoding": "autoregressive (no MTP)",
        "quant": os.environ.get("LA_QUANT_LABEL", "Q4_K_M (LLM) + BF16 mmproj"),
        "query": QUERY,
        "summary": summary,
        "per_request_A": perA,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n===== BATCH SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n[bench] wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
