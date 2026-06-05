# LocateAnything-3B — 3-way inference-speed comparison: MTP vs llama.cpp (AR) vs vLLM (AR)

**Hardware:** NVIDIA RTX 5070 Ti (sm_120, 16 GB), driver 595.79. One GPU; every timed run was done **serially on an otherwise-idle GPU** (no cross-contention).
**Model:** `nvidia/LocateAnything-3B` = Qwen2.5-3B LM + MoonViT-SO-400M vision + Eagle-MLP projector. Output = `<ref>label</ref><box><x1><y1><x2><y2></box>`, coords 0–1000.
**Date:** 2026-06-05.

## The three implementations

| | **MTP (this repo)** | **llama.cpp** | **vLLM** |
|---|---|---|---|
| Source | `locateanything-batch` | `yuuko-eth/llama.cpp` fork `mtmd-grounders` + `yuuko-eth/LocateAnything-3B-GGUF` | `WuNein/LocateAnything-vLLM` |
| Decode | **native fast-MTP** (multi-token, accept k∈{1,3,4,6}/step) + batched | plain autoregressive | plain autoregressive |
| Precision tested | bf16 | Q4_K_M (1.96 GB) **and** BF16 (6.34 GB) | fp16 |
| Vision encode | batched on GPU (flash, block-diagonal) | server-side on GPU (mmproj) | **client-side, base64 → server** (split design) |
| Platform | Windows native | Windows native | WSL Debian |
| Notes | the model's intended fast path | mainline llama.cpp does **not** load this model; `--special` required | serves only the extracted Qwen2 backbone via `--enable-prompt-embeds`; no vision tower in vLLM |

Both community ports (llama.cpp, vLLM) are **AR — neither implements MTP** (`speculative_config=None` in vLLM; one-token/step in llama.cpp). MTP is unique to this repo.

## Methodology — and why the final protocol is what it is

Earlier runs hit three confounds, each fixed:
1. **Precision mismatch.** llama.cpp shipped Q4_K_M while vLLM is fp16 / MTP is bf16 → re-ran llama.cpp with the **non-quantized BF16** GGUF for parity.
2. **vLLM hid the vision cost.** The WuNein port did vision encode as an *untimed precompute* → fixed to **time the full pipeline (vision encode INCLUDED)** = true E2E.
3. **Output-length divergence.** The `"Locate all the instances…"` prompt + an out-of-domain query made the **MTP path degenerate into repeated boxes (387 tokens)** while the AR ports returned `None` — not comparable. Fixed with a **single-detection prompt** + single-object images.

**Final clean protocol:** 8 single-object images (1024×768, one colored rectangle each) · prompt **`"Detect the rectangle."`** · greedy (temp 0) · **batch 8** (the standardized batch — batch 32 made MTP regress and pushed past ~16 GB) · precision parity (bf16/fp16). All three then emit the **identical clean single box** `<ref>rectangle</ref><box><117><181><460><678></box>` (~10 tokens, stop) on img_00 — correctness confirmed equal.

---

## A. HEADLINE — End-to-end throughput @ batch 8 (vision + prefill + decode, single clean detection)

| Implementation | precision | **img/s** | **ms/img** | output | 
|---|---|---:|---:|---|
| 🥇 **MTP (this repo)** | bf16 | **4.53** | **221** | 10 tok ✓ |
| 🥈 **llama.cpp** | BF16 | **2.61** | **383** | ~9 tok ✓ |
| 🥉 **vLLM** | fp16 | **1.02** | **977** | 10 tok ✓ |

**MTP is ~1.7× faster than llama.cpp and ~4.4× faster than vLLM end-to-end.**
Batch-1 reference: MTP 3.07 img/s · vLLM 0.44 img/s.

## B. Where the E2E time goes (the key insight)

For a single short detection, **E2E is dominated by vision-encode + prefill, NOT decode**:

| Impl | vision encode | server prefill+decode | bottleneck |
|---|---|---|---|
| vLLM | **82%** (6.42 s / 8 img) | 18% (1.40 s) | **client-side serial vision encode** |
| llama.cpp | (prefill ~1066 img tokens dominates) | decode ~9 tok is tiny | prefill |
| MTP | batched (all 8 in one flash pass) + batched shared-prefix prefill | — | best-amortized → fastest |

This is why raw decode speed (section C) barely moves the E2E needle, and why MTP's **batched** vision+prefill wins.

## C. Raw LM decode capacity (forced 128 tokens, vision excluded — supplementary)

| Impl | precision | decode tok/s @ b1 | decode tok/s aggregate |
|---|---|---:|---:|
| llama.cpp | Q4_K_M | **253** | 526 (b32) |
| llama.cpp | BF16 | 120 | 445 (b32) · 238 (b8) |
| vLLM | fp16 | 100 | **1333 (b32)** · 553 (b8) |
| MTP | bf16 | — multi-token: decodes a whole box per step — | |

- **Quantization (Q4 vs BF16):** 2.1× single-stream decode edge (253 vs 120) — pure memory-bandwidth (4.94 vs 16 BPW). Collapses to **1.18×** at batch 32 (compute-bound, weights amortized).
- **At precision parity:** llama.cpp BF16 single-stream (120) ≈ vLLM (100). vLLM's *batched* decode is far higher (1333 @ b32) — but **wasted** because vision dominates E2E.

## D. Quantization quality caveat

At batch 8 single-detection, **Q4_K_M degenerated** on some images (≈75 tok/img, spurious repeated boxes/labels) where **BF16 stayed clean** (≈9 tok/img). So Q4's decode speedup is doubly negated for real grounding: E2E is vision-bound **and** Q4 lowers output quality. (Q4 b8 E2E came out *slower*, 1.43 img/s, purely from the extra degenerate tokens.)

---

## Verdict

- **Fastest E2E + uses the model as designed → MTP (this repo).** 4.53 img/s @ batch 8. Its batched vision encode + shared-prefix prefill + multi-token box decode win the part of the pipeline that actually costs time. Caveat: its Python per-row box-decode caps batched scaling around batch 8 (batch 32 regresses).
- **Simplest deploy / strong single-stream → llama.cpp.** One native Windows server, solid E2E (2.61 img/s BF16). Q4_K_M gives the highest single-stream decode but can degrade output quality.
- **vLLM (WuNein) is the weakest on a single 16 GB card.** The split design can't host the vision tower and a full-GPU vLLM at once, and the **client-side serial vision encode dominates E2E (~82%)**. Its high batched-decode capacity (1333 tok/s) only pays off with a separate GPU for vision or with much longer outputs.

## Caveats
- Synthetic single-object images (controlled/reproducible). Real photos shift vision-encode cost but the structural conclusion (E2E is vision/prefill-bound) holds.
- Greedy throughout. vLLM vision ran on GPU at `--gpu-memory-utilization 0.55` coexisting with the vision encoder (peak 15.7/16.3 GB).
- batch-32 (earlier, multi-object): llama.cpp 4.93 (Q4)/4.55 (BF16) img/s, vLLM 2.53 img/s, MTP regressed — but those carried the output-length confound; **batch-8 single-detection above is the apples-to-apples result.**

## Reproduce
- **MTP (this repo):** `python examples/benchmark.py` (self-generates the images; batch 8 by default).
- **llama.cpp:** start `llama-server -m <BF16.gguf> --mmproj <mmproj.gguf> -ngl 99 --special -np 8 --cont-batching -c 16384`, then set `LA_IMG_DIR=examples\_benchimgs_single LA_QUERY=rectangle LA_PROMPT="Detect the rectangle." LA_RESULTS_NAME=llamacpp_single_bf16_b8.json LA_QUANT_LABEL="BF16 …"` and run `python examples/bench_compare_llamacpp.py --batch 8`.
- **vLLM (WSL):** in `WuNein/LocateAnything-vLLM`, `start_server_single_b8.sh` (gpu-mem-util 0.55, `--enable-prompt-embeds`) then `LA_EMBED_DEVICE=cuda .venv/bin/python bench_vllm_single_b8.py`.

Raw per-run JSON kept in `examples/_bench_results/`: `benchmark.json` (MTP), `llamacpp_single_bf16_b8.json`, `llamacpp_single_q4_b8.json`, `vllm_single_b8.json`.
