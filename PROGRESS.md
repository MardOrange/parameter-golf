# Parameter Golf — Progress Journal

**Participant:** MardOrange (GitHub)
**Challenge window:** 2026-03-18 → 2026-04-30
**Fork:** https://github.com/MardOrange/parameter-golf
**Upstream:** https://github.com/openai/parameter-golf
**Current SOTA (at start):** 1.0810 BPB (SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 + Legal TTT)

---

## 2026-04-17 — Setup + Smoke Test + Play 2 launched

### Environment
- Local: Windows 11, RTX 3060 6GB (not used for training — pipeline setup painful on Windows)
- Compute: RunPod 1× RTX 4000 Ada (20 GB VRAM), `runpod/parameter-golf:latest` image, $0.26/hr on-demand
- SSH: via `runpodctl` auto-managed `RunPod-Key-Go` key at `~/.runpod/ssh/`
- API key rotation: **TODO — rotate `rpa_QQNQZ6...brig` after challenge complete (exposed in Claude session)**

### Research findings (before any runs)
- Top submissions use LZMA-compressed `train_gpt.py` (2-line bootloader + compressed human-readable version). Saves ~30–40 KB of code budget.
- TTT by itself is worth only ~0.003 BPB (from LoRA TTT ablation) — strided/doc-isolated eval is the real gain.
- Entropy argument (Kev Clark): compressed size ≈ H(q), not raw bitwidth. Going from int6 → int4 yields less savings than naive math suggests.
- Factored tied embeddings (8192 × 254 bottleneck) proven in non-record 106M binary UNet submission — absent from record track.
- Teacher distillation likely infeasible in 10-min budget (compute overhead + pretrained teacher probably illegal per "no external compute" rule).
- Unchecked on OpenAI's "Requests for PRs": JEPA, text diffusion, H-net tokenization, universal transformer, megakernels, state-space models.

### Chosen direction (subject to validation)
**Primary:** Non-record track — Mamba/SSM hybrid (replace last 3 layers with SSM blocks). Addresses an explicit OpenAI wishlist item, zero existing record-track entries.
**Secondary:** Record track — adopt factored tied embeddings + reallocate freed ~2 MB budget to +1 physical layer. Proven in non-record; novel in record track.

### Smoke test (`runs/smoke_test/`)
**Config:** `NUM_LAYERS=2 MODEL_DIM=128 TRAIN_SEQ_LEN=256 ITERATIONS=100 TRAIN_BATCH_TOKENS=16384`
**Result:** `final_int8_zlib_roundtrip val_bpb: 3.1697`
- Training time: 8.9 s for 100 steps
- Peak VRAM: 78 MiB
- Submission size: 553 KB (well under 16 MB cap)
- Quantization stable: 3.1687 (pre-quant) → 3.1697 (roundtrip)
- **Verdict:** pipeline verified end-to-end, loss decreasing normally

### Play 2 — reduced baseline (COMPLETE)
**Config:** `NUM_LAYERS=9 MODEL_DIM=512 TRAIN_SEQ_LEN=1024 ITERATIONS=2000 TRAIN_BATCH_TOKENS=524288` (defaults)
**Final val_bpb (post-quant, submission-quality):** **1.3413**
**Pre-quant val_bpb:** 1.3401 (quantization drift only +0.0012 — very stable)
**Mid-eval at step 1000:** val_bpb 1.3867 (converged further to 1.3413 over last 1000 steps)
**Artifact size:** 15,053,903 bytes (15.05 MB — ~950 KB under cap)
**Train loss trajectory:** 6.93 → 3.31 (step 100) → 2.50 (step 500) → 2.15 (step 2000)
**Throughput:** 2.22 s/step on 1× RTX 4000 Ada, 100% GPU utilization, ~11 GB VRAM peak
**Total wall time:** 1h 22min (train 1h 14min + eval 82s + quantization/serialization)
**Cost:** ~$0.36 (at $0.26/hr on-demand)
**Log:** `runs/play2_baseline/play2_baseline.txt`

**Gap analysis vs leaderboard:**
- Current SOTA (1.0810): our run +0.26 BPB worse
- Naive baseline (1.2244, full 8xH100 × 20K iters × 8B tokens): our run +0.12 BPB worse
- We used roughly **1/320th the compute** of naive baseline setup (1/80 data × 1/10 iters × 1/8 GPU × ~1/4 speed)
- At full scale this same config would likely match naive baseline (~1.22)

---

## Decisions log

- **2026-04-17** — Chose RTX 4000 Ada on-demand over 2× A40 spot: cheaper for exploration, no risk of spot eviction during 80-min run, 20 GB fits everything we'd try at this scale.
- **2026-04-17** — Dropped earlier "FAME" proposal (FTE-TD teacher distillation) after learning: (a) online teacher ~4× compute overhead, (b) pretrained teacher likely violates "no external compute" rule, (c) leaderboard ablations show TTT itself is nearly marginal.
- **2026-04-17** — Adopted Style B (abay-style) fork layout: all personal work inside the fork, gitignored until ready for public push.

---

## Open questions

- Will Play 2 hit a reasonable val_bpb (<2.0)? If so, strong grant-pitch data.
- Does factored-embedding + MuonEq-R interact cleanly, or does the matrix-shape assumption in Muon break on the thin 8192×256 matrix?
- Which Mamba implementation to port — `mamba-ssm` pip package or handwritten S4D block? Need to check whether `mamba-ssm` is already in the RunPod image or needs building from source on Hopper-less hardware.

---

## 2026-04-17 — Factored Embeddings experiment (first novel implementation)

### What changed in code
`d:/parameter_golf/repo/experiments/factored_embed/train_gpt.py` — 25-line additive patch to baseline:
- New `EMBED_DIM` hyperparameter (defaults to `MODEL_DIM` = no-op)
- `GPT.__init__`: if `embed_dim < model_dim`, build `tok_emb` at `[vocab, embed_dim]` + `embed_proj: CastedLinear[embed_dim, model_dim]`
- Forward input path: `x = embed_proj(tok_emb(ids))` then RMS norm
- Forward output path: `F.linear(x, embed_proj.weight.T)` before tied-logit matmul
- Optimizer: `embed_proj.weight` added to Muon's `matrix_params`

### Smoke test (`runs/fte_vs_play2/fte_smoke.txt`)
Config: `NUM_LAYERS=2 MODEL_DIM=128 EMBED_DIM=64 ITERATIONS=100`
**val_bpb: 3.2362** (vs baseline smoke 3.1697 = +0.067)
Pipeline verified — no crashes, quantization clean, forward/backward stable.

### Apples-to-apples comparison (`runs/fte_vs_play2/fte_vs_play2.txt`)
Config: same as Play 2 (`MODEL_DIM=512 ITERATIONS=2000`) plus `EMBED_DIM=256`
**Final val_bpb (post-quant, submission-quality): 1.3599**
**Pre-quant val_bpb: 1.3553** (quantization drift: +0.0046, slightly higher than Play 2's +0.0012)
**Artifact size:** 14,847,986 bytes (14.85 MB — 206 KB smaller than Play 2's 15.05 MB)
**Model params:** 16,928,840 (131,072 fewer than Play 2 — all in the embedding subsystem)
**Throughput on L4:** 2.98 s/step (L4 is ~35% slower than RTX 4000 Ada)
**Wall time:** 1h 43min train + 108s eval = ~1h 45min
**Cost:** ~$0.68 (L4 at $0.39/hr)

### Gap analysis
| Run | val_bpb | Artifact | Gap vs Play 2 |
|---|---|---|---|
| Play 2 (no factoring) | 1.3413 | 15,053,903 | baseline |
| Factored (`EMBED_DIM=256`) | **1.3599** | 14,847,986 | **+0.019 BPB, −206 KB** |

**Verdict:** factored embeddings cost 0.019 BPB at sp1024 scale in exchange for 206 KB savings. At sp1024 this is a bad trade (too little saved for the cost). **At sp8192 the savings scale to ~2 MB** — enough to add +1 physical layer. If +1 layer buys more than 0.019 BPB (likely based on layer-scaling trends on the leaderboard), the trade flips to positive.

### Gap-narrowing over training
| Step | train_loss gap (factored - play2) |
|---|---|
| 100 | +0.19 |
| 400 | +0.10 |
| 1100 | +0.06 |
| 1900 | +0.05 |

Projection matrix continues learning throughout — gap narrows 4× from start to end. At the full 20k-step budget, gap would likely close further.

### Decisions made
- **Approach validated for code correctness**: the modified `train_gpt.py` compiles, trains, quantizes, and roundtrips cleanly. Ready to port to SOTA recipe.
- **Do not submit as a record at sp1024**: +0.019 BPB worse than baseline is the wrong direction for a record attempt.
- **Next: port to SOTA recipe at sp8192**: the factoring-freed ~2 MB becomes meaningful only when vocab is larger, and combined with SOTA tricks (GPTQ, parallel residuals, depth recurrence, legal TTT), the +1 layer gain should exceed the factoring cost.

---

## 2026-04-20 — SOTA recipe port (experiments/factored_sota)

### What we did
Copied `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py` (Kev Clark, val_bpb 1.08563) into `experiments/factored_sota/train_gpt.py`. Made FA3 optional with SDPA fallback so we can smoke-test on non-Hopper GPUs (L4, RTX 4000 Ada). Only 13 lines changed (2 edits).

### Key discovery
Kev Clark's script **already has factored embeddings built in** via separate `embed_proj` (embedding_dim → model_dim) and `head_proj` (model_dim → embedding_dim). Our earlier `experiments/factored_embed/` used a simpler shared-transpose approach, which is strictly less capacity. The SOTA port is the cleaner basis going forward.

### What's in this script out-of-the-box
SP8192 + GPTQ SDClip (int6/int8 + k-based clipping) + depth recurrence (loops 4-5 twice) + MuonEq-R + partial RoPE (16/64) + layerwise LN scale + U-Net skip gates + sliding-window eval + Brotli compression + factored embeddings (via `EMBEDDING_DIM < MODEL_DIM`).

### What's still missing (vs current 1.0810 SOTA)
- QK-Gain bump from 4.0 → 5.25 (env var change)
- 3-layer recurrence (loops 3-5, not 4-5) — small hparam change
- Parallel residuals from layer 7 — ~30 lines to port
- Legal score-first TTT — ~150 lines to port

These are the "additions on top of Kev Clark" that the current leaderboard leader (SP8192_3LayerRecur_ParResid_QK525_LegalTTT) uses.

### Ready for next session
1. Rent L4 or RTX 4000 Ada
2. Download sp8192 data (1 shard for smoke)
3. Smoke test (~3-5 min): verify SDPA fallback works
4. Reduced-scale validation (~2-3 hours): confirm convergence
5. When grant credits arrive: full-scale 8× H100 run

### Files created
- `experiments/factored_sota/train_gpt.py` (1421 lines, 13 added vs Kev Clark original)
- `experiments/factored_sota/README.md`

---

## 2026-04-25 — Recipe A attempt (Polar Express + z-loss) — partial result, no A/B verdict

### Goal
Implement the highest-ranked items from Claude's research report (Polar Express NS coefficients + PaLM z-loss) into `experiments/factored_sota/train_gpt.py`, validate they don't break training, then run a reduced-scale A/B against the unmodified factored_sota baseline.

### Patches applied (3 surgical edits, all opt-in via env var)
1. **Polar Express NS** (`POLAR_EXPRESS=1`): per-iteration minimax-optimal NS coefficients replacing the static Jordan triple `(3.4445, -4.7750, 2.0315)`. Default off → byte-identical to baseline.
2. **z-loss** (`Z_LOSS_COEF=1e-4`): PaLM-style auxiliary `coef * mean(logsumexp(logits)²)` added to CE loss. Default 0.0 → no-op.
3. **Hyperparam plumbing**: `Hyperparameters.z_loss_coef` field + threading through `GPT.__init__`.

### Bug found, fixed (worth saving for future)
The Polar Express coefficients in Claude's research report `(8.205, -23.953, 17.778)` and the standard `X /= X.norm() + eps` Frobenius scaling **diverge to NaN at step 2** of training. Confirmed empirically via PE-only smoke (loss → nan immediately).

**Root cause:** Polar Express step-0 coefficients have very high slope at zero. With Frobenius normalization, σ_max of the scaled X can equal 1, and step 0 amplifies σ_max ≈ 1 to roughly `8.205 - 23.953 + 17.778 = 2.030` instead of converging — diverges in 1 iteration.

**Fix:** pulled the canonical implementation from modded-nanogpt PR #134 directly:
- Replace Frobenius scaling with `X / (X.norm() * 1.02 + 1e-6)` (1.02× safety multiplier ensures σ_max < 1 strictly)
- Use the actually-tested 5-tuple coefficient table:
  - `(8.156554..., -22.483..., 15.879...)`
  - `(4.042930..., -2.808917..., 0.500018...)`
  - `(3.891668..., -2.772484..., 0.506065...)`
  - `(3.285754..., -2.368129..., 0.464490...)`
  - `(2.346541..., -1.709783..., 0.423236...)`

After fix: PE+z-loss validation smoke at NUM_LAYERS=6 MODEL_DIM=128 ITERATIONS=50 ran cleanly, loss 9.02 → 5.90, no NaN.

### Compute used
RunPod RTX 4000 Ada 20 GB on-demand at $0.26/hr. Total spend: $1.86 of allotted budget consumed (pod auto-terminated when credits hit zero during overnight run).

### What we measured
**Baseline (unmodified factored_sota at sp8192, NUM_LAYERS=11 MODEL_DIM=512 EMBEDDING_DIM=256 TRAIN_BATCH_TOKENS=262144 ITERATIONS=1500):**
- step 100: train_loss 4.66
- step 500: train_loss 3.65, **val_bpb 1.3931**
- step 700: train_loss 3.51
- step 750: depth recurrence enabled (`encoder:[0,1,2,3,4,5,4] decoder:[5,4,5,6,7,8,9,10]`) — clean activation, no instability
- step 900: train_loss 3.34, throughput 117K tok/s post-recurrence
- **No further data captured** — pod credits ran out during overnight run before baseline finished or variant started

**Variant (POLAR_EXPRESS=1 Z_LOSS_COEF=1e-4):** never measured at reduced scale.

### Why no A/B verdict
The reduced-scale runs were sequential (baseline → variant). Baseline got to step 900/1500 (60%) before the pod ran out of credits. Variant never started its reduced-scale run.

### Failures along the way (cost lessons)
- **First reduced A/B OOM'd**: factored_sota's default `TRAIN_BATCH_TOKENS=786432` (assumes 8×H100) creates a 384 MB attention buffer that doesn't fit in 20 GB VRAM. Lowered to 262144 (microbatch=16) — worked fine.
- **Two `runpod/parameter-golf:latest` image issues**: missing `brotli` package (pip install --break-system-packages brotli), stale willdepueoai HF dataset manifest cached at `data/manifest.json` blocking sp8192 manifest fetch (rm + retry).
- **Pod credits timing**: the user's $1.86 budget was depleted overnight when no one was monitoring. Without an automatic spend cap, pod ran until balance hit zero.

### What's still ready locally for next attempt
- `experiments/factored_sota/train_gpt.py` with PE+z-loss patches (validated to not NaN)
- Knowledge that depth recurrence + factored embeddings + sp8192 train cleanly on a 1× RTX 4000 Ada with `TRAIN_BATCH_TOKENS=262144`
- Validated baseline trajectory data point: val_bpb 1.3931 at step 500 (use as reference for next variant-only run)

### Decisions / lessons
- **2026-04-25** — When porting code patches from research reports (Claude or Gemini), **always cross-check against the canonical source PR** before running on rented compute. The Polar Express coefficients in the report were close but off, and the missing 1.02× safety multiplier caused immediate divergence — this would have wasted hours if not caught at smoke scale. Cost of catching at smoke: ~$0.05. Cost if it had hit at reduced scale: ~$0.30-0.50 + lost hours.
- **2026-04-25** — RunPod's "rent until balance zero" is a soft destruction trap for unattended overnight runs. For future runs: either set `MAX_WALLCLOCK_SECONDS` aggressively in the script OR stop the pod manually before sleeping.
- **2026-04-25** — `factored_sota` script defaults assume 8×H100 — every 1-GPU smoke MUST override `TRAIN_BATCH_TOKENS` and `NUM_LOOPS` (default 2 + LOOP_START=4 LOOP_END=5 requires NUM_LAYERS ≥ 6; below that it crashes with IndexError on the loop-expansion code).
