# Experiment: Factored Embeddings + SOTA Recipe Port

**Status:** Script ported, awaiting smoke test
**Started:** 2026-04-20
**Baseline to beat:**
- Our Play 2 (no factoring): 1.3413 @ reduced scale
- Current leaderboard SOTA: 1.0810 BPB
- Kev Clark's base (#1394, which we ported from): 1.08563

## What this is

A direct port of Kev Clark's `train_gpt_human.py` (PR #1394, val_bpb 1.08563) from `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/`, with one surgical modification: **FlashAttention 3 is now optional**, falling back to `F.scaled_dot_product_attention` on non-Hopper GPUs.

This lets us prototype and smoke-test on L4 / RTX 4000 Ada before committing to an 8× H100 run.

## Features already in the script (from Kev Clark)

| Feature | How it's controlled |
|---|---|
| **SP8192 vocab** | `VOCAB_SIZE=8192` (default) |
| **Factored tied embeddings** | `EMBEDDING_DIM < MODEL_DIM` enables `embed_proj` + `head_proj` (separate, not shared-transpose) |
| **GPTQ SDClip quantization** | int6 matrices (k=12.85), int8 embeddings (k=20.0) |
| **Depth recurrence** | `NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.5` (loops layers 4-5 twice, activates at 50% of training) |
| **MuonEq-R** | Row-normalized Muon (in-script) |
| **Partial RoPE** | `ROPE_DIMS=16` of 64 head_dim |
| **Layerwise LN scale** | `LN_SCALE=1` enables `1/sqrt(layer_idx+1)` scaling |
| **QK-Gain** | `QK_GAIN_INIT=4.0` (the current 1.0810 SOTA uses 5.25 — TODO bump) |
| **U-Net skip gates** | `SKIP_GATES_ENABLED=1` |
| **Sliding-window eval** | `SLIDING_WINDOW_ENABLED=1` |
| **Brotli-11 compression** | in-script (adds to LZMA wrapper trick) |

## What's NOT in this script (vs current 1.0810 SOTA)

The current leaderboard SOTA extends Kev Clark's base with:
1. **QK-Gain 5.25** (vs this script's default 4.0) — easy env-var tweak
2. **3-layer recurrence** (vs 2-layer in this script) — small hyperparam change
3. **Parallel residuals** (from layer 7) — ~30 lines to add
4. **Legal score-first TTT** (at eval) — ~150 lines to add

These will be separate follow-up patches if the base port works cleanly.

## Only modification vs the original

**FA3 → SDPA fallback** for non-Hopper GPUs:

```python
# Line ~22 (original import):
from flash_attn_interface import flash_attn_func as flash_attn_3_func

# Our change:
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except Exception:
    flash_attn_3_func = None

# Line ~413 (attention forward):
# Original: y = flash_attn_3_func(q, k, v, causal=True)
# Our change:
if flash_attn_3_func is not None:
    y = flash_attn_3_func(q, k, v, causal=True)
else:
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True,
        enable_gqa=(self.num_kv_heads != self.num_heads),
    ).transpose(1, 2).contiguous()
```

**Behavioral identity:** on H100, FA3 is used (identical to Kev Clark's submission). On L4 / RTX 4000 Ada, SDPA fallback kicks in. SDPA is slightly slower but numerically equivalent in bf16.

## Run configs

### 1. Quick smoke test on L4 (~3–5 min, ~$0.03)
Tiny config to verify the SDPA fallback works end-to-end:

```bash
cd /workspace/parameter-golf
# Requires the sp8192 dataset to be downloaded
RUN_ID=fsota_smoke \
NUM_LAYERS=3 MODEL_DIM=128 EMBEDDING_DIM=64 \
TRAIN_SEQ_LEN=512 ITERATIONS=100 \
TRAIN_BATCH_TOKENS=16384 \
VAL_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 2. Reduced-scale reproduction on L4 (~2–3 hours, ~$1)
Same config as Kev Clark's script defaults but reduced iters:

```bash
RUN_ID=fsota_smallrun \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Expected landing: somewhere in `~1.35–1.45 BPB` range (reduced iters + SDPA on L4). Useful mostly to verify training converges and quantization roundtrips clean.

### 3. Full record-track attempt (needs 8× H100, from grant credits)
Kev Clark's exact config (20000 iters, 10 min wallclock):

```bash
RUN_ID=fsota_full_8xh100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Should land at **~1.086 BPB** (matching the source submission).

### 4. Full attempt + push QK-Gain to 5.25 + 3-layer recurrence (small SOTA tweaks)
If #3 works, try the low-hanging SOTA differentiators:

```bash
RUN_ID=fsota_qk525_3loop \
QK_GAIN_INIT=5.25 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Data requirement

This script uses **SP8192** vocab (not sp1024). Before any run:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1
```

(`--train-shards 128` for a full run, 1 shard for smoke.)

## Attribution

Base script authored by **@clarkkev** (Kevin Clark) for [PR #1394](https://github.com/openai/parameter-golf/pull/1394). All architectural choices, optimizer tuning, GPTQ SDClip derivation, and hyperparameter selection are his work. Our only change is the FA3 fallback above. Licensed under the same MIT terms as the upstream openai/parameter-golf repo.

## Known limitations / next steps

- [ ] Requires sp8192 dataset download (~1 GB for 1 shard)
- [ ] SDPA fallback path is slower than FA3 — expect 1.5–2× step time on non-Hopper GPUs
- [ ] QK-Gain needs bump from 4.0 → 5.25 to match latest SOTA
- [ ] 3-layer recurrence (loops 3-5, not 4-5) — small hparam change
- [ ] Parallel residuals from layer 7 — not yet ported (~30 lines)
- [ ] Legal score-first TTT — not yet ported (~150 lines)

## Relationship to `experiments/factored_embed/`

Our earlier `experiments/factored_embed/` is a **minimal patch** to the baseline `train_gpt.py` — uses a shared-transpose projection, not separate up/down projections. Simpler but less capacity.

This experiment uses Kev Clark's **separate up/down projections** (`embed_proj` + `head_proj`), which is the proven SOTA approach. If `factored_sota` works, we can deprecate `factored_embed`.
