# Experiment: Factored Tied Embeddings

**Status:** Implementation complete, awaiting smoke test
**Started:** 2026-04-17
**Baseline to beat:** naive Play 2 reduced baseline (val_bpb 1.3413 at 1/320 compute) / eventually SOTA 1.0810

## The idea

Replace the single `[vocab_size, model_dim]` embedding table with a factored form:

```
tok_emb:    [vocab_size, embed_dim]     (smaller lookup table)
embed_proj: [embed_dim, model_dim]      (shared up-projection)

input:  x = embed_proj(tok_emb[ids])               # embed_dim -> model_dim
output: x = embed_proj.T @ x                        # model_dim -> embed_dim
        logits = tok_emb.weight @ x                 # -> vocab
```

The projection matrix is shared between input and output (transpose used at output), so only one extra matrix is added to the budget.

## Why it's worth trying

- **Proven at scale:** the 106M non-record Binary UNet submission uses `8192x254` factored tied embeddings with learned projections — tested and working.
- **Absent from the record track:** every record-track entry keeps full `[vocab, model_dim]` embeddings at int8.
- **Frees budget:** at SP8192 with MODEL_DIM=512 and EMBED_DIM=256:
  - Current embedding storage: `8192 * 512 * 1 byte (int8) = 4.20 MB`
  - New embedding + projection: `8192 * 256 * 1 + 512 * 256 * 1 = 2.10 MB + 0.13 MB = 2.23 MB`
  - **Savings: ~1.97 MB** — enough to add +1 physical transformer layer at int6

## Code changes vs baseline (25 lines added, nothing deleted)

1. **`EMBED_DIM` hyperparameter** — new env var, defaults to `MODEL_DIM` (no change when unset)
2. **`GPT.__init__`** — if `embed_dim < model_dim`, build `embed_proj` as `CastedLinear` and size `tok_emb` to `[vocab, embed_dim]`
3. **`GPT.forward` (input path)** — pass embedded tokens through `embed_proj` before RMS norm
4. **`GPT.forward` (output path)** — project down via `embed_proj.weight.T` before tied-logit matmul
5. **Optimizer routing** — add `embed_proj.weight` to the `matrix_params` list so Muon optimizes it (standard 2D weight, not a scalar/control tensor)

Minimal surgical changes. Backward compatible: setting `EMBED_DIM=MODEL_DIM` (or omitting it) yields byte-identical behavior to the baseline.

## Run configs

### Quick smoke test (~5 min, ~$0.05 on 1x RTX 4000 Ada)
Verifies the modified code compiles, runs, and produces non-NaN loss:

```bash
RUN_ID=fte_smoke \
NUM_LAYERS=2 MODEL_DIM=128 EMBED_DIM=64 \
TRAIN_SEQ_LEN=256 ITERATIONS=100 \
TRAIN_BATCH_TOKENS=16384 VAL_BATCH_SIZE=16384 \
VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Compare against naive baseline (~80 min, ~$0.35 on 1x RTX 4000 Ada)
Same config as our Play 2 run but with factored embeddings:

```bash
RUN_ID=fte_vs_play2 \
MODEL_DIM=512 EMBED_DIM=256 \
ITERATIONS=2000 \
VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Compare final `val_bpb` against Play 2's 1.3413 — any delta tells us whether factoring helps / hurts / is neutral at this scale.

### Full-scale record attempt (when grant credits arrive, 10 min, 8x H100)
Stack factored embeddings on top of the SOTA recipe. Requires porting the SOTA tricks (SP8192, GPTQ SDClip, parallel residuals, legal TTT) onto this script — separate work item.

## Known risks

- **Init scale:** `embed_proj.weight` uses PyTorch default init (`kaiming_uniform`). May need tuning — too large an init scale could blow up early training. To monitor.
- **LR interaction:** `embed_proj.weight` routes to Muon with default `matrix_lr=0.04`. Might want a separate learning rate group if training is unstable.
- **Tied-weight transpose at output:** the `F.linear(x, self.embed_proj.weight.to(x.dtype).t())` call creates a new tensor each forward pass. Minor memory/compute overhead; negligible at this scale but worth profiling if step time increases noticeably.
- **Quantization interaction:** `embed_proj.weight` is a `CastedLinear` weight (stored fp32 during training) — it falls through to per-row int8 quantization at serialization like other matrix params. Should be safe but first smoke test will confirm.

## Next steps after smoke test

If smoke passes (non-NaN loss, converges even slightly):
1. Run the 80-min "compare against naive baseline" config above
2. If val_bpb is close to or better than 1.3413 → good signal, proceed
3. If val_bpb is much worse → debug init scale or LR before giving up
4. Port factored embeddings onto the full SOTA recipe for the real record-track attempt
