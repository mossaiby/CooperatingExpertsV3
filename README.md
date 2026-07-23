# Cooperating Experts v3 — From-Scratch Experts + Multi-Vector Bridge

Combines V1's approach (small, from-scratch transformers, each with its own
vocabulary, trained on its own corpus) with V2's multi-vector shared-space
bridge, trained on real CodeSearchNet data with zero synthetic templates and
zero LLM-based data generation.

## Why this exists

V2 (two frozen pretrained 1-3B models bridged by a compressed handoff)
repeatedly lost the `<switch:*>` token to a strongly pretrained competing
habit -- both backbones had seen billions of tokens of backtick-fenced code
in their pretraining, and fine-tuning a rarely-seen new token couldn't
out-compete that. From-scratch experts never see backtick fencing unless we
put it in the corpus, so that specific failure mode structurally can't
happen here. The honest tradeoff: ~20-30M-param models are a real
capability step down from 1-3B pretrained ones -- watch for a new "the
model is just too small" confound, the same way v2 needed a base-model
quality check before trusting its own results.

## What's different from v2's bug

v2 had a real bug where `<switch:*>` was computed but never actually
inserted into Phase-3 training data -- the model never saw it once. Here,
switch tokens are ordinary vocabulary entries from the very first tokenizer
training run (see `tokenizer.py`), and `dataset.py`'s `encode_session`
prepends them to every non-first session segment from the start.
`smoke_test.py` includes an explicit assertion that would have caught the
v2 bug immediately, so it can't silently recur.

## Run order

```bash
pip install -r requirements.txt

# 0. Cheap end-to-end check on tiny models + fake data, minutes not hours
python smoke_test.py

# 1. Real data: CodeSearchNet pairs + per-expert raw corpora
python data.py

# 2. Train per-expert BPE tokenizers (includes <switch:*> from the start)
python prepare_tokenizers.py

# 3. Phase 1: pretrain each expert independently
python train_pretrain.py --expert english
python train_pretrain.py --expert python

# 4. Phase 2: freeze both, train only the bridge
python train_stitch.py

# 5. Phase 3: unfreeze everything, train end-to-end on real interleaved
#    sessions (switch tokens genuinely present, up-weighted since rare)
python train_mixed.py --bridge-ckpt checkpoints/model_stitched_best.pt

# 6. Generate
python generate.py "def quicksort(arr):" --expert python \
    --mixed-ckpt checkpoints/mixed_best
```

All scripts accept `--debug` to run against the tiny config instead.
`--num-vectors` and `--dim` override the bridge's multi-vector width and
shared-space dimension on `train_stitch.py`/`train_mixed.py`, matching v2's
CLI conventions.

## What's simplified relative to a production implementation

- **No KV cache in generation** (`generate.py`): full-sequence recompute
  every step. Fine for these small models and short sequences; would need
  real caching before scaling up model size or sequence length.
- **No padding within a batch for phase-1 pretraining** -- batches are
  formed from fixed-length windows of a concatenated corpus, so there's no
  variable-length padding to handle. Phase 2/3 use `attention_mask` where
  variable lengths do occur (handoff pairs, sessions).
- **Windowed pretraining corpus doesn't respect example boundaries** --
  occasional windows span two different functions/docstrings back to back.
  Standard practice for from-scratch LM pretraining at this scale (same
  approach used by e.g. nanoGPT); not expected to matter much given typical
  window/example-length ratios, but worth knowing.

## Honest expectations

This removes one specific, now-confirmed failure mode (competing pretrained
habit crowding out the switch token) but does **not** by itself resolve the
other open question from v2: whether a compressed vector handoff (even
multi-vector) is fundamentally information-limited compared to full
per-token cross-attention. If generation quality here is still capped in a
similar way to v2's `factorial`/`is_palindrome` tests, that's evidence
pointing at the bridge design itself, now with the confound of pretrained
model competition removed -- a cleaner result either way.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters, including a fast `--debug` config |
| `data.py` | CodeSearchNet loading, cleaning, dedup, corpus extraction |
| `tokenizer.py` | Per-expert byte-level BPE tokenizer training + wrapper |
| `prepare_tokenizers.py` | Driver: trains both experts' tokenizers |
| `model.py` | From-scratch `Expert` transformer + multi-vector bridge API |
| `dataset.py` | Windowed corpus, handoff pairs, real interleaved sessions |
| `train_pretrain.py` | Phase 1: independent per-expert pretraining |
| `train_stitch.py` | Phase 2: frozen backbones, bridge-only training |
| `train_mixed.py` | Phase 3: full unfreeze, real sessions, switch tokens |
| `generate.py` | Generation with live switching (no KV cache) |
| `smoke_test.py` | Full pipeline check on tiny models + fake data |
