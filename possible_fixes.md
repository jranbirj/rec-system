# possible-fixes — summary of implementation changes

Branch: `possible-fixes` (based on `TTA`, commit `fcefb98`).
Goal: SOTA-style training tweaks to lift two-tower recall without going deeper in the NN.

Three groups of changes (C1 = training signal, C2 = leakage fix, C6 = EMA weights)
landed in one commit across three files. Within each file, items are ranked by
expected impact on recall / importance.

---

## `tt_model.py`

1. **`InfoNCELoss` gains `item_log_q` (logQ correction).** Subtract `log P(item)` per
   column of the similarity matrix before softmax. Corrects popularity bias in
   in-batch negatives (Yi et al. 2019 — the standard Google/YouTube retrieval fix).
2. **`InfoNCELoss` gains `symmetric=True`.** Average user→item and item→user
   cross-entropy. Doubles supervision per pair (CLIP-style), essentially free.
3. **New `EMA` class.** Keeps a smoothed copy of weights (`decay=0.999`), updated
   every optimizer step. Eval swaps EMA weights in. Typically +1–3% recall.
4. **`TwoTowerDataset` carries per-sample `item_log_q`.** Infrastructural —
   required to plumb logQ through the loader. Backwards-compatible when
   `item_log_q=None`.

---

## `run_tt_local.py`

1. **`USE_CLEAN` flag + suffixed artifact paths.** Toggle between the original
   (leaky) and rebuilt (train-only) user features / labels. `True` by default.
   This is what makes the leakage fix actually usable without touching the rest
   of the pipeline.
2. **Item-frequency precompute → `item_log_q` dict.** From train-positives only,
   `log P(item)`. Passed into `TwoTowerDataset` and consumed by `InfoNCELoss`.
3. **Warmup + cosine LR schedule.** Replaced `ReduceLROnPlateau` with `LambdaLR`
   doing 500 linear-warmup steps → cosine decay to zero over all epochs. Steps
   every batch instead of every epoch. Standard for contrastive training.
4. **Batch size 4096 → 8192.** More in-batch negatives = stronger contrastive
   gradient. Single constant change.
5. **EMA integration.** Instantiate after model creation, update after each
   `optimizer.step()`, swap EMA weights in for validation (restored after), save
   `ema_state` in the checkpoint, load EMA shadow at eval time.
6. **`symmetric=True` passed to `InfoNCELoss`.** Activates the symmetric loss
   addition added in `tt_model.py`.
7. **Removed per-epoch `scheduler.step(vl)`.** The new scheduler is per-step;
   the old plateau-scheduler call would corrupt it.

---

## `scripts/rebuild_features_trainonly.py` (new)

1. **Recompute `user_mean` from train rows only** and re-derive `label` on all
   three splits using that mean. Fixes the primary leakage: the original
   `user_mean` was computed across train+val+test. Writes
   `data/{train,val,test}_reviews_clean.parquet`.
2. **Recompute per-user Qwen category embedding** as the mean of
   `business_category_embeddings` over each user's **train-only label==1**
   items. Fixes the secondary leakage: the original per-user Qwen feature was
   averaged over every review the user ever made, including val/test items.
   Writes `data/data_TTA/user_category_embeddings_trainonly.npy` and
   `data/data_TTA/user_embedding_index_trainonly.parquet`.
3. **Drops cold-start users** (present in val/test but absent from train) during
   re-labeling. Required because without train rows there is no train-only
   `user_mean` to threshold against.

---

## Not changed

- No notebooks modified (`01_eda`, `02_preprocessing`, `03_svd_baseline`,
  `04_two_towers`, `05_experiments` untouched).
- No new dependencies added.
- `Tower`, `TwoTowerModel`, `USER_SCALAR_COLS`, `BUSINESS_SCALAR_COLS`,
  `pick_device` in `tt_model.py` are unchanged — NN architecture is identical.
- Existing data artifacts not overwritten — rebuild script writes new files
  with `_clean` / `_trainonly` suffixes.

---

## Commits

- `TTA` → `2c19530`: committed the previously-untracked `run_tt_local.py`,
  `run_svd_local.py`, `tt_model.py` as a baseline so `possible-fixes` can diff
  cleanly against them.
- `possible-fixes` → `fcefb98`: everything above (+164 / −24 across 3 files).

Neither commit has been pushed.
