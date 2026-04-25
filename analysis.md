# Project analysis — bugs, leaks, methodology

Read of the four notebooks (`02_preprocessing`, `03_svd_baseline`, `04_two_towers`, `05_experiments`) plus the supporting `log.md`, `possible_fixes.md`, and `checklist.md`. The `.py` files in `old/` (`tt_model.py`, `run_tt_local.py`, `scripts/rebuild_features_trainonly.py`) are NOT under active development — they are referenced here only as suggestions for how to change the notebooks.

Items are ordered by impact on the project. Tier 1 items can change the conclusions you draw from the experiments; Tier 2 items are likely costing measurable performance; Tier 3 is hygiene.

---

## Tier 1 — Things that meaningfully invalidate results

### 1. The label itself is a questionable choice (`02_preprocessing.ipynb`)

`label = (stars > user_mean)`, with `user_mean` being that user's average across their train rows. Everything downstream — InfoNCE positives, BPR triplets, "positive rate", ablation metrics — rests on this.

Problems:
- A 3-star review is "positive" for a 2-star-mean user and "negative" for a 4-star-mean user. The model learns "did this user rate this above their typical?", **not** "did this user like it". Two different users can have opposite labels for the same star value.
- By construction the per-user positive rate hovers around 50%, not the true preference distribution. The global ~58% positive rate just reflects how the threshold splits each user's distribution; it does not reflect "users like 58% of restaurants they review".
- Ties (`stars == user_mean`) are silently dropped, shrinking val/test.
- This conflicts with InfoNCE's assumption that positives are confidently positive. Many of your "positives" are 3-star reviews that the model is being told to pull together with the user.

What recsys literature usually does: absolute threshold (`stars >= 4`), or treat any review as implicit positive (clicked/visited), or use the rating directly as a regression/ordinal target. The current relative scheme is unusual and not justified anywhere except the comment "addresses rating bias" — which it does at the cost of meaningfulness.

This is the most impactful "choice we shouldn't have made". Recommend running A0 with `label = stars >= 4` once just to see the delta. It's a 5-line change and is the most likely explanation for retrieval metrics being dominated by an SVD that's just predicting ratings.

### 2. `05_experiments.ipynb` evaluates every ablation on `test_reviews` — test set is being repeatedly peeked at

`run_experiment(...)` in [05_experiments.ipynb](notebooks/05_experiments.ipynb) calls `evaluate_sampled(model, test_reviews, ...)` and `evaluate_ranking(model, test_reviews, ...)`. Every config in `CONFIGS` (A0, A1, A2, A3, A4, B1, B2, C1) gets evaluated on test. By the time you select "the best run per family", test has been used as a model-selection signal 8 times.

Standard practice: ablate on **val**, then evaluate the final pick on **test** exactly once. Right now the test numbers in `experiments.parquet` are val-like, not test-like — they overstate generalization.

Fix: change `test_reviews` to `val_reviews` in `run_experiment`, then add a single final cell that loads each champion's checkpoint and evaluates on test once.

### 3. There is still a feature leak we didn't catch

`user_price_preference` in [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) is computed by averaging `price_range` across all of `yelp_reviews_df` — that includes the rows that later become val and test. Same class of leak as the original `user_mean` and the original Qwen user embedding.

`possible_fixes.md` and the rebuild script only fix `user_mean` and the Qwen user embedding. `user_price_preference` is still a feature on every user that encodes their val+test reviews. It's a small leak (one scalar) but it's a real leak and it's still there in every "clean" path. Either compute it from train rows only, or drop the feature.

### 4. `04_two_towers.ipynb` is missing the improvements documented in `log.md` / `possible_fixes.md`

The notebook does not contain:
- logQ sampling-bias correction in `InfoNCELoss` (Yi et al. 2019)
- Symmetric InfoNCE (averaging `user→item` and `item→user` cross-entropy)
- EMA weights for evaluation
- Warmup → cosine LR schedule (notebook still uses `ReduceLROnPlateau`)
- Train-only "clean" features (the leak fix described in `possible_fixes.md`)

These exist in the `old/` scripts as a reference but the notebook is the artifact you're shipping. Either port them into the notebook, or document clearly that the notebook is the un-fixed baseline and the scripts are the fixed version. Right now the two paths give different numbers and `log.md` reports both interleaved.

### 5. TT vs SVD full-catalogue head-to-head is not apples-to-apples

`evaluate_ranking` in `04_two_towers.ipynb` ranks against `business_features_dict` (~44k items, every restaurant in the corpus). `evaluate_full_catalogue` in `03_svd_baseline.ipynb` ranks against `trainset.n_items` (~39k items, only ones SVD has factors for). The TT pool is ~13% larger, so TT's full-cat numbers are slightly worse than they should be in the comparison.

This was patched for sampled eval (`train_item_set` filter in `evaluate_sampled`) but not for full-cat. `log.md` acknowledges this but the code wasn't updated. Restrict TT's full-cat candidate pool to items that exist in `train_reviews`.

---

## Tier 2 — Likely costing measurable performance

### 6. Median-imputing binary attributes is near-equivalent to dropping the feature

`good_for_kids`, all `ambience_*`, all `good_for_meal_*`, `outdoor_seating`, etc. — these are mostly NaN with sparse 0/1 fills. Median of a mostly-NaN binary column is 0 or 1; we fill all NaNs with that, then StandardScale, and the column is now near-constant for the majority of rows.

A column that's "missing for 80% of businesses, True for 15%, False for 5%" becomes: imputed-True for 80%+15%, scaled to z-score, and the model can no longer distinguish "actually True" from "we don't know". Effectively dead features.

Either drop them, or add a `*_missing` indicator column. If we do nothing, ~20 of our 25 business scalar features carry almost no signal, which is consistent with the architecture decisions effectively betting on the Qwen 1024-dim doing the heavy lifting.

### 7. The 1024-dim Qwen embedding for category strings is overkill and unbalanced

We feed `Tower(input_dim = 25 + 1024)` through `LayerNorm(input_dim)` and `Linear(1049 → 512)`. ~98% of the input vector is the Qwen embedding. The L2-normalization step in [04_two_towers.ipynb](notebooks/04_two_towers.ipynb) helps, but then `LayerNorm` immediately re-normalizes, partly undoing it.

The category strings here are like `"Italian, Pizza, Pasta"` — finite vocabulary (a few hundred unique categories). A 64-dim learned embedding over a multi-hot category vector would be 1/16 the dimension, more interpretable, faster, and probably as good. Qwen-0.6B was the wrong tool for short, finite-vocab categorical strings.

If we keep Qwen, at least project it (e.g. a learnable `Linear(1024 → 64)`) before concatenating with the 25 scalars. Otherwise the first layer is structurally biased toward ignoring the scalars.

### 8. `B1` (popularity-weighted negatives) is implemented as popularity-weighted *positives*

In [05_experiments.ipynb](notebooks/05_experiments.ipynb) `run_experiment` B1 branch, `WeightedRandomSampler(weights=row_w, ...)` weights training rows (i.e., `(user, pos_item)` pairs) by item popularity. That oversamples popular items as **positives**, which then makes in-batch negatives popular too — but it also distorts the training distribution in a way that's not what word2vec does.

Word2vec popularity-weighted negative sampling weights only the *negative* draws, keeping the positive distribution untouched. To replicate that here, we'd need to keep `PairDataset` random-shuffled and instead, inside `InfoNCELoss`, draw extra negatives from a popularity-biased index. The current B1 is closer to "biased data" than "biased negatives". The ablation comparison is not measuring what its name claims.

### 9. `train_items_per_user` in `04_two_towers.ipynb` is built with `iterrows()` over 1.9M rows

Cell ~30 in the notebook does:

```python
for _, row in train_reviews.iterrows():
    uid = row["user_id"]; bid = row["business_id"]
    if bid in business_id_to_idx:
        train_items_per_user[uid].add(business_id_to_idx[bid])
```

This is ~5+ minutes on a fresh kernel for what should be a 2-second `groupby`. Replace with `train_reviews.groupby("user_id")["business_id"].apply(set)`. Cosmetic in terms of correctness, but every notebook re-run pays the cost.

### 10. `device = "cuda" if cuda else "cpu"` but `torch.amp.autocast("cuda")` is hardcoded

[04_two_towers.ipynb](notebooks/04_two_towers.ipynb) and [05_experiments.ipynb](notebooks/05_experiments.ipynb) wrap forward passes in `torch.amp.autocast("cuda")` and use `torch.amp.GradScaler("cuda")`, even though the device fallback is CPU. On CPU these will error. On MPS (local Mac), they error with a different message. The notebooks only run on Colab GPU; nothing local works. Either guard `autocast` with `if device.type == "cuda":` or remove the autocast entirely if Colab is the only target.

### 11. `drop_last=True` on val_loader

With ~92k val positives and `batch_size=4096` you have ~22 batches; dropping the last loses ~4k val rows from the loss-on-val signal each epoch. Not catastrophic (used only for early stop), but if we're tracking val_loss to pick a checkpoint, the dropped tail is a small source of variance. For full-cat / sampled eval this doesn't apply — those re-iterate the full df.

---

## Tier 3 — Sloppy / will bite us later

### 12. `06_evaluation.ipynb` is empty, `05_experiments.ipynb` has been the workhorse

The phasing diagram in `checklist.md` puts ablations in 05 and final eval in 06, but 05 is doing both. Either delete 06 or move final-test eval there.

### 13. Three different seeds across the project

`SEED=42` in 03 and 04, `SEED=340` in 05. Anything seeded (negative sampling, init, dropout RNG) is not aligned across notebooks. Comparing TT-from-04 numbers to A0-from-05 numbers is an apples-vs-oranges seed comparison, not just an architecture comparison. Pick one project-wide seed.

### 14. No popularity / "always recommend top-K" baseline

The whole project is framed as TT-vs-SVD. With a long-tailed restaurant distribution, "recommend the top-K most-reviewed restaurants in the user's city" is often a brutal baseline that beats both. Without it we don't know whether either model is doing anything interesting.

### 15. No multi-seed variance reporting

Every ablation runs once. Differences smaller than ~10-15% NDCG@10 between configs may just be seed noise. The `run_experiment` results table doesn't have a `seed` axis to repeat over.

### 16. `BPRLoss` uses raw dot products on L2-normalized embeddings

`TwoTowerModel.forward` normalizes both towers, so `(u·p - u·n)` is bounded in [-2, 2]. After `logsigmoid`, the per-step loss has a much smaller dynamic range than InfoNCE. We may need a temperature on BPR too, or it'll plateau. Worth checking the C1 loss curve before drawing conclusions about BPR-vs-InfoNCE.

### 17. `categories_clean` filtering produces empty strings

Anything that's a "food/restaurant" by the open-restaurant filter but has no category in `yelp_food_taxonomy` becomes `""` and gets the Qwen embedding of `""` — same vector for every such business. Not common (we filtered for restaurants/food upstream) but not zero. Sanity check `(yelp_open_restaurants_df["categories_clean"] == "").sum()`.

### 18. Dead imports + dead code

- `from collections import defaultdict`, `cross_validate` in `03` — unused.
- `test_dataset = TwoTowerDataset(test_reviews, ...)` in `04` is built and never used.
- The hand-curated `CITY_ALIASES` map (~80 entries) in `03` for the demo. The model itself doesn't use city; this is purely UI-side and it's brittle. Dropdown can be built without the alias map.

### 19. `evaluate_sampled` everywhere uses `for _, row in df.iterrows()`

~92k iterations per split, with a small GPU forward pass inside each. Per `log.md` this takes ~74s on the SVD side — fine for a one-shot eval, painful when iterating. Vectorize: stack all positives + all sampled negatives, run one big batched forward.

### 20. `state_encoded` and `city_encoded` are still saved in `businesses.parquet`

We correctly removed them from `BUSINESS_SCALAR_COLS` in `04`, but [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) still writes them into `businesses.parquet`. Anyone (or any future ablation) that does `BUSINESS_SCALAR_COLS_ALL = businesses_df.columns.tolist()` will silently re-introduce them. Either drop them at preprocessing time or one-hot encode them properly in 02 itself.

---

## Direction-of-the-project concerns

1. **We are stacking patches on top of a load-bearing label that is itself suspect.** Every fix in `log.md` and `possible_fixes.md` (logQ, EMA, warmup-cosine, batch up, train-only features, BPR, hard negs) is a reasonable retrieval-training tweak — but it's all downstream of a label that says "this user rated this restaurant above their personal mean". If the label is wrong, none of these fixes will produce a model that recommends restaurants people will like; they'll produce a model that recommends restaurants people will rate-higher-than-their-typical, which is an oddly specific prediction problem. Run one ablation with `label = stars >= 4` before doing anything else.

2. **There are two divergent implementations of the same model.** The notebook (`04`) is what `checklist.md` treats as "the model"; the local script (`run_tt_local.py`) is the one with all the post-fix improvements. We've decided to work in the notebooks — that means the script-only improvements (logQ, EMA, cosine, symmetric, USE_CLEAN) need to be ported into `04` or explicitly written off. Right now `log.md` describes fixes that exist only in the script, and `checklist.md` Phase 3 says "Re-run end-to-end with all fixes" — but the notebook doesn't have them.

3. **The "sampled eval vs full-cat eval" gap is being hidden, not interpreted.** Sampled R@20 ≈ 0.40 for SVD, full-cat R@20 ≈ 0.002. That's a 200× gap and it's the same model. The honest framing is: "100-candidate sampled metrics measure pairwise scoring quality; full-catalogue measures retrieval. SVD does the first OK and the second poorly because it was never a retrieval model." Phase 4 should report only full-cat as the headline; sampled is a sanity check.

4. **No popularity / no-personalization baseline.** Given that the label is per-user-relative, "recommend the user's already-favorite categories" might be most of the signal. We don't know how much "personalization" the TT is actually adding versus just memorizing popular-in-category.

5. **Cold-start is silently dropped at every step** (in preprocessing, in `evaluate_ranking`, in feature dict joins). For a real demo, ~5-15% of evaluation users vanish from the metrics. Either commit to evaluating only on warm users (and say so) or define a cold-start fallback.

6. **`05_experiments.ipynb` is doing all of: ablations, hard-neg mining, cold/warm slicing, plotting, summary tables — in one giant notebook.** The cell dependencies are not obvious (`hard_negs_a0.parquet` only exists after A0 has run, B2 fails until then). If we re-run from top, B2 will fail the first time. Either order the cells so this works on a fresh kernel, or split mining into a separate `05a_` notebook.

---

## Recommended order of fixes (highest-ROI first)

1. Switch `label` to `stars >= 4` and re-run A0. Compare against current A0.
2. Move ablation eval in `05` from `test_reviews` to `val_reviews`. Run A0 again, write down the val numbers, save test for the end.
3. Fix the `user_price_preference` leak in `02_preprocessing.ipynb` (compute from train rows only, or drop the feature).
4. Port the `.py`-only improvements (logQ, symmetric InfoNCE, EMA, warmup-cosine LR) into `04_two_towers.ipynb`. Then delete the `.py` scripts so there is one source of truth.
5. Add a popularity-only baseline ("top-K reviewed restaurants in user's modal city") to `03_svd_baseline.ipynb` or a new cell in `05`.
6. Drop the binary attributes that are >70% NaN, or add `*_missing` indicators. Re-run A0.
7. Project the Qwen 1024-dim through a learnable `Linear(1024 → 64)` before concat. Cuts param count, balances input.
8. Vectorize `evaluate_sampled` and the `iterrows` in `train_items_per_user`.
