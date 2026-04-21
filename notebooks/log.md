# rec-system log

## 2026-04-11 — Two Towers vs SVD analysis & fixes

### Problem
Two Towers (04_two_towers.ipynb) appeared to be significantly beaten by the SVD baseline (03_svd_baseline.ipynb). SVD NDCG was ~2x better.

### Root cause analysis

**1. Evaluation protocol mismatch (primary reason metrics looked bad)**
- SVD evaluated with sampled negatives: 1 positive vs 99 random negatives = 100 candidates
- TT evaluated with full catalogue: 1 positive vs all ~44,582 items
- These are not comparable. SVD's task is ~450x easier by candidate pool size.

**2. InfoNCE loss poisoned by negative-labeled training rows (primary training issue)**
- Preprocessing (02_preprocessing.ipynb) creates label=1 (stars > user_mean) and label=0 (stars < user_mean)
- Positive rate is ~58.4%, so ~41.6% of train_reviews are label=0
- InfoNCE loss always treats diagonal pairs as positives regardless of label
- ~27k confirmed negative interactions per batch were being trained as positives
- Fix: filter train_reviews to label==1 before building TwoTowerDataset

**3. city_encoded / state_encoded are label-encoded categoricals scaled as continuous**
- LabelEncoder assigns arbitrary integers (0-918) to city names
- StandardScaler then treats these as a continuous variable with meaningful distances
- "Philadelphia"=574 and "Phoenix"=600 have no meaningful numerical relationship
- Geographic context already covered by category embeddings (Qwen3-Embedding-0.6B, 1024-dim)
- Fix: removed both columns from BUSINESS_SCALAR_COLS

### Changes made to 04_two_towers.ipynb
- Cell 8: removed `city_encoded` and `state_encoded` from BUSINESS_SCALAR_COLS (27 → 25 features)
- Cell 14: filter train_reviews to label==1 before TwoTowerDataset
- Cell 31 (new): added `evaluate_sampled()` — same 1-pos + 99-neg protocol as SVD
- Cell 32 (updated): added sampled eval calls after full-catalogue eval calls

### On negatives — why they're not used
InfoNCE can't correctly use explicit label=0 rows because they'd appear on the diagonal
(treated as positives). To use negatives properly, switch loss to BPR (pairwise ranking):
train on (user, pos_item, neg_item) triplets, maximize score(pos) - score(neg).
This is a meaningful future improvement.

### Data cleaning verdict
- Preprocessing pipeline (02_preprocessing.ipynb) is sound
- Temporal leave-last-out split is correct
- Median imputation for missing business attributes is acceptable
- Label normalization relative to user average is a good approach (addresses rating bias)
- Category embeddings from Qwen3-Embedding-0.6B are 1024-dim and semantically rich
- Only issue was city_encoded/state_encoded in how TT consumed the features (fixed above)

### Next steps if metrics still lag
- Implement BPR loss to properly leverage label=0 negatives
- Hard negative mining: after initial training, find high-scoring items user disliked
- Reduce dropout from 0.5 to 0.2-0.3 (may be too aggressive for 1.1M param model)
- Simplify CrossLayer from full matrix (input_dim²) to vector weight (DCN-v2 style)

---

## 2026-04-11 (second pass) — major architecture + training rewrite

User reported loss still stuck around log(batch_size) ≈ 11 after the first pass, meaning
the model was barely learning. Root causes:

1. **Batch 65536 meant only ~17 grad steps per epoch** — 850 total in 50 epochs.
   Severe undertraining. Dropped to 4096 → ~274 batches/epoch, ~13.7k grad steps.
2. **CrossLayer was massive** — `input_dim × input_dim` weight matrix, ~1M params per
   layer × 4 layers. Removed entirely; replaced with clean MLP + input LayerNorm.
   Modern two-tower retrieval at YouTube/Google scale uses plain MLPs.
3. **Val loss was meaningless** — val_dataset still had mixed labels but InfoNCE
   treats every diagonal as positive, so ~42% of val "positives" were actual
   negatives. Floor was way above 0. Fixed: filter val to label==1 too.
4. **Qwen embeddings had per-dim magnitude ~0.03** while scaled scalars have std ~1.
   First linear layer was mostly ignoring the 1024-dim embedding. Fixed: L2-normalize
   embeddings before concatenation, and added input LayerNorm in the tower.
5. **Dropout 0.5** too aggressive for an already-underfit model. → 0.2.
6. **Temperature 0.1** → 0.07 for sharper separation with L2-normed embeddings.
7. **Adam** → **AdamW with weight_decay=1e-5** for principled regularization.
8. **drop_last=True** on loaders so InfoNCE doesn't get a tiny final batch.

### Bug found from Sonnet's earlier pass
- Cell 23 training loop had an f-string with nested double quotes:
  `f"Change: {"↓" if delta < 0 else "↑"} {abs(delta):.4f}"`
  This is a SyntaxError on Python < 3.12 (Colab). The whole cell failed to parse,
  so train_losses was never initialized. Fixed by pre-computing the direction string.
