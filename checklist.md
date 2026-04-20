## Project Checklist

### Phase 1 — Data ✅

- [x] Download and explore Yelp Open Dataset (01_eda.ipynb)
- [x] Filter and clean user and business data
- [x] Feature engineering (user tower features, item tower features)
- [x] Create stratified train/validation/test splits (temporal leave-last-out)
- [x] Build data preprocessing pipeline (02_preprocessing.ipynb)
- [x] Qwen3-Embedding-0.6B category embeddings (1024-dim)

### Phase 2 — Baseline Model ✅

- [x] Implement SVD matrix factorization (03_svd_baseline.ipynb)
- [x] Train SVD on interaction data
- [x] Evaluate: Precision/Recall/NDCG/HitRate @ {5,10,20} with 100-candidate sampled eval

### Phase 3 — Two-Tower Model (in progress)

- [x] User tower DNN (PyTorch)
- [x] Item tower DNN (PyTorch)
- [x] InfoNCE contrastive loss training
- [x] Training fixes (2026-04-11): label==1 filter, removed city/state_encoded,
      batch 65536→4096, dropped CrossLayer, L2-norm Qwen embeddings, dropout 0.5→0.2,
      AdamW, temp 0.07, val label filter, f-string syntax fix
- [ ] **Re-run end-to-end with all fixes and record sampled-eval metrics**
- [ ] **Head-to-head TT vs SVD on identical 100-candidate sampled eval**

> ⚠️ **Roadblock risk — Phase 3**
> - Sampled-eval comparison is the load-bearing result for the whole project; if TT still
>   underperforms SVD after fixes, Phase 4 ablations lose their framing.
> - Full-catalogue eval (~44k items) is slow — keep sampled eval as the primary metric,
>   full-catalogue as a secondary.
> - InfoNCE still can't use label=0 rows — any "negatives experiment" requires BPR first.

### Phase 4 — Experiments (05_experiments.ipynb — empty)

- [ ] Ablation: user tower features (all / engagement-only / minimal)
- [ ] Ablation: item tower features (full / ratings-only / general)
- [ ] NSS 1: random sampling (current InfoNCE in-batch is effectively this)
- [ ] NSS 2: popularity-weighted sampling
- [ ] NSS 3: mixed/dynamic (e.g., hard negatives from top-scoring label=0 rows)
- [ ] BPR pairwise loss variant to leverage the ~42% label=0 rows
- [ ] Compare NSS strategies across all metrics

> ⚠️ **Roadblock risk — Phase 4**
> - Each ablation = full retrain. Budget compute; cache preprocessed tensors.
> - Hard-negative mining needs a first-pass trained model — sequential dependency.
> - Popularity-weighted sampling requires a precomputed item-frequency table from train split only (leakage risk if built from full data).
> - BPR changes loss signature → evaluation + dataset code both need a parallel path.

### Phase 5 — Vector DB & Demo (06_evaluation.ipynb — empty)

- [ ] Choose vector DB (Qdrant / Milvus / Pinecone)
- [ ] Export trained item tower → index 44k item embeddings
- [ ] ANN retrieval pipeline (user embedding → top-K items)
- [ ] Demo web interface for real-time recs
- [ ] Interpret integration results

> ⚠️ **Roadblock risk — Phase 5**
> - Item tower must be frozen and versioned before indexing — re-training invalidates the index.
> - Cold-start users have no embedding; decide fallback (popularity, category prior) up front.
> - Pinecone is managed (account/keys); Qdrant/Milvus are self-hostable — pick based on demo deployment target before indexing.
> - User features at inference must match training schema exactly (same scalers, same encoders) — persist the preprocessing pipeline alongside the model.
