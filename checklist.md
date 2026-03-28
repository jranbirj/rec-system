## Project Checklist

### Phase 1 — Data

- [x] Download and explore Yelp Open Dataset
- [x] Filter and clean user and business data
- [ ] Feature engineering (user tower features, item tower features)
- [ ] Create stratified train/validation/test splits
- [ ] Build data preprocessing pipeline

### Phase 2 — Baseline Model

- [ ] Implement SVD matrix factorization model
- [ ] Train SVD on interaction data
- [ ] Evaluate baseline: Precision@K, Recall@K, NDCG@K, Hit Rate@K (k=5,10,20)

### Phase 3 — Two-Tower Model

- [ ] Build user tower DNN in PyTorch
- [ ] Build item tower DNN in PyTorch
- [ ] Implement contrastive loss training
- [ ] Train two-tower model
- [ ] Compare results vs. SVD baseline across all metrics

### Phase 4 — Experiments

- [ ] Ablation study: user tower feature subsets (all features, engagement-only, minimal)
- [ ] Ablation study: item tower feature subsets (full, ratings-only, general data)
- [ ] Negative sampling strategy 1: random sampling
- [ ] Negative sampling strategy 2: popularity-weighted sampling
- [ ] Negative sampling strategy 3: mixed/dynamic approach
- [ ] Compare NSS strategies across metrics

### Phase 5 — Vector DB & Demo

- [ ] Choose vector database (Qdrant / Milvus / Pinecone)
- [ ] Index item embeddings into vector DB
- [ ] Build ANN inference retrieval pipeline
- [ ] Deploy demo web interface for real-time recommendations
- [ ] Interpret integration results
