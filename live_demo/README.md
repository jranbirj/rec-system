# Two-Tower Restaurant Recommendations — Live Demo

Local Gradio app for the two-tower model trained in `notebooks/04_two_towers_baseline_FINAL.ipynb`.

## Prerequisites

- macOS with **Python 3.10–3.13** (`python3 --version`)
- `pip` available (`python3 -m ensurepip --upgrade` if missing)
- `live_demo/qdrant_export/` populated with notebook 04's output:
  ```
  qdrant_export/
  ├── user_embeddings.npy
  ├── user_embedding_index.parquet
  ├── business_embeddings.npy
  ├── business_embedding_index.parquet
  ├── two_tower_best.pt
  └── model_meta.json
  ```
- Yelp dataset JSONs from https://www.yelp.com/dataset, placed at `data/yelp/`:
  - `data/yelp/yelp_academic_dataset_business.json`
  - `data/yelp/yelp_academic_dataset_user.json`
- `data/train_reviews.parquet` (output of notebook 02).

## Setup

From `live_demo/`:

```bash
# 1. Create venv + install deps.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Build the Qdrant collection + display metadata.
python3 build_qdrant.py \
  --export-dir     qdrant_export \
  --train-reviews  ../data/train_reviews.parquet \
  --yelp-json      ../data/yelp/yelp_academic_dataset_business.json \
  --yelp-user-json ../data/yelp/yelp_academic_dataset_user.json

# 3. Launch.
python3 app.py
```

First launch downloads the Qwen embedder (~1.2GB, one time). Open the URL Gradio prints (`http://127.0.0.1:7860`).

## Subsequent runs

```bash
cd live_demo
source .venv/bin/activate
python3 app.py
```
