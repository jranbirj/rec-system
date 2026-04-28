# Yelp Recommender System

Two-tower recommendation system trained on Yelp U.S. data with Qdrant vector database integration and a local Gradio demo.

This README aggregates the notebook run instructions and the live demo setup. Run the notebooks first to produce embeddings and the Qdrant export, then follow the Live Demo section to serve recommendations locally.

---

## Notebooks

### Dataset

Yelp Open Dataset: https://business.yelp.com/data/resources/open-dataset/

Requires ~4GB of space; the full project needs ~10GB. Download the bundle and move the files into a Google Drive folder. Only three of the bundled files are used:

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

`checkin.json` and `tip.json` are not used.

### Environment

The five notebooks were written and run on Google Colab. Open each in Colab, mount Drive, and run top-to-bottom. The only edits needed are the path strings (Drive folder for the raw JSON, and the output directory each notebook writes to).

Known environment issues:

- The Colab kernel sometimes needs `numpy` reinstalled. One reinstall + kernel restart usually fixes it.
- Notebooks 04 and 05 had PyTorch install issues. Each now has a `pip install` cell at the top to handle this.

### Run order

Run in order through 03. 04 and 05 can be run without 03, but the comparison table at the bottom of 04 pulls from 03's output. 04 and 05 can run side by side.

### 01 — EDA (`01_eda.ipynb`)

Inspects business, review, and user JSON. Filters businesses to restaurants/food, computes review-count and star-rating distributions, runs k-core statistics at k = 5, 6, 7, and reports user-side coverage.

To run: edit the three path strings (`yelp_business_data_path`, `yelp_reviews_data_path`, `yelp_users_data_path`) to point at your Drive copy of the JSON files.

Produces no files — purely for inspection.

### 02 — Preprocessing (`02_preprocessing.ipynb`)

Produces every file the later notebooks read. Filters to open restaurants, applies k=5 k-core filtering iteratively until stable, builds user features (`avg_stars_given`, `review_count_log`, `yelping_since_years`, `is_elite`, `avg_useful_per_review`, `social_activity_log`, post-split `user_price_preference`), builds business features, generates Qwen category embeddings for users and businesses, and writes the train/val/test split.

Change these paths:

1. Path to `yelp_academic_dataset_business.json`
2. Path to `yelp_academic_dataset_review.json`
3. Path to `yelp_academic_dataset_user.json`
4. `save_dir` — output directory. Notebooks 03, 04, and 05 all read from this.

Outputs (in `save_dir`):

- `train_reviews.parquet`, `val_reviews.parquet`, `test_reviews.parquet`
- `users.parquet`, `businesses.parquet`
- `user_category_embeddings.npy`, `business_category_embeddings.npy`
- `user_embedding_index.parquet`, `business_embedding_index.parquet`

### 03 — SVD baseline (`03_svd_baseline.ipynb`)

Trains a Surprise SVD model on train and evaluates on val and test under two protocols: sampled (1 positive + 99 negatives) and full-catalogue (rank against all ~39k items). Reports NDCG@K, Precision@K, Recall@K for K = 5, 10, 20.

To run:

1. Set `origin_dir` to the `save_dir` from notebook 02.
2. Set `results_dir` to wherever you want the metric CSVs written.

Seed is fixed at 42 throughout the project for reproducibility.

Outputs: `svd_val_metrics_k100.csv`, `svd_test_metrics_k100.csv`, `svd_val_metrics_fullcat.csv`, `svd_test_metrics_fullcat.csv`.

### 04 — Two-tower baseline (`04_two_towers_baseline.ipynb`)

Trains the two-tower model. Each tower is LayerNorm → Linear(512) → LayerNorm → GELU → Dropout → Linear(256) → LayerNorm → GELU → Dropout → Linear(128), with L2-normalized 128-d output. Loss is InfoNCE with in-batch negatives. User and business inputs are scalar features concatenated with L2-normalized Qwen category embeddings. Scalers are fit on train-set users/businesses only to avoid leakage. Exports trained item embeddings to a Qdrant collection.

Change these paths:

1. `DATA_DIR` — `save_dir` from notebook 02.
2. `RESULTS_DIR` — output for metrics and checkpoints.
3. `EXPORT_DIR` — output for Qdrant export files (consumed by the live demo).
4. At the bottom of the notebook, the comparison cell reads notebook 03's results directory. Set this to 03's `results_dir` for the side-by-side SVD vs two-tower table (optional).

A GPU runtime is required.

### 05 — Experiments

Two versions exist:

- `05_experiments.ipynb` — full experiment grid.

Training/eval library, sweep through the experiment configs (A0 baseline, then B variants), mine hard negatives from A0's label==0 scores for B2, and write a unified `experiments.parquet` table plus per-run plots, configs, and history.

Change these paths:

1. `DATA_DIR` — `save_dir` from notebook 02.
2. `RESULTS_DIR` — top-level output directory. `CKPT_DIR`, `CONFIG_DIR`, `HISTORY_DIR`, `PLOTS_DIR` are created as subdirectories.

A GPU runtime is required, nothing above a T4 did we find necessary for any of this.

Outputs in `RESULTS_DIR`: `experiments.parquet`, `experiments.csv`, `final_summary.csv`, `cold_start_slice.csv`, `pip_freeze.txt`, plus per-run plots, configs, history, and checkpoints in their respective subdirectories.

---

## Live Demo

Local Gradio app for the two-tower model trained in `notebooks/04_two_towers_baseline.ipynb`. The vector database is built from notebook 04's `EXPORT_DIR`.

### Prerequisites

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
- Yelp dataset JSONs from https://www.yelp.com/dataset, Yelp JSON folder placed in root of `"rec-system`:
  - `data/yelp/yelp_academic_dataset_business.json`
  - `data/yelp/yelp_academic_dataset_user.json`
- `data/train_reviews.parquet` (output of notebook 02).

### Setup

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

these steps should take no more than 2-3 minutes each, except when loading Qwen embedder for first time.

First launch downloads the Qwen embedder (~1.2GB, one time). Open the URL Gradio prints (`http://127.0.0.1:7860/`).

### Subsequent runs

```bash
cd live_demo
source .venv/bin/activate
python3 app.py
```
