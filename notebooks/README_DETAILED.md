# Yelp Recommender System — Notebooks

## Dataset

Yelp Open Dataset: https://business.yelp.com/data/resources/open-dataset/

Note that it requires about 4GB of space. These must be downloaded and then moved all into a Google Drive folder. This whole project requires about 10GB of space total.

The Yelp download is a single bundle. Out of the files inside it, the notebooks only read three:

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

The remaining files (`checkin.json`, `tip.json`) are not used.

## Environment

The five notebooks were written and run on Google Colab. Open each one in Colab, mount your Drive, and run top-to-bottom. The only edits you should need are the path strings (drive folder for the raw JSON files, and the output directory each notebook writes to).

Known environment issues we hit:

- The Colab kernel sometimes needs `numpy` reinstalled. One reinstall + kernel restart usually fixes it.
- Notebooks 04 and 05 had PyTorch install issues. Each notebook now has a `pip install` cell at the top to handle this, should no longer be an issue. 

## Run order

The notebooks must be run in order, until notebook 03. 04 and 05 can be run without 03, but there is a comparison table in 04_two_towers_baseline that pulls from 03_svd_baseline. 
04_two_towers_baseline and 05_experiments can be run side by side. 

### 01 — EDA (`01_eda.ipynb`)

Inspects the business, review, and user JSON files. Filters businesses to restaurants/food, computes review-count and star-rating distributions, runs k-core statistics at k = 5, 6, 7 to inform the preprocessing cutoff, and reports user-side coverage.

To run: edit the three path strings (`yelp_business_data_path`, `yelp_reviews_data_path`, `yelp_users_data_path`) to point at your Drive copy of the JSON files.

This notebook produces no files. It is purely for inspection.

### 02 — Preprocessing (`02_preprocessing.ipynb`)

Produces every file the later notebooks read. Filters to open restaurants, applies k=5 k-core filtering iteratively until stable, builds user features (`avg_stars_given`, `review_count_log`, `yelping_since_years`, `is_elite`, `avg_useful_per_review`, `social_activity_log`, and a post-split `user_price_preference`), builds business features, generates Qwen category embeddings for users and businesses, and writes the train/val/test split.

To run, change these paths:

1. Path to `yelp_academic_dataset_business.json`
2. Path to `yelp_academic_dataset_review.json`
3. Path to `yelp_academic_dataset_user.json`
4. `save_dir` — the output directory. Notebooks 03, 04, and 05 all read from this directory.

Outputs written to `save_dir`:

- `train_reviews.parquet`, `val_reviews.parquet`, `test_reviews.parquet`
- `users.parquet`, `businesses.parquet`
- `user_category_embeddings.npy`, `business_category_embeddings.npy`
- `user_embedding_index.parquet`, `business_embedding_index.parquet`

### 03 — SVD baseline (`03_svd_baseline.ipynb`)

Trains a Surprise SVD model on the train split and evaluates on val and test under two protocols: sampled (1 positive + 99 negatives) and full-catalogue (rank against all ~39k items). Reports NDCG@K, Precision@K, Recall@K for K = 5, 10, 20.

To run:

1. Set `origin_dir` to the `save_dir` from notebook 02.
2. Set `results_dir` to wherever you want the metric CSVs written.

Seed is fixed at 42 throughout the project for reproducibility.

Outputs: `svd_val_metrics_k100.csv`, `svd_test_metrics_k100.csv`, `svd_val_metrics_fullcat.csv`, `svd_test_metrics_fullcat.csv`.

### 04 — Two-tower baseline (`04_two_towers_baseline_FINAL.ipynb`)

Trains the two-tower model. Each tower is LayerNorm → Linear(512) → LayerNorm → GELU → Dropout → Linear(256) → LayerNorm → GELU → Dropout → Linear(128), with L2-normalized 128-d output. Loss is InfoNCE with in-batch negatives. User and business inputs are scalar features concatenated with L2-normalized Qwen category embeddings. Scalers are fit on train-set users/businesses only to avoid leakage. Exports the trained item embeddings to a Qdrant collection.

To run, change these paths:

1. `DATA_DIR` — point to the `save_dir` from notebook 02.
2. `RESULTS_DIR` — output directory for metrics and checkpoints.
3. `EXPORT_DIR` — output directory for the Qdrant export files.
4. At the bottom of the notebook, the comparison cell reads notebook 03's results directory. Set this to 03's `results_dir` if you want the side-by-side SVD vs two-tower table.

A GPU runtime is required.

### 05 — Experiments

Two versions exist:

- `05_experiments_FINAL.ipynb` — full experiment grid.
- `05_experiments_FINAL_noC1.ipynb` — same orchestration with the C1 experiment removed.

Both run the same training/eval library, sweep through the experiment configs (A0 baseline, then B/C/D variants), mine hard negatives from A0's label==0 scores for B2, and write a unified `experiments.parquet` table plus per-run plots, configs, and history.

To run, change these paths:

1. `DATA_DIR` — point to the `save_dir` from notebook 02.
2. `RESULTS_DIR` — top-level output directory. The notebook also creates `CKPT_DIR`, `CONFIG_DIR`, `HISTORY_DIR`, `PLOTS_DIR` as subdirectories of this.

A GPU runtime is required.

Outputs in `RESULTS_DIR`: `experiments.parquet`, `experiments.csv`, `final_summary.csv`, `cold_start_slice.csv`, `pip_freeze.txt`, plus per-run plots, configs, history, and checkpoints in their respective subdirectories.

## Live demo (Gradio + Qdrant)

The Gradio app and the Qdrant database files for the live demo live in the `live_demo/` folder at the repo root. See `live_demo/README.md` for setup. The vector database is built from notebook 04's `EXPORT_DIR`.
