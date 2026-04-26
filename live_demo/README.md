---
title: Two-Tower Restaurant Recommendations
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Two-Tower Restaurant Recommendations — Live Demo

Gradio frontend for the two-tower model trained in `notebooks/04_two_towers_baseline.ipynb`.
A visitor picks a user from the dropdown, optionally filters by city / max
price, and sees the top-K personalized restaurant recommendations. ANN search
runs against a local Qdrant collection of trained business embeddings.

## Layout

```
live_demo/
├── app.py             # Gradio app — what HF Spaces serves
├── build_qdrant.py    # one-time local script: builds qdrant_data/ + display metadata
├── requirements.txt
└── README.md          # this file (also the HF Space metadata header)
```

## Local setup

1. Run `notebooks/04_two_towers_baseline.ipynb` to produce `qdrant_export/`
   on Drive. Copy that directory next to `live_demo/`:
   ```
   rec-system/
   ├── live_demo/
   └── qdrant_export/
       ├── business_embeddings.npy
       ├── business_embedding_index.parquet
       ├── user_embeddings.npy
       ├── user_embedding_index.parquet
       ├── two_tower_best.pt
       ├── user_scaler.joblib
       ├── business_scaler.joblib
       ├── business_medians.json
       └── model_meta.json
   ```

2. Install deps and build the local Qdrant store + display metadata:
   ```
   pip install -r requirements.txt
   python build_qdrant.py \
     --export-dir   ../qdrant_export \
     --yelp-json    /path/to/yelp_academic_dataset_business.json \
     --train-reviews ../path/to/train_reviews.parquet
   ```
   This adds two files to `qdrant_export/` (`businesses_meta.parquet`,
   `user_history.parquet`) and creates a `qdrant_data/` directory with the
   Qdrant collection.

3. Run the app:
   ```
   python app.py
   ```
   Gradio prints a local URL.

## Deploying to Hugging Face Spaces

1. Create a new Space on huggingface.co/spaces (SDK = Gradio).
2. Clone the Space repo locally.
3. Copy these files into the Space repo:
   - `app.py`, `requirements.txt`, `README.md` (this file)
   - `qdrant_export/` (only the files `app.py` reads at runtime —
     `user_embeddings.npy`, `user_embedding_index.parquet`,
     `businesses_meta.parquet`, `user_history.parquet`)
   - `qdrant_data/` (output of `build_qdrant.py`)
4. Track binaries with git LFS:
   ```
   git lfs track "qdrant_export/*.npy" "qdrant_export/*.parquet" "qdrant_data/**"
   git add .gitattributes
   ```
5. Commit and push. The Space rebuilds automatically.

The model checkpoint, scalers, and `business_embeddings.npy` are NOT
needed at runtime in this v1 — they're already baked into either the user
embedding vectors or the Qdrant collection. Skipping them keeps the Space
small.

## Configuration

Environment variables read by `app.py`:

| Variable             | Default              |
|----------------------|----------------------|
| `EXPORT_DIR`         | `./qdrant_export`    |
| `QDRANT_PATH`        | `./qdrant_data`      |
| `QDRANT_COLLECTION`  | `businesses`         |

## What the demo does NOT support yet

- Synthetic users built from form input (sliders + free-text categories).
  Would need the trained user tower + a Qwen embedding service running in
  the Space. Possible v2.
- Business-side feature filters beyond city + max price. Add more
  `FieldCondition` entries in `app.py::recommend` to expose them.
- Cold-start users (no training history). The dropdown only surfaces users
  with a precomputed embedding.
