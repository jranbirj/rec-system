"""Gradio demo for the two-tower restaurant recommender.

Loads precomputed user + item embeddings (from notebook 04's qdrant_export/)
and a local Qdrant collection (built by build_qdrant.py). Visitors pick a
user, optionally filter by city / max price, and see top-K recommendations
ranked by the trained two-tower model.

For local run:
    python app.py
For HF Spaces: app_file = app.py in the Space's README.md frontmatter.
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

EXPORT_DIR  = Path(os.environ.get("EXPORT_DIR",  "./qdrant_export"))
QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", "./qdrant_data"))
COLLECTION  = os.environ.get("QDRANT_COLLECTION", "businesses")
N_USERS_IN_PICKER = 200
RERANK_POOL_FACTOR = 4   # fetch this many × top_k from Qdrant before score perturbation
RERANK_NOISE_STD   = 0.02  # std of Gaussian noise added to scores


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. {hint}"
        )


# Fail loudly at startup if the deployment is incomplete.
_require(EXPORT_DIR / "user_embeddings.npy",
         "Re-run notebook 04 to produce qdrant_export/.")
_require(EXPORT_DIR / "user_embedding_index.parquet",
         "Re-run notebook 04 to produce qdrant_export/.")
_require(EXPORT_DIR / "businesses_meta.parquet",
         "Run build_qdrant.py to enrich payloads with name / city.")
_require(EXPORT_DIR / "user_history.parquet",
         "Run build_qdrant.py to produce user_history.parquet.")
_require(QDRANT_PATH,
         "Run build_qdrant.py to populate the local Qdrant collection.")

print(f"Loading user embeddings from {EXPORT_DIR}...")
user_embs  = np.load(EXPORT_DIR / "user_embeddings.npy")
user_index = pd.read_parquet(EXPORT_DIR / "user_embedding_index.parquet")
user_id_to_idx = {uid: i for i, uid in enumerate(user_index["user_id"])}

print(f"Loading business + user-history metadata...")
biz_meta     = pd.read_parquet(EXPORT_DIR / "businesses_meta.parquet")
user_history = pd.read_parquet(EXPORT_DIR / "user_history.parquet")

# Picker shows the users with the richest training history first — visitors
# get a recognizable "this user clearly likes Italian" prior to the recs.
# Build (label, value) tuples so the dropdown shows "Linus L. (123 reviews)"
# while the underlying value remains the user_id used for lookups.
_uniq = (
    user_history.drop_duplicates(subset="user_id", keep="first")
                .head(N_USERS_IN_PICKER)
)
def _label(row):
    name = (row.get("display_name") or "").strip() or "(anon)"
    n    = int(row.get("review_count") or 0)
    suffix = row["user_id"][:6]   # tail-distinguish identical first names
    return f"{name} — {n} reviews ({suffix})"
sample_user_choices = [(_label(r), r["user_id"]) for _, r in _uniq.iterrows()]

# City list capped at the most common 50; drop blanks so we don't show "" in the picker.
nonblank_cities = biz_meta["city"].dropna()
nonblank_cities = nonblank_cities[nonblank_cities.astype(str).str.strip() != ""]
top_cities = nonblank_cities.value_counts().head(50).index.tolist()
CITIES = ["(any)"] + top_cities

print(f"Connecting to Qdrant at {QDRANT_PATH}...")
qdrant = QdrantClient(path=str(QDRANT_PATH))
n_pts = qdrant.count(COLLECTION).count
print(f"  collection '{COLLECTION}': {n_pts:,} points\n")


def recommend(user_id: str, city: str, max_price: int, top_k: int):
    if not user_id or user_id not in user_id_to_idx:
        return "User not found.", pd.DataFrame()

    vec = user_embs[user_id_to_idx[user_id]].tolist()

    must = []
    if city and city != "(any)":
        must.append(FieldCondition(key="city",  match=MatchValue(value=city)))
    if max_price and max_price < 4:
        must.append(FieldCondition(key="price", range=Range(lte=max_price)))
    qfilter = Filter(must=must) if must else None

    pool_size = int(top_k) * RERANK_POOL_FACTOR
    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        query_filter=qfilter,
        limit=pool_size,
        with_payload=True,
    )
    hits = result.points
    scores = np.array([h.score for h in hits], dtype=np.float32)
    scores += np.random.normal(0, RERANK_NOISE_STD, size=len(scores))
    hits = [hits[i] for i in np.argsort(scores)[::-1][:int(top_k)]]

    rows = []
    for h in hits:
        p = h.payload or {}
        rows.append({
            "Score":      round(float(h.score), 3),
            "Name":       p.get("name", p.get("business_id", "")),
            "City":       p.get("city", ""),
            "Stars":      p.get("stars", ""),
            "Price":      p.get("price", ""),
            "Categories": (p.get("categories") or "")[:80],
        })

    history = user_history[user_history["user_id"] == user_id].head(5)
    if history.empty:
        history_md = "_(no training history available for this user)_"
    else:
        history_md = "**Recent positive reviews from this user:**\n\n"
        for _, r in history.iterrows():
            history_md += f"- *{r['name']}* ({r['city']}) — {r['stars']} stars\n"

    return history_md, pd.DataFrame(rows)


with gr.Blocks(title="Two-Tower Restaurant Recs") as demo:
    gr.Markdown("# Two-Tower Restaurant Recommendations")
    gr.Markdown(
        "Personalized restaurant recommendations from a two-tower neural net "
        "trained on Yelp reviews. Pick a user, optionally filter by city or "
        "max price, hit Recommend."
    )

    with gr.Row():
        with gr.Column(scale=1):
            user_picker = gr.Dropdown(
                choices=sample_user_choices,
                value=sample_user_choices[0][1] if sample_user_choices else None,
                label=f"User (top {len(sample_user_choices)} by training history)",
            )
            city_picker  = gr.Dropdown(choices=CITIES, value="(any)", label="City filter")
            price_slider = gr.Slider(1, 4, value=4, step=1, label="Max price ($1 to $4)")
            topk_slider  = gr.Slider(1, 50, value=10, step=1, label="Top K")
            go_btn       = gr.Button("Recommend", variant="primary")

        with gr.Column(scale=2):
            history_box  = gr.Markdown()
            results_tbl  = gr.Dataframe(
                headers=["Score", "Name", "City", "Stars", "Price", "Categories"],
                label="Recommendations",
            )

    go_btn.click(
        recommend,
        inputs=[user_picker, city_picker, price_slider, topk_slider],
        outputs=[history_box, results_tbl],
    )


if __name__ == "__main__":
    demo.launch()
