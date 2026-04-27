"""Gradio demo for the two-tower restaurant recommender.

Two modes:
  1. Existing User — pick a user from a dropdown, get recommendations from
     their precomputed embedding.
  2. Create a User — enter cuisine preferences as free text + optional city
     filter. Qwen encodes the text at query time; the user tower produces a
     cold-start embedding; Qdrant returns top-K matches.

For local run:
    python app.py
For HF Spaces: app_file = app.py in the Space's README.md frontmatter.
"""
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

EXPORT_DIR  = Path(os.environ.get("EXPORT_DIR",  "./qdrant_export"))
QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", "./qdrant_data"))
COLLECTION  = os.environ.get("QDRANT_COLLECTION", "businesses")
N_USERS_IN_PICKER  = 200
RERANK_POOL_FACTOR = 4
RERANK_NOISE_STD   = 0.02
QWEN_MODEL         = "Qwen/Qwen3-Embedding-0.6B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. {hint}")


_require(EXPORT_DIR / "user_embeddings.npy",         "Re-run notebook 04.")
_require(EXPORT_DIR / "user_embedding_index.parquet", "Re-run notebook 04.")
_require(EXPORT_DIR / "businesses_meta.parquet",      "Run build_qdrant.py.")
_require(EXPORT_DIR / "user_history.parquet",         "Run build_qdrant.py.")
_require(EXPORT_DIR / "model_meta.json",              "Re-run notebook 04.")
_require(EXPORT_DIR / "two_tower_best.pt",            "Re-run notebook 04.")
_require(QDRANT_PATH,                                 "Run build_qdrant.py.")


# ── Precomputed user embeddings ───────────────────────────────────────────────
print("Loading user embeddings...")
user_embs  = np.load(EXPORT_DIR / "user_embeddings.npy")
user_index = pd.read_parquet(EXPORT_DIR / "user_embedding_index.parquet")
user_id_to_idx = {uid: i for i, uid in enumerate(user_index["user_id"])}

print("Loading business + user-history metadata...")
biz_meta     = pd.read_parquet(EXPORT_DIR / "businesses_meta.parquet")
user_history = pd.read_parquet(EXPORT_DIR / "user_history.parquet")

_uniq = user_history.drop_duplicates(subset="user_id", keep="first").head(N_USERS_IN_PICKER)
def _label(row):
    name   = (row.get("display_name") or "").strip() or "(anon)"
    n      = int(row.get("review_count") or 0)
    suffix = row["user_id"][:6]
    return f"{name} — {n} reviews ({suffix})"
sample_user_choices = [(_label(r), r["user_id"]) for _, r in _uniq.iterrows()]

nonblank_cities = biz_meta["city"].dropna()
nonblank_cities = nonblank_cities[nonblank_cities.astype(str).str.strip() != ""]
top_cities = nonblank_cities.value_counts().head(50).index.tolist()
CITIES = ["(any)"] + top_cities


# ── User tower model (for cold-start) ────────────────────────────────────────
class Tower(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev, embedding_dim)

    def forward(self, x):
        return self.output(self.hidden(self.input_norm(x)))


print("Loading model meta + user tower weights...")
with open(EXPORT_DIR / "model_meta.json") as f:
    meta = json.load(f)

user_input_dim = meta["user_input_dim"]   # 1031
embedding_dim  = meta["embedding_dim"]    # 128
hidden_dims    = tuple(meta["hidden_dims"])
dropout        = meta["dropout"]
qwen_dim       = meta["qwen_dim"]         # 1024
n_scalars      = user_input_dim - qwen_dim  # 7

user_tower = Tower(user_input_dim, embedding_dim, hidden_dims, dropout).to(device)
state = torch.load(EXPORT_DIR / "two_tower_best.pt", map_location=device)
# Extract only user tower weights from the full model state dict
user_tower_state = {k.replace("user_tower.", ""): v for k, v in state.items() if k.startswith("user_tower.")}
user_tower.load_state_dict(user_tower_state)
user_tower.eval()
print("User tower loaded.")

print(f"Loading Qwen embedder: {QWEN_MODEL} ...")
embedder = SentenceTransformer(QWEN_MODEL, device=str(device))
print("Embedder loaded.")


# ── Qdrant ────────────────────────────────────────────────────────────────────
print(f"Connecting to Qdrant at {QDRANT_PATH}...")
qdrant = QdrantClient(path=str(QDRANT_PATH))
n_pts  = qdrant.count(COLLECTION).count
print(f"  collection '{COLLECTION}': {n_pts:,} points\n")


# ── Shared retrieval helper ───────────────────────────────────────────────────
def _query_qdrant(vec: list, city: str, top_k: int):
    must = []
    if city and city != "(any)":
        must.append(FieldCondition(key="city", match=MatchValue(value=city)))
    qfilter = Filter(must=must) if must else None

    pool_size = int(top_k) * RERANK_POOL_FACTOR
    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        query_filter=qfilter,
        limit=pool_size,
        with_payload=True,
    )
    hits   = result.points
    scores = np.array([h.score for h in hits], dtype=np.float32)
    scores += np.random.normal(0, RERANK_NOISE_STD, size=len(scores))
    return [hits[i] for i in np.argsort(scores)[::-1][:int(top_k)]]


def _hits_to_df(hits) -> pd.DataFrame:
    rows = []
    for h in hits:
        p = h.payload or {}
        rows.append({
            "Score":      round(float(h.score), 3),
            "Name":       p.get("name", p.get("business_id", "")),
            "City":       p.get("city", ""),
            "Stars":      p.get("stars", ""),
            "Categories": (p.get("categories") or "")[:80],
        })
    return pd.DataFrame(rows)


# ── Tab 1: existing user ──────────────────────────────────────────────────────
def recommend_existing(user_id: str, city: str, top_k: int):
    if not user_id or user_id not in user_id_to_idx:
        return "User not found.", pd.DataFrame()

    vec  = user_embs[user_id_to_idx[user_id]].tolist()
    hits = _query_qdrant(vec, city, top_k)

    history = user_history[user_history["user_id"] == user_id].head(5)
    if history.empty:
        history_md = "_(no training history available for this user)_"
    else:
        history_md = "**Recent positive reviews from this user:**\n\n"
        for _, r in history.iterrows():
            history_md += f"- *{r['name']}* ({r['city']}) — {r['stars']} stars\n"

    return history_md, _hits_to_df(hits)


# ── Tab 2: cold-start user ────────────────────────────────────────────────────
def recommend_cold(cuisine_text: str, city: str, top_k: int):
    if not cuisine_text.strip():
        return "Please describe your food preferences above.", pd.DataFrame()

    qwen_emb = embedder.encode([cuisine_text], normalize_embeddings=True)[0].astype(np.float32)
    user_vec = np.concatenate([np.zeros(n_scalars, dtype=np.float32), qwen_emb])
    user_tensor = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = F.normalize(user_tower(user_tensor), dim=1).cpu().numpy()[0]

    hits = _query_qdrant(emb.tolist(), city, top_k)
    return _hits_to_df(hits)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Two-Tower Restaurant Recs") as demo:
    gr.Markdown("# Two-Tower Restaurant Recommendations")
    gr.Markdown(
        "Personalized restaurant recommendations from a two-tower neural net "
        "trained on Yelp reviews."
    )

    with gr.Tabs():

        with gr.Tab("Existing User"):
            gr.Markdown("Pick a user from the dropdown to get recommendations based on their review history.")
            with gr.Row():
                with gr.Column(scale=1):
                    user_picker  = gr.Dropdown(
                        choices=sample_user_choices,
                        value=sample_user_choices[0][1] if sample_user_choices else None,
                        label=f"User (top {len(sample_user_choices)} by training history)",
                    )
                    city_picker1 = gr.Dropdown(choices=CITIES, value="(any)", label="City filter")
                    topk1        = gr.Slider(1, 50, value=10, step=1, label="Top K")
                    btn1         = gr.Button("Recommend", variant="primary")
                with gr.Column(scale=2):
                    history_box = gr.Markdown()
                    results1    = gr.Dataframe(
                        headers=["Score", "Name", "City", "Stars", "Categories"],
                        label="Recommendations",
                    )
            btn1.click(recommend_existing, inputs=[user_picker, city_picker1, topk1],
                       outputs=[history_box, results1])

        with gr.Tab("Create a User"):
            gr.Markdown("Describe your food preferences and get personalized recommendations — no account needed.")
            with gr.Row():
                with gr.Column(scale=1):
                    cuisine_input = gr.Textbox(
                        label="What kind of food are you looking for?",
                        placeholder="e.g. spicy Thai noodles, late night ramen, casual brunch...",
                        lines=3,
                    )
                    city_picker2  = gr.Dropdown(choices=CITIES, value="(any)", label="City filter")
                    topk2         = gr.Slider(1, 50, value=10, step=1, label="Top K")
                    btn2          = gr.Button("Find Restaurants", variant="primary")
                with gr.Column(scale=2):
                    results2 = gr.Dataframe(
                        headers=["Score", "Name", "City", "Stars", "Categories"],
                        label="Recommendations",
                    )
            btn2.click(recommend_cold, inputs=[cuisine_input, city_picker2, topk2],
                       outputs=results2)


if __name__ == "__main__":
    demo.launch()
