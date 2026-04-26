"""Build the local Qdrant collection + display metadata that app.py needs.

Inputs (produced by notebook 04 + optionally the raw Yelp dump):
  - {export_dir}/business_embeddings.npy
  - {export_dir}/business_embedding_index.parquet
  - {yelp_json}                         (optional — yelp_academic_dataset_business.json)
  - {train_reviews}                     (train_reviews.parquet from notebook 02)

Outputs:
  - {qdrant_out}/                       (Qdrant local persistent store)
  - {export_dir}/businesses_meta.parquet  (id -> name/city/stars/price/categories)
  - {export_dir}/user_history.parquet     (top positive interactions per user, with names)

Without --yelp-json the demo still runs — recommendations show business_id
strings instead of restaurant names, and the city filter is disabled.

Example:
    python build_qdrant.py \\
      --export-dir    qdrant_export \\
      --train-reviews ../data/train_reviews.parquet \\
      --yelp-json     ../data/yelp_academic_dataset_business.json   # optional
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def load_yelp_meta(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            attrs = r.get("attributes") or {}
            price = attrs.get("RestaurantsPriceRange2")
            try:
                price = int(price) if price not in (None, "None") else None
            except (TypeError, ValueError):
                price = None
            rows.append({
                "business_id": r["business_id"],
                "name":        r.get("name", ""),
                "city":        r.get("city", ""),
                "state":       r.get("state", ""),
                "stars":       float(r.get("stars", 0.0)),
                "price":       price,
                "categories":  r.get("categories") or "",
            })
    return pd.DataFrame(rows)


def load_yelp_user_meta(path: Path, keep_user_ids: set) -> pd.DataFrame:
    """Stream Yelp user JSON, keeping only users we care about (memory-light)."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("user_id")
            if uid not in keep_user_ids:
                continue
            rows.append({
                "user_id":      uid,
                "display_name": (r.get("name") or "").strip() or "(anon)",
                "review_count": int(r.get("review_count", 0)),
                "avg_stars":    float(r.get("average_stars", 0.0)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir",     required=True, type=Path)
    ap.add_argument("--yelp-json",      default=None,  type=Path,
                    help="optional — without it, name/city/etc payload fields are blank")
    ap.add_argument("--yelp-user-json", default=None,  type=Path,
                    help="optional — yelp_academic_dataset_user.json. Adds display_name + review_count to user_history.")
    ap.add_argument("--train-reviews",  required=True, type=Path)
    ap.add_argument("--qdrant-out",   default=Path("./qdrant_data"), type=Path)
    ap.add_argument("--collection",   default="businesses")
    args = ap.parse_args()

    # 1. Load embeddings + business_id alignment
    print(f"Loading embeddings from {args.export_dir}...")
    item_embs  = np.load(args.export_dir / "business_embeddings.npy")
    item_index = pd.read_parquet(args.export_dir / "business_embedding_index.parquet")
    assert len(item_embs) == len(item_index), "embedding count vs index parquet mismatch"
    dim = item_embs.shape[1]
    print(f"  {len(item_embs):,} business embeddings, dim={dim}")

    # 2. Pull display fields out of the raw Yelp JSON if provided; otherwise
    #    fabricate a stub frame (recs will show business_id strings).
    if args.yelp_json and args.yelp_json.exists():
        print(f"Loading Yelp business metadata from {args.yelp_json}...")
        yelp_meta = load_yelp_meta(args.yelp_json)
        print(f"  {len(yelp_meta):,} businesses in raw JSON")
        df = item_index.merge(yelp_meta, on="business_id", how="left")
        n_missing = df["name"].isna().sum()
        if n_missing:
            print(f"  WARNING: {n_missing:,} embeddings had no metadata match — using business_id as name fallback")
            df["name"] = df["name"].fillna(df["business_id"])
            df["city"] = df["city"].fillna("")
    else:
        print("No --yelp-json provided — payloads will use business_id as the display name.")
        df = item_index.copy()
        df["name"]       = df["business_id"]
        df["city"]       = ""
        df["state"]      = ""
        df["stars"]      = float("nan")
        df["price"]      = pd.NA
        df["categories"] = ""

    # 3. (Re)create Qdrant collection
    args.qdrant_out.mkdir(parents=True, exist_ok=True)
    print(f"Initializing Qdrant local store at {args.qdrant_out}...")
    client = QdrantClient(path=str(args.qdrant_out))

    if args.collection in [c.name for c in client.get_collections().collections]:
        print(f"  Dropping existing collection '{args.collection}'")
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # 4. Upsert in batches
    batch_size = 1000
    print(f"Upserting {len(df):,} points (batch={batch_size})...")
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        points = []
        for offset, (_, r) in enumerate(batch.iterrows()):
            row_idx = start + offset
            points.append(PointStruct(
                id=row_idx,
                vector=item_embs[row_idx].tolist(),
                payload={
                    "business_id": r["business_id"],
                    "name":        r["name"],
                    "city":        r["city"],
                    "stars":       float(r["stars"]) if pd.notna(r["stars"]) else None,
                    "price":       int(r["price"])   if pd.notna(r["price"]) else None,
                    "categories":  (r["categories"] or "")[:200],
                },
            ))
        client.upsert(args.collection, points)

    print(f"  Done. Collection size: {client.count(args.collection).count:,}")

    # 5. Save businesses_meta.parquet for the app's display lookups
    meta_out = args.export_dir / "businesses_meta.parquet"
    df.to_parquet(meta_out, index=False)
    print(f"Saved businesses meta -> {meta_out}")

    # 6. Build user_history.parquet — top 10 positives per user, with display names
    print("Building user_history.parquet from train positives...")
    train_reviews = pd.read_parquet(args.train_reviews)
    pos = train_reviews[train_reviews["label"] == 1].copy()
    pos = pos.sort_values(["user_id", "stars"], ascending=[True, False])
    pos = pos.groupby("user_id").head(10)
    pos = pos.merge(df[["business_id", "name", "city"]], on="business_id", how="left")

    # Order users by their training-positive count so the demo dropdown
    # surfaces users with rich history first.
    counts = train_reviews[train_reviews["label"] == 1].groupby("user_id").size().sort_values(ascending=False)
    order  = {uid: i for i, uid in enumerate(counts.index)}
    pos["_order"] = pos["user_id"].map(order)
    pos = pos.sort_values("_order").drop(columns="_order")

    # Optional: enrich with display_name + review_count from the user JSON,
    # so the demo dropdown shows "John D. (123 reviews)" instead of a hash.
    if args.yelp_user_json and args.yelp_user_json.exists():
        print(f"Loading Yelp user metadata from {args.yelp_user_json} (filtered)...")
        keep_uids = set(pos["user_id"].unique())
        user_meta = load_yelp_user_meta(args.yelp_user_json, keep_uids)
        print(f"  matched {len(user_meta):,} of {len(keep_uids):,} users")
        pos = pos.merge(user_meta, on="user_id", how="left")
        pos["display_name"] = pos["display_name"].fillna("(anon)")
        pos["review_count"] = pos["review_count"].fillna(0).astype(int)
    else:
        pos["display_name"] = "(anon)"
        pos["review_count"] = 0

    history_out = args.export_dir / "user_history.parquet"
    pos[["user_id", "display_name", "review_count", "business_id", "name", "city", "stars", "date"]].to_parquet(history_out, index=False)
    print(f"Saved user history -> {history_out}")

    print("\nNext: cd live_demo && python app.py")


if __name__ == "__main__":
    main()
