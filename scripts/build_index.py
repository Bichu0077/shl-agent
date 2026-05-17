"""
Build a FAISS vector index from catalog.json using sentence-transformers.
Run once after building the catalog: python scripts/build_index.py
Saves: data/index.faiss, data/index_meta.json
"""

import json
import os
import pickle
import numpy as np

def build_index():
    # Lazy imports so the main app doesn't need these at import time
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("Installing required packages...")
        os.system("pip install sentence-transformers faiss-cpu --quiet")
        from sentence_transformers import SentenceTransformer
        import faiss

    # Load catalog
    catalog_path = "data/catalog.json"
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found at {catalog_path}. Run build_seed_catalog.py first.")

    with open(catalog_path) as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} products from catalog.")

    # Build text chunks for embedding
    # We create rich text for each product to maximize retrieval quality
    def product_to_text(p: dict) -> str:
        parts = [
            f"Name: {p.get('name', '')}",
            f"Description: {p.get('description', '')}",
            f"Test types: {', '.join(p.get('test_types', []))}",
            f"Job levels: {', '.join(p.get('job_levels', []))}",
            f"Duration: {p.get('duration', '')}",
        ]
        raw = p.get("raw_text", "")
        if raw:
            parts.append(f"Details: {raw[:500]}")
        return " | ".join(filter(None, parts))

    texts = [product_to_text(p) for p in catalog]

    # Embed
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding catalog...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (inner product = cosine since normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/index.faiss")
    with open("data/index_meta.json", "w") as f:
        json.dump({"texts": texts, "catalog": catalog}, f)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    print("Saved: data/index.faiss, data/index_meta.json")


if __name__ == "__main__":
    build_index()
