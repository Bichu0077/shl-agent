"""
retriever.py - Semantic retrieval over the SHL catalog FAISS index.
Loaded once at app startup and reused across all requests.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional
import numpy as np


class CatalogRetriever:
    """
    Wraps the FAISS index and catalog metadata.
    Provides:
      - search(query, k) -> list of catalog items
      - get_by_name(name) -> single catalog item or None
      - full_catalog_text() -> compact text of entire catalog for LLM context
    """

    def __init__(self, index_path="data/index.faiss", meta_path="data/index_meta.json"):
        self._index = None
        self._model = None
        self._catalog: list[dict] = []
        self._texts: list[str] = []
        self._index_path = index_path
        self._meta_path = meta_path
        self._loaded = False

    def load(self):
        """Load index and model. Called once at startup."""
        if self._loaded:
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            os.system("pip install sentence-transformers faiss-cpu --quiet")
            import faiss
            from sentence_transformers import SentenceTransformer

        if not os.path.exists(self._index_path) or not os.path.exists(self._meta_path):
            raise FileNotFoundError(
                "FAISS index not found. Run: python scripts/build_index.py"
            )

        self._index = faiss.read_index(self._index_path)
        with open(self._meta_path) as f:
            meta = json.load(f)
        self._catalog = meta["catalog"]
        self._texts = meta["texts"]

        print(f"[Retriever] Loaded {len(self._catalog)} products, {self._index.ntotal} vectors")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Retriever] Embedding model loaded.")
        self._loaded = True

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k catalog items for a query string."""
        if not self._loaded:
            self.load()

        vec = self._model.encode([query], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")

        actual_k = min(k, len(self._catalog))
        scores, indices = self._index.search(vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self._catalog[idx])
            item["_score"] = float(score)
            results.append(item)
        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Fuzzy name lookup - returns best match from catalog."""
        if not self._loaded:
            self.load()

        name_lower = name.lower().strip()
        # Exact match first
        for item in self._catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self._catalog:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                return item
        # Search-based fallback
        results = self.search(name, k=1)
        return results[0] if results else None

    def get_all(self) -> list[dict]:
        if not self._loaded:
            self.load()
        return self._catalog

    def full_catalog_summary(self) -> str:
        """
        Compact catalog text for injection into LLM context.
        Each item on one line to save tokens.
        """
        if not self._loaded:
            self.load()

        lines = ["SHL Individual Test Solutions Catalog:\n"]
        for p in self._catalog:
            types = ", ".join(p.get("test_types", [])) or "?"
            levels = ", ".join(p.get("job_levels", [])) or "All levels"
            lines.append(
                f"• {p['name']} | Types: {types} | Levels: {levels} | {p.get('duration','?')} | URL: {p['url']}"
            )
        return "\n".join(lines)

    def catalog_for_comparison(self, names: list[str]) -> str:
        """Retrieve rich descriptions for a list of assessment names (for compare queries)."""
        if not self._loaded:
            self.load()

        parts = []
        for name in names:
            item = self.get_by_name(name)
            if item:
                parts.append(
                    f"### {item['name']}\n"
                    f"URL: {item['url']}\n"
                    f"Test Types: {', '.join(item.get('test_types', []))}\n"
                    f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
                    f"Duration: {item.get('duration', 'N/A')}\n"
                    f"Description: {item.get('description', '')}\n"
                    f"Details: {item.get('raw_text', '')[:600]}\n"
                )
        return "\n".join(parts) if parts else "No matching assessments found in catalog."


# Singleton instance
retriever = CatalogRetriever()
