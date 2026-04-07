"""
retriever.py
------------
Hybrid RAG retrieval module for DRBot.

Combines:
  - Dense retrieval  : FAISS + BioLORD-2023 (SentenceTransformer)
  - Sparse retrieval : BM25 (rank_bm25)
  - Fusion           : Reciprocal Rank Fusion (RRF)
  - Re-ranking       : CrossEncoder (ms-marco-MiniLM-L-6-v2)
"""

import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# 1. Load components once at import time
# ---------------------------------------------------------------------------
print("Loading retriever components...")

index = faiss.read_index("data/dr_faiss.index")

with open("data/dr_chunks.json") as f:
    _data = json.load(f)

chunks   = _data["chunks"]
metadata = _data["metadata"]

# BioLORD-2023 for biomedical dense embeddings
retriever_encoder = SentenceTransformer("FremyCompany/BioLORD-2023", device="cpu")
tokenized         = [c.lower().split() for c in chunks]
bm25              = BM25Okapi(tokenized)

# CrossEncoder reranker (~90 MB, fits easily in 4 GB VRAM)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

print("✅ Retriever components loaded.")


# ---------------------------------------------------------------------------
# 2. Dense retrieval (FAISS)
# ---------------------------------------------------------------------------
def dense_retrieve(query: str, k: int = 20) -> list[tuple[int, float]]:
    """Return top-k (index, score) pairs from FAISS dense search."""
    q_emb = retriever_encoder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


# ---------------------------------------------------------------------------
# 3. Sparse retrieval (BM25)
# ---------------------------------------------------------------------------
def bm25_retrieve(query: str, k: int = 20) -> list[tuple[int, float]]:
    """Return top-k (index, score) pairs from BM25 sparse search."""
    scores = bm25.get_scores(query.lower().split())
    top_k  = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_k]


# ---------------------------------------------------------------------------
# 4. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense_res: list[tuple[int, float]],
    bm25_res: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse dense + sparse rankings with RRF."""
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_res):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_res):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 5. Hybrid retrieval (dense + sparse + RRF)
# ---------------------------------------------------------------------------
def hybrid_retrieve(query: str, top_k: int = 15) -> list[tuple[str, dict, float]]:
    """
    Retrieve top-k relevant chunks using hybrid search.

    Returns list of (chunk_text, metadata_dict, rrf_score).
    """
    fused = reciprocal_rank_fusion(dense_retrieve(query), bm25_retrieve(query))
    return [(chunks[i], metadata[i], score) for i, score in fused[:top_k]]


# ---------------------------------------------------------------------------
# 6. CrossEncoder re-ranking
# ---------------------------------------------------------------------------
def rerank(
    query: str,
    candidates: list[tuple[str, dict, float]],
    top_k: int = 3,
) -> list[tuple[str, dict, float]]:
    """
    Re-rank hybrid candidates with a CrossEncoder for higher precision.

    Returns top-k re-ranked (chunk_text, metadata_dict, ce_score) tuples.
    """
    pairs  = [(query, chunk) for chunk, _, _ in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "What are cotton wool spots in diabetic retinopathy?"
    candidates = hybrid_retrieve(query, top_k=15)
    top3 = rerank(query, candidates, top_k=3)

    print(f"Query: {query}\n")
    for i, (chunk, meta, score) in enumerate(top3, 1):
        print(f"--- Result {i} (score={score:.4f}) ---")
        print(chunk[:400])
        print()
