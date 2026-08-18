"""Tests for the hybrid retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.models import Chunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import Retriever
from retrieval.rank_fusion import reciprocal_rank_fusion
from vectorstore.faiss_store import FAISSStore


def test_bm25_retriever_build_and_search(tmp_path: Path) -> None:
    chunks = [
        Chunk(chunk_id="1", text="IIT Bombay campus life is vibrant.", metadata={"document_name": "Campus Guide"}),
        Chunk(chunk_id="2", text="Admissions information is available online.", metadata={"document_name": "Admissions"}),
    ]

    index_path = tmp_path / "bm25_index.json"
    retriever = BM25Retriever(index_path=index_path, top_k=2)
    retriever.build(chunks)
    results = retriever.search("campus life")

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "1"
    assert results[0].retrieval_method == "bm25"

    retriever_loaded = BM25Retriever(index_path=index_path, top_k=2)
    retriever_loaded.load()
    loaded_results = retriever_loaded.search("admissions")

    assert len(loaded_results) == 2
    assert loaded_results[0].chunk.chunk_id == "2"


def test_reciprocal_rank_fusion_merges_results() -> None:
    chunk_a = Chunk(chunk_id="a", text="A text chunk.", metadata={})
    chunk_b = Chunk(chunk_id="b", text="B text chunk.", metadata={})

    result_a1 = Retriever.__new__(Retriever)  # type: ignore[attr-defined]
    # Not used directly; test only fusion logic with dummy RetrievalResult objects.
    from retrieval.models import RetrievalResult

    r1 = RetrievalResult(chunk=chunk_a, score=1.0, rank=1, retrieval_method="dense", metadata={})
    r2 = RetrievalResult(chunk=chunk_b, score=0.8, rank=2, retrieval_method="dense", metadata={})
    r3 = RetrievalResult(chunk=chunk_b, score=1.2, rank=1, retrieval_method="bm25", metadata={})

    fused = reciprocal_rank_fusion([[r1, r2], [r3]], k=60, top_k=2)

    assert len(fused) == 2
    assert fused[0].chunk.chunk_id == "b"
    assert fused[1].chunk.chunk_id == "a"


def test_hybrid_retriever_fuses_dense_and_sparse(tmp_path: Path) -> None:
    chunks = [
        Chunk(chunk_id="1", text="IIT Bombay student hostel details.", metadata={"document_name": "Hostel Guide"}),
        Chunk(chunk_id="2", text="The academic calendar is published every year.", metadata={"document_name": "Academic Calendar"}),
    ]

    faiss_store = FAISSStore(index_path=tmp_path / "faiss.index", metadata_path=tmp_path / "faiss.json")
    import faiss

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    faiss_store.create_index(2)
    faiss_store.add_embeddings(embeddings, [
        {"chunk_id": "1", "text": chunks[0].text},
        {"chunk_id": "2", "text": chunks[1].text},
    ])

    dense_retriever = Retriever(vector_store=faiss_store, top_k=2)
    bm25_index_path = tmp_path / "bm25_index.json"
    bm25_retriever = BM25Retriever(index_path=bm25_index_path, top_k=2)
    bm25_retriever.build(chunks)

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=2,
        bm25_top_k=2,
        final_top_k=2,
        rrf_k=60,
    )

    query_embedding = np.array([1.0, 0.0], dtype=np.float32)
    results = hybrid.search("hostel details", query_embedding)

    assert len(results) == 2
    assert results[0].chunk.chunk_id in {"1", "2"}
