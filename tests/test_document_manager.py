"""Tests for the Document Manager and indexing workflows."""

from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pytest

from documents.metadata_store import MetadataStore, DocumentMetadata
from documents.upload_handler import UploadHandler
from documents.document_manager import DocumentManager
from embeddings.embedding_pipeline import EmbeddingPipeline
from ingestion.base_loader import Document


class FakeEmbedder:
    """Offline deterministic embedder with the MiniLM-compatible FAISS dimension."""

    embedding_dimension = 384
    model_name = "fake-embedder"

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided to embedding fake.")
        vectors = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for index, byte in enumerate(text.encode("utf-8")):
                vectors[row, index % self.embedding_dimension] += (byte % 31) / 31.0
        return vectors

    def encode_many_with_progress(self, texts: list[str]) -> np.ndarray:
        return self.encode_many(texts)


@pytest.fixture
def offline_indexing_dependencies(monkeypatch):
    """Replace model-backed embedding and PDF loading with deterministic test doubles."""
    def fake_embedding_pipeline() -> EmbeddingPipeline:
        return EmbeddingPipeline(embedder=FakeEmbedder())

    def fake_load_document(self, document_path: str) -> Document:
        return Document(
            content="This is deterministic offline content for indexing.",
            metadata={"total_pages": 1, "document_name": Path(document_path).name},
            source=Path(document_path).name,
            document_type="pdf",
        )

    monkeypatch.setattr("documents.indexing_service.EmbeddingPipeline", fake_embedding_pipeline)
    monkeypatch.setattr("documents.indexing_service.IndexingService._load_document_for_index", fake_load_document)


def test_upload_and_duplicate_detection(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    metadata_path = tmp_path / "metadata.json"
    store = MetadataStore(metadata_path)
    handler = UploadHandler(raw_dir, store)

    file_content = b"PDF data page1"
    uploaded = io.BytesIO(file_content)
    uploaded.name = "test.pdf"

    metadata, created = handler.save_uploaded_file(uploaded)
    assert created is True
    assert metadata.filename == "test.pdf"
    assert store.exists_hash(metadata.hash)

    # duplicate
    uploaded2 = io.BytesIO(file_content)
    uploaded2.name = "copy.pdf"
    existing, created2 = handler.save_uploaded_file(uploaded2)
    assert created2 is False
    assert existing.document_id == metadata.document_id


def test_document_manager_indexing_and_deletion(tmp_path: Path, offline_indexing_dependencies) -> None:
    raw_dir = tmp_path / "raw"
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "faiss.json"

    # create fake pdf file
    raw_dir.mkdir(parents=True, exist_ok=True)
    test_file = raw_dir / "doc.pdf"
    test_file.write_bytes(b"This is a test pdf content for indexing.")

    manager = DocumentManager(raw_dir=raw_dir, index_path=index_path, metadata_path=metadata_path)

    # simulate upload entry
    meta = DocumentMetadata(document_id="pdf_doc_pdf", filename="doc.pdf", title="doc", upload_time="now", document_type="pdf", file_size=123, pages=1, chunks=0, status="uploaded", hash="abc123")
    manager.metadata_store.add(meta)

    # Index, then reindex synchronously using only the offline test doubles.
    manager.indexing_service.index_document(meta.document_id, str(test_file))
    initial_ids = manager.indexing_service.store.get_vector_ids_for_document(meta.document_id)
    assert initial_ids
    assert manager.indexing_service.store.index is not None
    assert manager.indexing_service.store.index.ntotal == len(initial_ids)

    manager.reindex(document_id=meta.document_id)
    reindexed_ids = manager.indexing_service.store.get_vector_ids_for_document(meta.document_id)
    assert reindexed_ids
    assert reindexed_ids != initial_ids
    assert manager.metadata_store.get(meta.document_id).status == "indexed"

    # delete
    manager.delete(meta.document_id)
    assert manager.metadata_store.get(meta.document_id) is None
    assert manager.indexing_service.store.get_vector_ids_for_document(meta.document_id) == []
    assert manager.indexing_service.store.index.ntotal == 0
