"""Indexing service to process uploaded documents into the vector index."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from documents.metadata_store import DocumentMetadata, MetadataStore
from embeddings.embedding_pipeline import EmbeddingPipeline
from preprocessing.chunk_pipeline import ChunkPipeline
from preprocessing.models import Chunk
from utils.logging_utils import setup_logging
from vectorstore.faiss_store import FAISSStore

logger = setup_logging("indexing.log")


class IndexingService:
    """Handle document indexing lifecycle and safe FAISS updates."""

    def __init__(
        self,
        raw_dir: str | Path,
        index_path: str | Path,
        metadata_path: str | Path,
        vector_store: Optional[FAISSStore] = None,
        metadata_store: Optional[MetadataStore] = None,
        chunk_pipeline: Optional[ChunkPipeline] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.store = vector_store or FAISSStore(index_path=self.index_path, metadata_path=self.metadata_path)
        self.metadata_store = metadata_store or MetadataStore(metadata_path=self.raw_dir.parent / "metadata.json")
        self.chunk_pipeline = chunk_pipeline or ChunkPipeline()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self._index_lock = threading.RLock()
        self._load_index()

    def _load_index(self) -> None:
        try:
            self.store.load()
        except FileNotFoundError:
            self.store.index = None
            self.store.metadata = []
            logger.info("No existing FAISS index found. A new one will be created on first indexing.")

    def index_document(self, document_id: str, document_path: str) -> DocumentMetadata:
        with self._index_lock:
            logger.info("Starting indexing for document %s", document_id)
            document = self._load_document_for_index(document_path)
            chunks = self.chunk_pipeline.process_documents([document])
            embeddings, chunk_metadata = self.embedding_pipeline.embed_chunks(chunks)
            if self.store.index is None or self.store.index.ntotal == 0:
                self.store.create_index(embeddings.shape[1])
            # add embeddings with persistent IDs
            allocated_ids = self.store.add_embeddings(embeddings, chunk_metadata)
            self.store.save()

            metadata = self.metadata_store.get(document_id)
            if metadata is None:
                metadata = DocumentMetadata(
                    document_id=document_id,
                    filename=Path(document_path).name,
                    title=Path(document_path).stem,
                    upload_time=datetime.utcnow().isoformat(),
                    document_type="pdf",
                    file_size=Path(document_path).stat().st_size,
                    pages=self._get_document_pages(document),
                    chunks=len(chunks),
                    status="indexed",
                    indexed_at=datetime.utcnow().isoformat(),
                    last_updated=datetime.utcnow().isoformat(),
                    hash=metadata.hash if metadata else "",
                )
                self.metadata_store.add(metadata)
            else:
                self.metadata_store.update(
                    document_id,
                    chunks=len(chunks),
                    status="indexed",
                    indexed_at=datetime.utcnow().isoformat(),
                )
            logger.info("Completed indexing for document %s", document_id)
            return metadata

    def reindex_document(self, document_id: str, document_path: str) -> DocumentMetadata:
        with self._index_lock:
            logger.info("Reindexing document %s", document_id)
            current_metadata = self.metadata_store.get(document_id)
            if current_metadata is None:
                raise KeyError(f"Document not found: {document_id}")
            # remove only vectors for this document
            ids = self.store.get_vector_ids_for_document(document_id)
            if ids:
                self.store.remove_ids(ids)
            metadata = self.index_document(document_id=document_id, document_path=document_path)
            return metadata

    def reindex_all(self) -> List[DocumentMetadata]:
        with self._index_lock:
            logger.info("Reindexing all documents.")
            documents = self.metadata_store.list_documents()
            self._remove_all_vectors()
            aggregated_chunks: List[Chunk] = []
            for metadata in documents:
                document_path = str(self.raw_dir / metadata.filename)
                if not Path(document_path).exists():
                    logger.warning("Skipping missing document file during reindex: %s", document_path)
                    continue
                document = self._load_document_for_index(document_path)
                aggregated_chunks.extend(self.chunk_pipeline.process_documents([document]))

            if aggregated_chunks:
                embeddings, chunk_metadata = self.embedding_pipeline.embed_chunks(aggregated_chunks)
                self.store.create_index(embeddings.shape[1])
                self.store.add_embeddings(embeddings, chunk_metadata)
                self.store.save()

            for metadata in documents:
                self.metadata_store.update(
                    metadata.document_id,
                    status="indexed",
                    indexed_at=datetime.utcnow().isoformat(),
                )
            logger.info("Completed full reindex of %s documents.", len(documents))
            return self.metadata_store.list_documents()

    def delete_document(self, document_id: str) -> None:
        with self._index_lock:
            logger.info("Deleting document %s", document_id)
            metadata = self.metadata_store.get(document_id)
            if metadata is None:
                raise KeyError(f"Document not found: {document_id}")
            ids = self.store.get_vector_ids_for_document(document_id)
            if ids:
                self.store.remove_ids(ids)
            self.metadata_store.remove(document_id)
            file_path = self.raw_dir / metadata.filename
            if file_path.exists():
                file_path.unlink()
                logger.info("Deleted raw file %s", file_path)

    def get_statistics(self) -> Dict[str, Any]:
        self.store.load()
        stats = self.store.stats() if self.store.index is not None else {"ntotal": 0, "dimension": 0, "metadata_count": 0}
        metadata_stats = self.metadata_store.stats()
        return {
            **metadata_stats,
            "index_size": stats["ntotal"],
            "vector_dimension": stats["dimension"],
            "metadata_count": stats["metadata_count"],
            "last_indexed": max(
                (doc.indexed_at for doc in self.metadata_store.list_documents() if doc.indexed_at),
                default=None,
            ),
            "embedding_model": self.embedding_pipeline.embedder.model_name,
        }

    def _remove_document_vectors(self, document_id: str) -> None:
        # Deprecated: old method of rebuilding index. Kept for compatibility.
        ids = self.store.get_vector_ids_for_document(document_id)
        if not ids:
            logger.info("No vectors to remove for document %s", document_id)
            return
        self.store.remove_ids(ids)

    def _remove_all_vectors(self) -> None:
        self.store.index = None
        self.store.metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def _load_document_for_index(self, document_path: str):
        from ingestion.pdf_loader import PDFLoader

        loader = PDFLoader(document_path)
        document = loader.load()
        loader.validate(document)
        return document

    def _get_document_pages(self, document: Any) -> int:
        return int(document.metadata.get("total_pages", 0))
