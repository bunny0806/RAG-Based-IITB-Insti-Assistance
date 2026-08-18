"""Metadata store for uploaded and indexed documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from utils.logging_utils import setup_logging

logger = setup_logging("documents.log")


@dataclass(slots=True)
class DocumentMetadata:
    document_id: str
    filename: str
    title: str
    upload_time: str
    document_type: str
    file_size: int
    pages: int
    chunks: int
    status: str
    indexed_at: Optional[str] = None
    last_updated: Optional[str] = None
    hash: str = ""


class MetadataStore:
    """Persist metadata for uploaded documents and keep it synchronized."""

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._documents: Dict[str, DocumentMetadata] = {}
        self._load()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            self._documents = {}
            return

        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._documents = {
                item["document_id"]: DocumentMetadata(**item) for item in data
            }
            logger.info("Loaded %s document metadata records.", len(self._documents))
        except Exception as exc:
            logger.error("Failed to load metadata store: %s", exc)
            self._documents = {}

    def _save(self) -> None:
        try:
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump([asdict(item) for item in self._documents.values()], handle, indent=2)
            logger.info("Saved metadata store with %s records.", len(self._documents))
        except Exception as exc:
            logger.error("Failed to save metadata store: %s", exc)
            raise

    def list_documents(self) -> List[DocumentMetadata]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: str) -> Optional[DocumentMetadata]:
        with self._lock:
            return self._documents.get(document_id)

    def exists_hash(self, file_hash: str) -> bool:
        with self._lock:
            return any(doc.hash == file_hash for doc in self._documents.values())

    def add(self, metadata: DocumentMetadata) -> None:
        with self._lock:
            metadata.last_updated = datetime.utcnow().isoformat()
            self._documents[metadata.document_id] = metadata
            self._save()
            logger.info("Added document metadata: %s", metadata.document_id)

    def update(self, document_id: str, **fields: Any) -> None:
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                raise KeyError(f"Document not found: {document_id}")
            for key, value in fields.items():
                if hasattr(document, key):
                    setattr(document, key, value)
            document.last_updated = datetime.utcnow().isoformat()
            self._save()
            logger.info("Updated document metadata: %s", document_id)

    def remove(self, document_id: str) -> None:
        with self._lock:
            if document_id in self._documents:
                del self._documents[document_id]
                self._save()
                logger.info("Removed metadata for document: %s", document_id)
            else:
                raise KeyError(f"Document not found: {document_id}")

    @staticmethod
    def compute_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_docs = len(self._documents)
            total_chunks = sum(doc.chunks for doc in self._documents.values())
            avg_chunks = total_chunks / total_docs if total_docs else 0
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "average_chunks_per_document": avg_chunks,
            }
