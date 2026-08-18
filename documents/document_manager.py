"""High-level Document Manager that ties upload, indexing, deletion, and stats together."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from documents.indexing_service import IndexingService
from documents.metadata_store import DocumentMetadata, MetadataStore
from documents.upload_handler import UploadHandler
from utils.logging_utils import setup_logging

logger = setup_logging("document_manager.log")


class DocumentManager:
    """Orchestrate document lifecycle for the app and UI integration."""

    def __init__(
        self,
        raw_dir: str | Path,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.metadata_store = MetadataStore(metadata_path)
        self.upload_handler = UploadHandler(raw_dir, self.metadata_store)
        self.indexing_service = IndexingService(raw_dir, index_path, metadata_path, metadata_store=self.metadata_store)
        self._task_lock = threading.Lock()
        self._current_task: Optional[str] = None

    def upload(self, uploaded_file) -> Dict[str, Any]:
        metadata, created = self.upload_handler.save_uploaded_file(uploaded_file)
        if created:
            # kick off background indexing
            self._start_background_indexing(metadata.document_id, self.raw_dir / metadata.filename)
        return {"metadata": metadata, "created": created}

    def delete(self, document_id: str) -> None:
        self._ensure_no_concurrent_tasks()
        self.indexing_service.delete_document(document_id)

    def reindex(self, document_id: Optional[str] = None) -> None:
        self._ensure_no_concurrent_tasks()
        if document_id:
            metadata = self.metadata_store.get(document_id)
            if metadata is None:
                raise KeyError(f"Document not found: {document_id}")
            # Run reindex synchronously for a single document to ensure completion before returning.
            with self._task_lock:
                self._current_task = document_id
            try:
                self.indexing_service.reindex_document(document_id, str(self.raw_dir / metadata.filename))
            finally:
                with self._task_lock:
                    self._current_task = None
        else:
            threading.Thread(target=self.indexing_service.reindex_all, daemon=True).start()

    def stats(self) -> Dict[str, Any]:
        return self.indexing_service.get_statistics()

    def list_documents(self) -> List[DocumentMetadata]:
        return self.metadata_store.list_documents()

    def _start_background_indexing(self, document_id: str, document_path: Path, reindex: bool = False) -> None:
        def task():
            try:
                self._set_current_task(document_id)
                if reindex:
                    self.indexing_service.reindex_document(document_id, str(document_path))
                else:
                    self.indexing_service.index_document(document_id, str(document_path))
            except Exception as exc:
                logger.error("Indexing task failed for %s: %s", document_id, exc)
            finally:
                self._set_current_task(None)

        threading.Thread(target=task, daemon=True).start()

    def _set_current_task(self, task_id: Optional[str]) -> None:
        with self._task_lock:
            self._current_task = task_id

    def _ensure_no_concurrent_tasks(self) -> None:
        with self._task_lock:
            if self._current_task is not None:
                raise RuntimeError(f"Indexing task in progress: {self._current_task}")
