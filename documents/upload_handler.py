"""Streamlit upload handler for document manager."""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from utils.logging_utils import setup_logging
from documents.metadata_store import MetadataStore, DocumentMetadata

logger = setup_logging("upload.log")


class UploadHandler:
    """Handle file validation, duplicate detection, and saving uploaded files."""

    def __init__(self, raw_dir: str | Path, metadata_store: MetadataStore) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_store = metadata_store

    def save_uploaded_file(self, uploaded_file) -> Tuple[DocumentMetadata, bool]:
        """Validate and persist an uploaded file. Returns (metadata, created).

        If duplicate detected, returns existing metadata and created=False.
        """
        filename = Path(uploaded_file.name).name
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        file_hash = self.metadata_store.compute_hash(file_bytes)

        if self.metadata_store.exists_hash(file_hash):
            logger.info("Duplicate upload rejected: %s", filename)
            # find existing doc
            existing = next(doc for doc in self.metadata_store.list_documents() if doc.hash == file_hash)
            return existing, False

        # Save file
        document_id = f"pdf_{int(datetime.utcnow().timestamp())}_{filename}"
        dest_path = self.raw_dir / filename
        with open(dest_path, "wb") as handle:
            handle.write(file_bytes)

        # Minimal metadata; indexing service will fill pages/chunks
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            title=Path(filename).stem,
            upload_time=datetime.utcnow().isoformat(),
            document_type="pdf",
            file_size=dest_path.stat().st_size,
            pages=0,
            chunks=0,
            status="uploaded",
            hash=file_hash,
        )
        self.metadata_store.add(metadata)
        logger.info("Saved uploaded file %s as %s", filename, document_id)
        return metadata, True
