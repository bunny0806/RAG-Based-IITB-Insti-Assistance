Document Management
===================

Overview
--------
This module implements a lightweight Document Management System (DMS) that allows users to upload, index, reindex, inspect, and delete documents via the Streamlit UI. The DMS integrates with the existing ingestion, preprocessing, embedding, and FAISS vectorstore components.

Components
----------
- `documents.metadata_store.MetadataStore` — file-backed JSON store for document metadata.
- `documents.upload_handler.UploadHandler` — validates uploads, computes SHA256 hash, detects duplicates, and saves files to `data/raw/`.
- `documents.indexing_service.IndexingService` — orchestrates chunking, embedding, and safe updates to the FAISS store; supports reindexing and deletion.
- `documents.document_manager.DocumentManager` — high-level orchestrator used by the UI to perform uploads, deletes, reindexing and to retrieve stats.

Metadata schema
---------------
- `document_id`: string unique id for the document
- `filename`: stored filename under `data/raw/`
- `title`: filename without extension
- `upload_time`: ISO timestamp
- `document_type`: e.g., "pdf"
- `file_size`: bytes
- `pages`: number of pages (PDF)
- `chunks`: number of derived chunks
- `status`: uploaded/indexed/failed
- `indexed_at`: ISO timestamp or null
- `last_updated`: ISO timestamp
- `hash`: SHA256 of file bytes

Incremental indexing workflow
-----------------------------
1. Upload saves file under `data/raw/` and writes a metadata record with `status=uploaded`.
2. IndexingService processes only new documents: it loads the file with the ingestion loader, cleans and chunks it, generates embeddings, and appends vectors to existing FAISS index without rebuilding everything.
3. Metadata is updated with chunk counts, `status=indexed`, and `indexed_at` timestamp.

Removing documents
------------------
Deletion removes the chunk metadata from FAISS metadata, rebuilds a new FAISS index from the remaining metadata (embedding chunk texts again), and removes the raw file and metadata entry. This approach avoids partially inconsistent index states and ensures vector/metadata alignment.

Thread safety and progress
--------------------------
- Indexing operations are guarded by a lock in `IndexingService` and `DocumentManager`.
- Only a single indexing task runs concurrently.
- Uploads trigger background indexing for better UX; single-document reindex is synchronous to ensure determinism for operations like delete.

Index statistics
----------------
`DocumentManager.stats()` exposes:
- `total_documents`
- `total_chunks`
- `average_chunks_per_document`
- `index_size` (vectors)
- `vector_dimension`
- `metadata_count`
- `last_indexed`
- `embedding_model`

Why incremental indexing
------------------------
Incremental indexing avoids re-embedding and re-adding existing vectors, significantly reducing compute cost and allowing rapid ingestion of new documents. It enables near-real-time updates and scales better for large corpora.
