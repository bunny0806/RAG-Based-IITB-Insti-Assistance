import os
import numpy as np
from pathlib import Path

from documents.id_manager import IDManager
from documents.vector_metadata_store import VectorMetadataStore
from vectorstore.faiss_store import FAISSStore


def test_id_manager_allocation_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    mgr = IDManager(path)
    ids1 = mgr.allocate_ids(3)
    assert ids1 == [1, 2, 3]
    # reload
    mgr2 = IDManager(path)
    next_id = mgr2.peek_next()
    assert next_id == 4
    ids2 = mgr2.allocate_ids(2)
    assert ids2 == [4, 5]


def test_faiss_add_remove_and_reload(tmp_path: Path) -> None:
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "vectors.json"

    store = FAISSStore(index_path=index_path, metadata_path=metadata_path)

    # create simple embeddings for two documents
    emb1 = np.random.rand(3, 16).astype("float32")
    meta1 = [{"document_id": "docA", "chunk_index": i, "text": f"a{i}"} for i in range(3)]

    emb2 = np.random.rand(2, 16).astype("float32")
    meta2 = [{"document_id": "docB", "chunk_index": i, "text": f"b{i}"} for i in range(2)]

    ids_a = store.add_embeddings(emb1, meta1)
    ids_b = store.add_embeddings(emb2, meta2)
    store.save()

    stats = store.stats()
    assert stats["ntotal"] == 5
    all_meta = store.vector_metadata.get_all()
    assert len(all_meta) == 5

    # remove document A vectors
    ids_to_remove = store.get_vector_ids_for_document("docA")
    assert set(ids_to_remove) == set(ids_a)
    store.remove_ids(ids_to_remove)

    stats2 = store.stats()
    assert stats2["ntotal"] == 2
    all_meta2 = store.vector_metadata.get_all()
    assert len(all_meta2) == 2

    # reload new store instance and verify persistence
    store2 = FAISSStore(index_path=index_path, metadata_path=metadata_path)
    store2.load()
    stats3 = store2.stats()
    assert stats3["ntotal"] == 2
    assert len(store2.vector_metadata.get_all()) == 2


def test_reindex_document_preserves_other(tmp_path: Path) -> None:
    index_path = tmp_path / "faiss2.index"
    metadata_path = tmp_path / "vectors2.json"

    store = FAISSStore(index_path=index_path, metadata_path=metadata_path)

    emb_a = np.random.rand(2, 12).astype("float32")
    meta_a = [{"document_id": "A", "text": "a"}, {"document_id": "A", "text": "a2"}]
    emb_b = np.random.rand(1, 12).astype("float32")
    meta_b = [{"document_id": "B", "text": "b"}]

    ids_a = store.add_embeddings(emb_a, meta_a)
    ids_b = store.add_embeddings(emb_b, meta_b)
    store.save()

    # reindex A: remove and add new vectors
    ids_remove = store.get_vector_ids_for_document("A")
    store.remove_ids(ids_remove)
    new_emb_a = np.random.rand(3, 12).astype("float32")
    new_meta_a = [{"document_id": "A", "text": f"a{i}"} for i in range(3)]
    new_ids = store.add_embeddings(new_emb_a, new_meta_a)
    store.save()

    # ensure B vectors are still present
    remaining_b = store.get_vector_ids_for_document("B")
    assert set(remaining_b) == set(ids_b)
    assert store.stats()["ntotal"] == 4
