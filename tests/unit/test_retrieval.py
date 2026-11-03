import pytest
from trustrag.chunking.chunk_schema import Chunk, compute_chunk_hash
from trustrag.parsing.metadata_extractor import (
    ExtractedDocument,
    DocumentMetadata,
    ExtractedPage,
    ExtractedBlock,
)
from trustrag.parsing.table_parser import ExtractedTable
from trustrag.indexing.embeddings import get_embedding_model
from trustrag.indexing.vector_store import VectorStore
from trustrag.indexing.index_manager import IndexManager


@pytest.fixture
def embedding_model():
    return get_embedding_model("deterministic_mock", dimension=384)


@pytest.fixture
def in_memory_store(embedding_model):
    return VectorStore(
        collection_name="test_collection",
        embedding_model=embedding_model,
        use_in_memory=True
    )


def test_embedding_model_properties(embedding_model):
    assert embedding_model.dimension == 384
    v1 = embedding_model.embed_text("Reserve Bank of India Master Direction")
    v2 = embedding_model.embed_text("Reserve Bank of India Master Direction")
    v3 = embedding_model.embed_text("Completely unrelated topic about astronomy")

    assert len(v1) == 384
    assert v1 == v2
    assert v1 != v3


def test_vector_store_add_and_search(in_memory_store):
    chunk1 = Chunk(
        chunk_id="chunk_001",
        document_id="doc_kyc_2016",
        version="1.0",
        chunk_type="paragraph",
        section_path="Chapter II > Definitions",
        page_start=3,
        page_end=3,
        text="A customer identification procedure shall be conducted by the bank.",
        content_hash=compute_chunk_hash("A customer identification procedure shall be conducted by the bank.")
    )
    chunk2 = Chunk(
        chunk_id="chunk_002",
        document_id="doc_kyc_2016",
        version="1.0",
        chunk_type="table",
        table_id="table_limits",
        section_path="Chapter IV > Thresholds",
        page_start=12,
        page_end=12,
        text="Table: Cash limits and thresholds. Maximum cross-border transfer is INR 50,000.",
        content_hash=compute_chunk_hash("Table: Cash limits and thresholds. Maximum cross-border transfer is INR 50,000.")
    )

    in_memory_store.upsert_chunks([chunk1, chunk2])
    assert in_memory_store.count() == 2

    # Verify retrieval of text chunk
    res_text = in_memory_store.search("customer identification procedure", top_k=1)
    assert len(res_text) == 1
    retrieved_chunk, score = res_text[0]
    assert retrieved_chunk.chunk_id == "chunk_001"
    assert retrieved_chunk.document_id == "doc_kyc_2016"
    assert retrieved_chunk.page_start == 3
    assert retrieved_chunk.section_path == "Chapter II > Definitions"

    # Verify retrieval of table chunk
    res_table = in_memory_store.search("cash limits cross-border transfer", top_k=1)
    assert len(res_table) == 1
    retrieved_table_chunk, _ = res_table[0]
    assert retrieved_table_chunk.chunk_type == "table"
    assert retrieved_table_chunk.table_id == "table_limits"
    assert retrieved_table_chunk.page_start == 12


def test_idempotent_reindexing(in_memory_store, tmp_path):
    manifest_file = tmp_path / "test_manifest.json"
    index_mgr = IndexManager(vector_store=in_memory_store, manifest_path=manifest_file)

    doc = ExtractedDocument(
        metadata=DocumentMetadata(
            document_id="rbi_md_test",
            title="Test Regulatory Document",
            content_hash="static_hash_aaa",
            version="1.0"
        ),
        pages=[
            ExtractedPage(
                page_number=1,
                text="Page text",
                blocks=[
                    ExtractedBlock(
                        text="Obligation to report cash transactions.",
                        bbox=(0, 0, 100, 100),
                        block_type="paragraph",
                        page_number=1
                    )
                ]
            )
        ],
        total_pages=1
    )

    # First index operation -> indexed
    res1 = index_mgr.index_document(doc)
    assert res1.status == "indexed"
    initial_count = in_memory_store.count()
    assert initial_count > 0

    # Second index operation with identical document -> skipped_unchanged
    res2 = index_mgr.index_document(doc)
    assert res2.status == "skipped_unchanged"
    assert in_memory_store.count() == initial_count


def test_version_update_purges_stale_chunks(in_memory_store, tmp_path):
    manifest_file = tmp_path / "test_manifest_v.json"
    index_mgr = IndexManager(vector_store=in_memory_store, manifest_path=manifest_file)

    doc_v1 = ExtractedDocument(
        metadata=DocumentMetadata(
            document_id="doc_evolving",
            title="Evolving Circular",
            content_hash="hash_v1",
            version="1.0"
        ),
        pages=[
            ExtractedPage(
                page_number=1,
                text="Version 1 requirement: old limit INR 20,000",
                blocks=[
                    ExtractedBlock(
                        text="Version 1 requirement: old limit INR 20,000",
                        bbox=(0, 0, 100, 100),
                        block_type="paragraph",
                        page_number=1
                    )
                ]
            )
        ],
        total_pages=1
    )

    index_mgr.index_document(doc_v1)
    assert in_memory_store.count() == 1
    old_chunk = in_memory_store.list_chunks()[0]
    assert "old limit" in old_chunk.text

    # Now update document to version 2.0 with new content and new hash
    doc_v2 = ExtractedDocument(
        metadata=DocumentMetadata(
            document_id="doc_evolving",
            title="Evolving Circular",
            content_hash="hash_v2_updated",
            version="2.0"
        ),
        pages=[
            ExtractedPage(
                page_number=1,
                text="Version 2 requirement: amended limit INR 50,000",
                blocks=[
                    ExtractedBlock(
                        text="Version 2 requirement: amended limit INR 50,000",
                        bbox=(0, 0, 100, 100),
                        block_type="paragraph",
                        page_number=1
                    )
                ]
            )
        ],
        total_pages=1
    )

    res_v2 = index_mgr.index_document(doc_v2)
    assert res_v2.status == "updated"
    assert in_memory_store.count() == 1  # Stale version 1 chunk was removed!

    current_chunk = in_memory_store.list_chunks()[0]
    assert current_chunk.version == "2.0"
    assert "amended limit INR 50,000" in current_chunk.text
