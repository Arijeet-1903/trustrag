import pytest
from trustrag.chunking.chunk_schema import (
    Chunk,
    compute_chunk_hash,
    generate_chunk_id,
)
from trustrag.chunking.fixed_size import FixedSizeChunker, find_protected_spans
from trustrag.chunking.recursive import RecursiveChunker
from trustrag.chunking.structure_aware import StructureAwareChunker
from trustrag.parsing.metadata_extractor import (
    ExtractedDocument,
    DocumentMetadata,
    ExtractedPage,
    ExtractedBlock,
)
from trustrag.parsing.table_parser import ExtractedTable


def test_chunk_hash_and_id_determinism():
    text1 = "All regulated entities shall maintain KYC records for 5 years."
    text2 = "All regulated entities shall maintain KYC records for 5 years."
    text3 = "All regulated entities shall maintain KYC records for 10 years."

    h1 = compute_chunk_hash(text1)
    h2 = compute_chunk_hash(text2)
    h3 = compute_chunk_hash(text3)

    assert h1 == h2
    assert h1 != h3

    id1 = generate_chunk_id("doc_kyc", "1.0", "paragraph", "0001")
    id2 = generate_chunk_id("doc_kyc", "1.0", "paragraph", "0001")
    id_v2 = generate_chunk_id("doc_kyc", "2.0", "paragraph", "0001")

    assert id1 == id2
    assert id1 != id_v2
    assert "doc_kyc_v1_0_paragraph_0001" == id1


def test_fixed_size_chunker_basic():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunker = FixedSizeChunker(chunk_size=35, chunk_overlap=10, protect_units=False)
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    # Verify no chunk exceeds chunk_size by a crazy margin
    for c in chunks:
        assert len(c) <= 45


def test_protected_numbers_and_units():
    text = (
        "The regulated entity shall report any cash transaction exceeding INR 50,000 "
        "within 15 days of the occurrence pursuant to Section 35A of the Banking Regulation Act."
    )
    # Target cut right near "INR 50,000"
    chunker = FixedSizeChunker(chunk_size=65, chunk_overlap=10, protect_units=True)
    chunks = chunker.split_text(text)

    full_reconstructed = " ".join(chunks)
    assert "INR 50,000" in full_reconstructed
    assert "15 days" in full_reconstructed
    assert "Section 35A" in full_reconstructed

    # None of the chunks should have broken "INR" from "50,000"
    for c in chunks:
        if "INR" in c:
            assert "INR 50,000" in c or c.endswith("INR 50,000") or c.startswith("INR 50,000")


def test_recursive_chunker():
    text = "Paragraph 1 line A.\nParagraph 1 line B.\n\nParagraph 2 line C.\nParagraph 2 line D."
    chunker = RecursiveChunker(chunk_size=40, chunk_overlap=5)
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 45


def test_structure_aware_heading_preservation():
    doc = ExtractedDocument(
        metadata=DocumentMetadata(
            document_id="test_doc_001",
            title="Test Master Direction",
            content_hash="hash123",
            version="1.0"
        ),
        pages=[
            ExtractedPage(
                page_number=1,
                text="Page 1",
                blocks=[
                    ExtractedBlock(
                        text="CHAPTER I - PRELIMINARY",
                        bbox=(50, 50, 200, 70),
                        block_type="heading",
                        page_number=1
                    ),
                    ExtractedBlock(
                        text="1. Short Title and Commencement. These directions shall be called KYC Directions.",
                        bbox=(50, 80, 400, 100),
                        block_type="paragraph",
                        page_number=1
                    ),
                    ExtractedBlock(
                        text="CHAPTER II - GENERAL OBLIGATIONS",
                        bbox=(50, 150, 250, 170),
                        block_type="heading",
                        page_number=1
                    ),
                    ExtractedBlock(
                        text="2. Designated Director shall be nominated by the board to ensure compliance.",
                        bbox=(50, 180, 400, 200),
                        block_type="paragraph",
                        page_number=1
                    ),
                ]
            )
        ],
        total_pages=1
    )

    chunker = StructureAwareChunker(chunk_size=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    # Verify section paths
    assert "CHAPTER I" in chunks[0].section_path
    assert "CHAPTER II" in chunks[1].section_path
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].chunk_type == "paragraph"
    assert "CHAPTER I" in chunks[0].text


def test_table_chunking_with_headers_and_structure():
    table = ExtractedTable(
        table_id="table_p2_1",
        page_number=2,
        bbox=(50, 100, 500, 300),
        headers=["Risk Category", "Update Frequency", "Verification Threshold"],
        rows=[
            ["Low Risk", "Every 10 years", "INR 50,000"],
            ["Medium Risk", "Every 8 years", "INR 2,00,000"],
            ["High Risk", "Every 2 years", "INR 10,00,000"],
        ],
        caption="Customer Risk Categorization Matrix"
    )

    chunker = StructureAwareChunker(chunk_size=600, max_table_rows_per_chunk=10)
    chunks = chunker.chunk_table(
        table=table,
        document_id="rbi_md_kyc_2016",
        version="1.0",
        section_path="Chapter VI > Risk Management"
    )

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_type == "table"
    assert c.table_id == "table_p2_1"
    assert c.page_start == 2
    assert c.column_labels == ["Risk Category", "Update Frequency", "Verification Threshold"]
    assert "Low Risk" in c.row_labels
    assert "High Risk" in c.row_labels

    # Verify Markdown representation
    assert "| Risk Category | Update Frequency | Verification Threshold |" in c.text
    assert "| Low Risk | Every 10 years | INR 50,000 |" in c.text

    # Verify structured JSON
    assert c.structured_content is not None
    assert c.structured_content["headers"] == ["Risk Category", "Update Frequency", "Verification Threshold"]
    assert len(c.structured_content["rows"]) == 3


def test_large_table_splitting_preserves_headers():
    rows = [[f"Row_{i}", f"Data_{i}", f"Val_{i}"] for i in range(15)]
    table = ExtractedTable(
        table_id="table_large_p4",
        page_number=4,
        bbox=(50, 100, 500, 600),
        headers=["Col A", "Col B", "Col C"],
        rows=rows,
        caption="Large Regulatory Schedule"
    )

    # Chunk with max 5 rows per chunk -> 3 parts
    chunker = StructureAwareChunker(max_table_rows_per_chunk=5)
    chunks = chunker.chunk_table(
        table=table,
        document_id="rbi_large",
        version="1.0"
    )

    assert len(chunks) == 3
    for idx, c in enumerate(chunks):
        assert c.chunk_type == "table"
        assert c.column_labels == ["Col A", "Col B", "Col C"]
        # Every chunk repeats headers in text
        assert "| Col A | Col B | Col C |" in c.text
        assert f"Part {idx+1}/3" in c.text
        assert len(c.structured_content["rows"]) == 5

