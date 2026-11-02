from pathlib import Path
import pytest
from trustrag.ingestion.hashing import compute_sha256
from trustrag.ingestion.manifest import ManifestManager
from trustrag.parsing.quality_checks import assess_page_quality
from trustrag.parsing.pdf_parser import PDFParser
from trustrag.parsing.table_parser import format_table_as_markdown


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def raw_docs_dir(repo_root):
    return repo_root / "data" / "raw" / "rbi_docs"


@pytest.fixture
def manifest(repo_root):
    return ManifestManager(repo_root / "data" / "sources.json")


def test_manifest_loads_seed_documents(manifest):
    docs = manifest.documents
    assert len(docs) >= 3
    assert "rbi_md_internal_ombudsman_2026" in docs
    assert "rbi_md_kyc_2016" in docs
    assert "rbi_amend_kyc_2025_51" in docs


def test_hashing_matches_manifest(raw_docs_dir, manifest):
    target_file = raw_docs_dir / "NT517A41C334FB0A42BC9D58A37DE3793F4E.pdf"
    assert target_file.exists(), f"Sample PDF not found: {target_file}"

    calculated_hash = compute_sha256(target_file)
    meta = manifest.get_by_filename(target_file.name)
    assert meta is not None
    assert calculated_hash == meta.content_hash


def test_quality_checks():
    # Empty page -> detected as scanned / empty
    is_scanned, warning = assess_page_quality(page_number=1, text="", min_chars_per_page=40)
    assert is_scanned is True
    assert "0 extracted characters" in warning

    # Normal page -> healthy
    is_scanned, warning = assess_page_quality(
        page_number=2,
        text="This is a standard regulatory directive text containing valid legal obligations.",
        min_chars_per_page=40
    )
    assert is_scanned is False
    assert warning is None


def test_table_markdown_formatter():
    headers = ["Category", "Threshold (INR)", "Frequency"]
    rows = [["Low Risk", "50,000", "Annual"], ["High Risk", "10,00,000", "Quarterly"]]
    md = format_table_as_markdown(headers, rows)
    assert "| Category | Threshold (INR) | Frequency |" in md
    assert "| Low Risk | 50,000 | Annual |" in md


def test_parse_rbi_amendment_pdf(raw_docs_dir, manifest):
    parser = PDFParser(manifest_manager=manifest)
    target_file = raw_docs_dir / "NT517A41C334FB0A42BC9D58A37DE3793F4E.pdf"

    doc = parser.parse_file(target_file)

    # Metadata assertion
    assert doc.metadata.document_id == "rbi_amend_kyc_2025_51"
    assert doc.metadata.regulator == "RBI"
    assert doc.total_pages == 3
    assert len(doc.pages) == 3

    # Provenance assertion: page 1 must have blocks and text
    page1 = doc.pages[0]
    assert page1.page_number == 1
    assert page1.char_count > 100
    assert not page1.is_scanned
    assert any("Know Your Customer" in b.text for b in page1.blocks)
