import re
from pathlib import Path
from typing import Optional, List, Union
import fitz  # PyMuPDF

from trustrag.parsing.metadata_extractor import (
    DocumentMetadata,
    ExtractedBlock,
    ExtractedPage,
    ExtractedDocument,
)
from trustrag.parsing.table_parser import extract_tables_from_fitz_page
from trustrag.parsing.quality_checks import assess_page_quality, assess_document_quality
from trustrag.ingestion.hashing import compute_sha256
from trustrag.ingestion.manifest import ManifestManager

HEADING_PATTERN = re.compile(
    r"^(CHAPTER\s+[IVXLCDM\d]+|SECTION\s+\d+|PART\s+[A-Z\d]+|\d+\.\s+[A-Z]|\bDefinitions\b|\bShort title\b|\bApplicability\b)",
    re.IGNORECASE
)


class PDFParser:
    """
    Layout-aware regulatory PDF parser preserving section hierarchy, 
    page provenance, structured tables, and content quality checks.
    """

    def __init__(self, manifest_manager: Optional[ManifestManager] = None, min_chars_per_page: int = 40):
        self.manifest_manager = manifest_manager or ManifestManager()
        self.min_chars_per_page = min_chars_per_page

    def parse_file(
        self, 
        file_path: Union[str, Path], 
        metadata: Optional[DocumentMetadata] = None
    ) -> ExtractedDocument:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        file_hash = compute_sha256(path)

        # Retrieve metadata or build base metadata
        if metadata is None:
            metadata = self.manifest_manager.get_by_filename(path.name)
            if metadata is None:
                metadata = DocumentMetadata(
                    document_id=f"rbi_{path.stem.lower()[:32]}",
                    title=path.stem,
                    filename=path.name,
                    content_hash=file_hash,
                    status="active"
                )

        doc = fitz.open(path)
        extracted_pages: List[ExtractedPage] = []
        all_warnings: List[str] = []
        full_text_parts: List[str] = []

        try:
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]

                raw_text = page.get_text("text") or ""
                page_dict = page.get_text("dict") or {}
                extracted_blocks: List[ExtractedBlock] = []

                for b in page_dict.get("blocks", []):
                    if b.get("type") == 0 and "lines" in b:  # Text block
                        block_text_lines = []
                        max_size = 0.0
                        is_bold = False

                        for line in b["lines"]:
                            line_spans = []
                            for span in line.get("spans", []):
                                line_spans.append(span.get("text", ""))
                                max_size = max(max_size, span.get("size", 10.0))
                                if span.get("flags", 0) & 2 != 0:
                                    is_bold = True
                            block_text_lines.append(" ".join(line_spans))

                        block_text = "\n".join(block_text_lines).strip()
                        if block_text:
                            is_heading = max_size >= 12.0 or is_bold or bool(HEADING_PATTERN.match(block_text))
                            block_type = "heading" if is_heading else "paragraph"

                            extracted_blocks.append(
                                ExtractedBlock(
                                    text=block_text,
                                    bbox=tuple(b.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                                    block_type=block_type,
                                    font_size=round(max_size, 2),
                                    is_bold=is_bold,
                                    page_number=page_num
                                )
                            )

                # Assess page quality (scanned detection, empty checks)
                is_scanned, warning = assess_page_quality(
                    page_number=page_num,
                    text=raw_text,
                    min_chars_per_page=self.min_chars_per_page
                )
                if warning:
                    all_warnings.append(warning)

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=raw_text,
                        blocks=extracted_blocks,
                        char_count=len(raw_text.strip()),
                        is_scanned=is_scanned,
                        warning=warning
                    )
                )
                full_text_parts.append(raw_text)

        finally:
            doc.close()

        metadata.page_count = len(extracted_pages)
        metadata.content_hash = file_hash

        doc_quality_warnings = assess_document_quality(extracted_pages)
        for w in doc_quality_warnings:
            if w not in all_warnings:
                all_warnings.append(w)

        return ExtractedDocument(
            metadata=metadata,
            pages=extracted_pages,
            full_text="\n\n--- PAGE BREAK ---\n\n".join(full_text_parts),
            total_pages=len(extracted_pages),
            parser_warnings=all_warnings
        )
