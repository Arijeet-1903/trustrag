from datetime import datetime, timezone
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document-level metadata tracking regulatory authority, version, and provenance."""
    document_id: str = Field(..., description="Stable logical identity of circular/regulation")
    source_id: Optional[str] = Field(None, description="Official circular/notification reference (e.g. RBI/2025-26/51)")
    circular_number: Optional[str] = Field(None, description="Department circular number")
    title: str = Field(..., description="Official title of the regulatory document")
    regulator: str = Field("RBI", description="Issuing authority (e.g. RBI, SEBI, SEC)")
    document_type: str = Field("Master Direction", description="Regulation, circular, FAQ, master direction, etc.")
    canonical_url: Optional[str] = Field(None, description="Official source URL")
    publication_date: Optional[str] = Field(None, description="Date published in YYYY-MM-DD")
    effective_from: Optional[str] = Field(None, description="Effective from date in YYYY-MM-DD")
    effective_to: Optional[str] = Field(None, description="Effective until date if superseded")
    status: str = Field("active", description="active, superseded, withdrawn, draft, unknown")
    supersedes_document_id: Optional[str] = Field(None, description="Prior document ID superseded by this")
    amends_document_id: Optional[str] = Field(None, description="Document ID amended by this circular")
    version: str = Field("1.0", description="Internal or source version")
    content_hash: str = Field(..., description="SHA-256 hash of original file")
    filename: Optional[str] = Field(None, description="Original filename")
    page_count: Optional[int] = Field(None, description="Total number of pages")
    ingested_at: Optional[str] = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parser_version: str = Field("1.0.0", description="Parser version for reproducibility")


class ExtractedBlock(BaseModel):
    """An individual layout block (paragraph, heading, or table segment) extracted from a page."""
    text: str
    bbox: Tuple[float, float, float, float] = Field(..., description="(x0, y0, x1, y1) bounding box")
    block_type: str = Field("paragraph", description="heading, paragraph, table, list_item, header_footer")
    font_size: float = 10.0
    is_bold: bool = False
    page_number: int


class ExtractedPage(BaseModel):
    """Page-level extraction with provenance and quality flags."""
    page_number: int
    text: str
    blocks: List[ExtractedBlock] = Field(default_factory=list)
    char_count: int = 0
    is_scanned: bool = False
    warning: Optional[str] = None


class ExtractedDocument(BaseModel):
    """Full extracted document retaining complete page/section provenance."""
    metadata: DocumentMetadata
    pages: List[ExtractedPage] = Field(default_factory=list)
    full_text: str = ""
    total_pages: int = 0
    parser_warnings: List[str] = Field(default_factory=list)
    parsed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
