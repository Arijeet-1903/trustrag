from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Standard schema for a document chunk maintaining provenance, 
    table structures, and hierarchical heading path.
    """
    chunk_id: str = Field(..., description="Stable unique chunk ID")
    document_id: str = Field(..., description="Logical document ID")
    version: str = Field("1.0", description="Document version")
    chunk_type: str = Field("paragraph", description="paragraph, heading, table, definition, etc.")
    section_path: str = Field("", description="Hierarchical heading path e.g. Chapter II > Section 4")
    page_start: int = Field(..., description="Starting page in original PDF (1-based)")
    page_end: int = Field(..., description="Ending page in original PDF (1-based)")
    table_id: Optional[str] = Field(None, description="Identifier of table if chunk is a table")
    row_labels: Optional[List[str]] = Field(None, description="Labels for table rows")
    column_labels: Optional[List[str]] = Field(None, description="Labels for table columns")
    text: str = Field(..., description="Human-readable chunk text")
    structured_content: Optional[Dict[str, Any]] = Field(None, description="Structured table JSON if applicable")
    embedding: Optional[List[float]] = Field(None, description="Dense vector embedding")
    content_hash: str = Field(..., description="Hash of chunk text for idempotency")
    source_uri: Optional[str] = Field(None, description="Canonical source URL or filename")
