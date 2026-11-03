import hashlib
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


def compute_chunk_hash(text: str, structured_content: Optional[Dict[str, Any]] = None) -> str:
    """Compute a deterministic SHA-256 hash of chunk content."""
    hasher = hashlib.sha256()
    hasher.update(text.strip().encode("utf-8"))
    if structured_content:
        struct_str = json.dumps(structured_content, sort_keys=True)
        hasher.update(struct_str.encode("utf-8"))
    return hasher.hexdigest()


def generate_chunk_id(
    document_id: str,
    version: str,
    chunk_type: str,
    index_or_hash: str
) -> str:
    """Generate a stable, traceable chunk ID e.g. doc1_v1.0_table_001 or doc1_v1.0_para_a1b2c3d4."""
    clean_version = version.replace(".", "_")
    return f"{document_id}_v{clean_version}_{chunk_type}_{index_or_hash}"


class Chunk(BaseModel):
    """
    Standard schema for a document chunk maintaining provenance, 
    table structures, and hierarchical heading path.
    """
    chunk_id: str = Field(..., description="Stable unique chunk ID")
    document_id: str = Field(..., description="Logical document ID")
    version: str = Field("1.0", description="Document version")
    chunk_type: str = Field("paragraph", description="paragraph, heading, table, table_row, definition, etc.")
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

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to a flat dictionary suitable for vector databases like Chroma."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "version": self.version,
            "chunk_type": self.chunk_type,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "table_id": self.table_id or "",
            "content_hash": self.content_hash,
            "source_uri": self.source_uri or "",
            "has_table": bool(self.table_id),
        }
