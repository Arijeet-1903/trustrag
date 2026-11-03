import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from trustrag.chunking.chunk_schema import Chunk
from trustrag.chunking.structure_aware import StructureAwareChunker
from trustrag.parsing.metadata_extractor import ExtractedDocument
from trustrag.parsing.table_parser import ExtractedTable
from trustrag.indexing.vector_store import VectorStore

logger = logging.getLogger("trustrag.indexing")


class IndexResult(BaseModel):
    """Result of an index operation for a single document."""
    document_id: str
    version: str
    status: str = Field(..., description="'indexed', 'skipped_unchanged', 'updated'")
    chunk_count: int
    content_hash: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class IndexManifest(BaseModel):
    """Persistent ledger of indexed documents for change detection and idempotency."""
    documents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_chunks: int = 0
    last_rebuild: Optional[str] = None
    embedding_model: str = "deterministic_mock"


class IndexManager:
    """
    Orchestrates structure-aware chunking, embeddings, vector indexing,
    deterministic rebuilds, and idempotent updates.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: Optional[StructureAwareChunker] = None,
        manifest_path: Optional[Path] = None
    ):
        self.vector_store = vector_store
        self.chunker = chunker or StructureAwareChunker()
        self.manifest_path = manifest_path or Path("data/indexes/index_manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> IndexManifest:
        if self.manifest_path and self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return IndexManifest(**data)
            except Exception as e:
                logger.warning(f"Could not load index manifest from {self.manifest_path}: {e}")
        return IndexManifest()

    def _save_manifest(self):
        if self.manifest_path:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest.model_dump(), f, indent=2)

    def index_document(
        self,
        extracted_doc: ExtractedDocument,
        tables: Optional[List[ExtractedTable]] = None,
        force_reindex: bool = False
    ) -> IndexResult:
        """
        Idempotently index an ExtractedDocument.
        - If document content hash and version match the manifest: skips (no-op).
        - If updated or force_reindex: deletes stale chunks and indexes new chunks.
        """
        doc_id = extracted_doc.metadata.document_id
        version = extracted_doc.metadata.version or "1.0"
        content_hash = extracted_doc.metadata.content_hash

        # Check existing manifest record
        existing = self.manifest.documents.get(doc_id)
        if existing and not force_reindex:
            if existing.get("content_hash") == content_hash and existing.get("version") == version:
                # Content has not changed -> idempotent no-op
                return IndexResult(
                    document_id=doc_id,
                    version=version,
                    status="skipped_unchanged",
                    chunk_count=existing.get("chunk_count", 0),
                    content_hash=content_hash
                )

        # Document changed or new: purge existing chunks to prevent stale remnants
        if existing:
            self.vector_store.delete_document(document_id=doc_id)
            status = "updated"
        else:
            status = "indexed"

        # Generate structure-aware chunks
        chunks = self.chunker.chunk_document(extracted_doc=extracted_doc, tables=tables)

        # Upsert into vector store
        self.vector_store.upsert_chunks(chunks)

        # Update manifest
        self.manifest.documents[doc_id] = {
            "version": version,
            "content_hash": content_hash,
            "chunk_count": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "has_tables": bool(tables),
            "page_count": extracted_doc.total_pages
        }
        self.manifest.total_chunks = self.vector_store.count()
        self.manifest.embedding_model = self.vector_store.embedding_model.model_name
        self._save_manifest()

        return IndexResult(
            document_id=doc_id,
            version=version,
            status=status,
            chunk_count=len(chunks),
            content_hash=content_hash
        )

    def rebuild_all(
        self,
        documents: List[Tuple[ExtractedDocument, Optional[List[ExtractedTable]]]]
    ) -> Dict[str, Any]:
        """
        Deterministically wipe and rebuild the entire index from source documents.
        """
        self.vector_store.clear()
        self.manifest = IndexManifest(
            embedding_model=self.vector_store.embedding_model.model_name,
            last_rebuild=datetime.now(timezone.utc).isoformat()
        )

        results = []
        for doc, tables in documents:
            res = self.index_document(extracted_doc=doc, tables=tables, force_reindex=True)
            results.append(res)

        self._save_manifest()
        return {
            "total_documents": len(documents),
            "total_chunks": self.vector_store.count(),
            "results": [r.model_dump() for r in results]
        }

    def remove_document(self, document_id: str) -> int:
        """Remove a document and all its chunks from the index."""
        deleted = self.vector_store.delete_document(document_id)
        if document_id in self.manifest.documents:
            del self.manifest.documents[document_id]
            self.manifest.total_chunks = self.vector_store.count()
            self._save_manifest()
        return deleted

    def get_index_status(self) -> Dict[str, Any]:
        """Return the current health and statistics of the index."""
        return {
            "total_documents": len(self.manifest.documents),
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.vector_store.embedding_model.model_name,
            "documents": self.manifest.documents
        }
