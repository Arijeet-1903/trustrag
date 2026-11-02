import json
from pathlib import Path
from typing import List, Optional, Dict
from trustrag.parsing.metadata_extractor import DocumentMetadata
from trustrag.ingestion.hashing import compute_sha256

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "sources.json"


class ManifestManager:
    """Manages reading, validating, and updating the corpus document manifest."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path)
        self.documents: Dict[str, DocumentMetadata] = {}
        self.load()

    def load(self) -> List[DocumentMetadata]:
        """Load documents from sources.json."""
        if not self.manifest_path.exists():
            self.documents = {}
            return []

        with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        self.documents = {
            item["document_id"]: DocumentMetadata(**item)
            for item in data
        }
        return list(self.documents.values())

    def save(self) -> None:
        """Write current manifest to sources.json."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [doc.model_dump() for doc in self.documents.values()]
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Lookup metadata by document_id."""
        return self.documents.get(document_id)

    def get_by_filename(self, filename: str) -> Optional[DocumentMetadata]:
        """Lookup metadata by PDF filename."""
        for doc in self.documents.values():
            if doc.filename == filename:
                return doc
        return None

    def upsert_document(self, doc_meta: DocumentMetadata) -> None:
        """Add or update document metadata."""
        self.documents[doc_meta.document_id] = doc_meta
        self.save()

    def sync_raw_directory(self, raw_dir: Path, regulator: str = "RBI") -> List[DocumentMetadata]:
        """Scan raw directory for PDFs and ensure all are registered with accurate hashes."""
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            return list(self.documents.values())

        for pdf_file in raw_path.glob("*.pdf"):
            file_hash = compute_sha256(pdf_file)
            existing = self.get_by_filename(pdf_file.name)
            if existing:
                if existing.content_hash != file_hash:
                    existing.content_hash = file_hash
                    self.upsert_document(existing)
            else:
                doc_id = f"{regulator.lower()}_{pdf_file.stem.lower()[:30]}"
                new_doc = DocumentMetadata(
                    document_id=doc_id,
                    title=pdf_file.stem,
                    regulator=regulator,
                    filename=pdf_file.name,
                    content_hash=file_hash,
                    status="active"
                )
                self.upsert_document(new_doc)
        return list(self.documents.values())
