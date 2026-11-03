import sys
import json
import logging
import argparse
from pathlib import Path

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from trustrag.parsing.metadata_extractor import ExtractedDocument
from trustrag.parsing.table_parser import ExtractedTable
from trustrag.chunking.structure_aware import StructureAwareChunker
from trustrag.indexing.embeddings import get_embedding_model
from trustrag.indexing.vector_store import VectorStore
from trustrag.indexing.index_manager import IndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="Build or update vector index from processed documents")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if document hash has not changed")
    args = parser.parse_args()

    processed_dir = repo_root / "data" / "processed"
    persist_dir = str(repo_root / "data" / "indexes" / "chroma")
    manifest_path = repo_root / "data" / "indexes" / "index_manifest.json"

    json_files = list(processed_dir.glob("*_extracted.json"))
    if not json_files:
        logger.error(f"No extracted documents found in {processed_dir}. Run scripts/ingest_documents.py first.")
        return

    logger.info(f"Found {len(json_files)} extracted document(s) in {processed_dir}")

    embedding_model = get_embedding_model("deterministic_mock", dimension=384)
    vector_store = VectorStore(
        collection_name="rbi_regulatory_chunks",
        persist_directory=persist_dir,
        embedding_model=embedding_model,
        use_in_memory=False
    )
    chunker = StructureAwareChunker(chunk_size=600, chunk_overlap=60)
    index_manager = IndexManager(
        vector_store=vector_store,
        chunker=chunker,
        manifest_path=manifest_path
    )

    total_chunks = 0
    for jf in json_files:
        logger.info(f"Processing: {jf.name}")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        tables_data = data.pop("tables", [])
        tables = [ExtractedTable(**t) for t in tables_data]
        doc = ExtractedDocument(**data)

        result = index_manager.index_document(extracted_doc=doc, tables=tables, force_reindex=args.force)
        logger.info(f"  -> Result: {result.status.upper()} | Chunks: {result.chunk_count} | Doc: {result.document_id}")
        total_chunks += result.chunk_count

    logger.info("=" * 60)
    logger.info(f"[+] Indexing Complete! Total Indexed Chunks in Store: {vector_store.count()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
