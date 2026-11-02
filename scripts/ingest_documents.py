import sys
import json
from pathlib import Path

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from trustrag.ingestion.manifest import ManifestManager
from trustrag.parsing.pdf_parser import PDFParser
from trustrag.parsing.table_parser import extract_tables_from_fitz_page
import fitz


def main():
    raw_dir = repo_root / "data" / "raw" / "rbi_docs"
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_manager = ManifestManager(repo_root / "data" / "sources.json")
    parser = PDFParser(manifest_manager=manifest_manager)

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return

    print(f"[*] Starting ingestion for {len(pdf_files)} PDF document(s)...")

    total_pages = 0
    total_tables = 0

    for pdf_file in pdf_files:
        print(f"\n--> Parsing: {pdf_file.name}")
        extracted = parser.parse_file(pdf_file)

        # Detect tables
        doc = fitz.open(pdf_file)
        doc_tables = []
        for idx, page in enumerate(doc):
            tables = extract_tables_from_fitz_page(page, idx + 1)
            doc_tables.extend(tables)
        doc.close()

        total_pages += extracted.total_pages
        total_tables += len(doc_tables)

        print(f"    Document ID: {extracted.metadata.document_id}")
        print(f"    Pages: {extracted.total_pages} | Tables: {len(doc_tables)}")
        print(f"    Warnings: {len(extracted.parser_warnings)}")

        # Save structured JSON
        out_file = out_dir / f"{extracted.metadata.document_id}_extracted.json"
        dump_data = extracted.model_dump()
        dump_data["tables"] = [t.model_dump() for t in doc_tables]

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)

        print(f"    [SAVED] -> {out_file.name}")

    print("\n" + "=" * 50)
    print(f"[+] Corpus Ingestion Complete!")
    print(f"    Total Documents: {len(pdf_files)}")
    print(f"    Total Pages:     {total_pages}")
    print(f"    Total Tables:    {total_tables}")
    print("=" * 50)


if __name__ == "__main__":
    main()

