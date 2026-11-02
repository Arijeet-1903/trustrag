import sys
from pathlib import Path

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from trustrag.ingestion.manifest import ManifestManager


def main():
    raw_dir = repo_root / "data" / "raw" / "rbi_docs"
    manifest_path = repo_root / "data" / "sources.json"

    print(f"[*] Scanning {raw_dir}...")
    manager = ManifestManager(manifest_path=manifest_path)
    docs = manager.sync_raw_directory(raw_dir=raw_dir, regulator="RBI")

    print(f"[+] Total registered documents: {len(docs)}")
    for d in docs:
        print(f"  - [{d.document_id}] {d.title[:60]}... ({d.filename})")
    print(f"[+] Manifest updated at: {manifest_path}")


if __name__ == "__main__":
    main()

