import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 checksum of a file for idempotency and duplicate detection."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_text_hash(text: str) -> str:
    """Compute the SHA-256 checksum of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest().lower()
