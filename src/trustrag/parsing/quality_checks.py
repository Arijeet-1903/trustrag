from typing import List, Tuple, Optional
from trustrag.parsing.metadata_extractor import ExtractedPage


def assess_page_quality(page_number: int, text: str, min_chars_per_page: int = 40) -> Tuple[bool, Optional[str]]:
    """
    Assess whether a page appears to be scanned/image-only or corrupt.

    Returns:
        (is_scanned, warning_message)
    """
    cleaned = text.strip()
    char_count = len(cleaned)

    if char_count == 0:
        return True, f"Page {page_number} has 0 extracted characters. Likely scanned, image-only, or blank."

    if char_count < min_chars_per_page:
        return False, f"Page {page_number} has low character density ({char_count} chars). Check for logos or drawings."

    # Check for excessive unprintable / control characters
    unprintable_ratio = sum(1 for c in cleaned if not c.isprintable()) / char_count
    if unprintable_ratio > 0.15:
        return False, f"Page {page_number} contains {unprintable_ratio:.1%} unprintable characters. Text encoding issue."

    return False, None


def assess_document_quality(pages: List[ExtractedPage]) -> List[str]:
    """Aggregate warnings across all pages of a document."""
    warnings = []
    scanned_count = sum(1 for p in pages if p.is_scanned)

    if pages and scanned_count == len(pages):
        warnings.append("CRITICAL: Entire document appears to be scanned or image-based. OCR required.")
    elif scanned_count > 0:
        warnings.append(f"WARNING: {scanned_count} of {len(pages)} pages appear to be scanned or image-based.")

    for p in pages:
        if p.warning:
            warnings.append(p.warning)

    return warnings
