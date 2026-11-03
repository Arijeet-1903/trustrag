import re
from typing import List, Optional
from trustrag.chunking.chunk_schema import Chunk, compute_chunk_hash, generate_chunk_id


# Common regulatory number + unit patterns that should never be severed
PROTECTED_PATTERNS = [
    # Currency + amount + multiplier (e.g., "INR 50,000", "Rs. 10 lakh", "Rs 5 crore")
    re.compile(r"(?:INR|Rs\.?|Rupees?|₹)\s*\d+(?:[,\.]\d+)*(?:\s*(?:lakh|crore|thousand|million|billion))?", re.IGNORECASE),
    # Numbers + percentages or time units (e.g., "15 days", "3 years", "5 percent", "10%")
    re.compile(r"\b\d+(?:[,\.]\d+)*\s*(?:%|percent(?:age)?|days?|months?|years?|weeks?|hours?|basis points?|bps)\b", re.IGNORECASE),
    # Regulatory citations (e.g., "Section 35A", "paragraph 38(a)", "Rule 9(14)", "Clause 3(i)")
    re.compile(r"\b(?:Section|Sec\.|Paragraph|Para\.|Clause|Article|Rule)\s+\d+[a-zA-Z0-9]*(?:\([a-zA-Z0-9]+\))*", re.IGNORECASE),
]


def find_protected_spans(text: str) -> List[tuple[int, int]]:
    """Return a sorted list of (start, end) indices for protected numeric and citation spans."""
    spans = []
    for pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    # Sort and merge overlapping spans
    if not spans:
        return []
    spans.sort(key=lambda s: s[0])
    merged = [spans[0]]
    for cur_start, cur_end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


def is_inside_span(idx: int, spans: List[tuple[int, int]]) -> Optional[tuple[int, int]]:
    """Check if character index idx falls strictly inside a protected span."""
    for s_start, s_end in spans:
        if s_start < idx < s_end:
            return (s_start, s_end)
    return None


class FixedSizeChunker:
    """
    Fixed-size chunker with configurable chunk size, overlap, 
    and protection against splitting numeric values from their units.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        protect_units: bool = True
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.protect_units = protect_units

    def split_text(self, text: str) -> List[str]:
        """Split a string into fixed-size segments respecting boundaries and protected units."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        protected_spans = find_protected_spans(text) if self.protect_units else []
        chunks = []
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            if end < total_len:
                # If cut point falls inside a protected span, shift cut point
                span = is_inside_span(end, protected_spans)
                if span:
                    # Prefer extending after the unit if within 25% tolerance, otherwise cut before
                    span_start, span_end = span
                    if span_end - start <= int(self.chunk_size * 1.25):
                        end = span_end
                    else:
                        end = max(start + 1, span_start)

                # Look for a clean word / punctuation break before cut point
                search_start = max(start + self.chunk_size // 2, end - 60)
                punctuation_break = -1
                for punct in [". ", "\n\n", "\n", "; ", ", "]:
                    pos = text.rfind(punct, search_start, end)
                    if pos != -1 and not is_inside_span(pos, protected_spans):
                        punctuation_break = pos + len(punct)
                        break

                if punctuation_break > start:
                    end = punctuation_break
                else:
                    # Fall back to whitespace
                    space_pos = text.rfind(" ", search_start, end)
                    if space_pos > start and not is_inside_span(space_pos, protected_spans):
                        end = space_pos + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= total_len:
                break

            # Calculate next start with overlap, ensuring forward progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def chunk_document(
        self,
        document_id: str,
        text: str,
        version: str = "1.0",
        page_start: int = 1,
        page_end: int = 1,
        section_path: str = "",
        source_uri: Optional[str] = None
    ) -> List[Chunk]:
        """Split text into standardized Chunk models with stable IDs."""
        text_segments = self.split_text(text)
        result = []

        for idx, segment in enumerate(text_segments):
            content_hash = compute_chunk_hash(segment)
            chunk_id = generate_chunk_id(
                document_id=document_id,
                version=version,
                chunk_type="fixed",
                index_or_hash=f"{idx+1:04d}"
            )
            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=version,
                    chunk_type="paragraph",
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    text=segment,
                    content_hash=content_hash,
                    source_uri=source_uri
                )
            )

        return result
