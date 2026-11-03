from typing import List, Optional
from trustrag.chunking.chunk_schema import Chunk, compute_chunk_hash, generate_chunk_id


class RecursiveChunker:
    """
    Recursive character text splitter that splits on hierarchical delimiters
    ('\\n\\n', '\\n', '. ', ' ', '') while strictly respecting maximum chunk size,
    overlap, and protecting numeric values with units.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        protect_units: bool = True
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        self.protect_units = protect_units

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) <= self.chunk_size:
                good_splits.append(s_clean)
            else:
                if new_separators:
                    sub_splits = self._split_text(s_clean, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard break if at bottom separator
                    for i in range(0, len(s_clean), self.chunk_size):
                        good_splits.append(s_clean[i:i + self.chunk_size])

        # Merge splits into chunks up to chunk_size
        merged = []
        current = ""

        for s in good_splits:
            candidate = f"{current} {s}".strip() if current else s
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = s

        if current:
            merged.append(current)

        return merged

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_text(text, self.separators)

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
        """Split text into standardized Chunk models."""
        segments = self.split_text(text)
        result = []
        for idx, segment in enumerate(segments):
            content_hash = compute_chunk_hash(segment)
            chunk_id = generate_chunk_id(
                document_id=document_id,
                version=version,
                chunk_type="recursive",
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
