import re
from typing import List, Optional, Dict, Any, Tuple
from trustrag.chunking.chunk_schema import Chunk, compute_chunk_hash, generate_chunk_id
from trustrag.chunking.fixed_size import FixedSizeChunker
from trustrag.parsing.metadata_extractor import ExtractedDocument, ExtractedBlock, ExtractedPage
from trustrag.parsing.table_parser import ExtractedTable, format_table_as_markdown


class StructureAwareChunker:
    """
    Structure-aware chunker that respects document hierarchy, section headings,
    page boundaries, and handles regulatory tables with row-column preservation.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 60,
        max_table_rows_per_chunk: int = 8,
        include_section_in_text: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows_per_chunk = max_table_rows_per_chunk
        self.include_section_in_text = include_section_in_text
        self.fallback_chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            protect_units=True
        )

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving newlines for structure."""
        return re.sub(r"[ \t]+", " ", text).strip()

    def _is_heading(self, block: ExtractedBlock) -> bool:
        """
        Determine if a block represents an actual structural section heading.
        Even if the parser flagged it as heading due to font size, body sentences
        and multi-sentence blocks are paragraphs, not structural headings.
        """
        text = block.text.strip()
        if not text:
            return False

        if block.block_type in ("table", "header_footer"):
            return False

        # Headings are short titles or labels, not long narrative paragraphs
        if len(text) > 100:
            return False

        heading_patterns = [
            r"^(?:CHAPTER|PART|SECTION)\s+[IVXLCDM0-9]+",
            r"^\d+\.\s+[A-Z][a-zA-Z\s]{2,40}$",
            r"^(?:Short title|Definitions|Application|Commencement|Repeal|Amendments?)\b",
            r"^Reserve Bank of India\s*\(.*?\)\s*Directions",
        ]
        if any(re.match(p, text, re.IGNORECASE) for p in heading_patterns):
            return True

        if block.block_type == "heading" and len(text) < 60 and not text.endswith((".", ";", ",")):
            return True

        return False

    def _build_table_readable_text(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None,
        section_path: str = "",
        page_num: int = 1
    ) -> str:
        """
        Generate human-readable, semantically searchable text representation
        of a table preserving row-column relationships and values.
        """
        lines = []
        if section_path:
            lines.append(f"Section Context: {section_path}")
        if caption:
            lines.append(f"Table Caption: {caption}")
        lines.append(f"Page: {page_num}")

        # Add Markdown rendering
        md_table = format_table_as_markdown(headers, rows)
        if md_table:
            lines.append("\n" + md_table + "\n")

        # Add row-by-row natural language descriptions for semantic retrieval
        norm_headers = [h.strip() or f"Column_{i+1}" for i, h in enumerate(headers)]
        lines.append("Row-by-Row Values:")
        for idx, row in enumerate(rows):
            cell_descriptions = []
            for col_idx, cell in enumerate(row):
                header_name = norm_headers[col_idx] if col_idx < len(norm_headers) else f"Col_{col_idx+1}"
                cell_val = str(cell).strip()
                if cell_val:
                    cell_descriptions.append(f"{header_name}: {cell_val}")
            if cell_descriptions:
                lines.append(f"- Row {idx+1}: {'; '.join(cell_descriptions)}")

        return "\n".join(lines).strip()

    def chunk_table(
        self,
        table: ExtractedTable,
        document_id: str,
        version: str = "1.0",
        section_path: str = "",
        source_uri: Optional[str] = None,
        chunk_index_start: int = 1
    ) -> List[Chunk]:
        """
        Convert an ExtractedTable into one or more structured, self-contained table chunks.
        Large tables are split across rows while preserving headers and captions on every chunk.
        """
        headers = table.headers or []
        rows = table.rows or []
        page_num = table.page_number
        caption = table.caption

        row_labels = [r[0].strip() for r in rows if r and len(r) > 0 and r[0].strip()]
        column_labels = [h.strip() for h in headers if h.strip()]

        chunks = []

        if not rows:
            text = self._build_table_readable_text(headers, [], caption, section_path, page_num)
            struct_content = {
                "table_id": table.table_id,
                "headers": headers,
                "rows": [],
                "page": page_num,
                "caption": caption
            }
            c_hash = compute_chunk_hash(text, struct_content)
            c_id = generate_chunk_id(document_id, version, "table", f"{chunk_index_start:04d}")
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    document_id=document_id,
                    version=version,
                    chunk_type="table",
                    section_path=section_path,
                    page_start=page_num,
                    page_end=page_num,
                    table_id=table.table_id,
                    row_labels=row_labels,
                    column_labels=column_labels,
                    text=text,
                    structured_content=struct_content,
                    content_hash=c_hash,
                    source_uri=source_uri
                )
            )
            return chunks

        row_batches = [
            rows[i:i + self.max_table_rows_per_chunk]
            for i in range(0, len(rows), self.max_table_rows_per_chunk)
        ]

        for sub_idx, row_batch in enumerate(row_batches):
            sub_table_id = (
                table.table_id if len(row_batches) == 1 
                else f"{table.table_id}_part{sub_idx+1}"
            )
            batch_row_labels = [r[0].strip() for r in row_batch if r and len(r) > 0 and r[0].strip()]

            text = self._build_table_readable_text(
                headers=headers,
                rows=row_batch,
                caption=f"{caption} (Part {sub_idx+1}/{len(row_batches)})" if len(row_batches) > 1 and caption else caption,
                section_path=section_path,
                page_num=page_num
            )

            struct_content = {
                "table_id": sub_table_id,
                "parent_table_id": table.table_id,
                "headers": headers,
                "rows": row_batch,
                "row_start_index": sub_idx * self.max_table_rows_per_chunk + 1,
                "page": page_num,
                "caption": caption
            }

            c_hash = compute_chunk_hash(text, struct_content)
            idx_str = f"{chunk_index_start + sub_idx:04d}"
            c_id = generate_chunk_id(document_id, version, "table", idx_str)

            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    document_id=document_id,
                    version=version,
                    chunk_type="table",
                    section_path=section_path,
                    page_start=page_num,
                    page_end=page_num,
                    table_id=sub_table_id,
                    row_labels=batch_row_labels,
                    column_labels=column_labels,
                    text=text,
                    structured_content=struct_content,
                    content_hash=c_hash,
                    source_uri=source_uri
                )
            )

        return chunks

    def chunk_document(
        self,
        extracted_doc: ExtractedDocument,
        tables: Optional[List[ExtractedTable]] = None
    ) -> List[Chunk]:
        """
        Structure-aware chunking for an ExtractedDocument.
        Preserves section hierarchy, page spans, table relationships, and units.
        """
        doc_id = extracted_doc.metadata.document_id
        version = extracted_doc.metadata.version or "1.0"
        source_uri = extracted_doc.metadata.canonical_url or extracted_doc.metadata.filename

        all_chunks: List[Chunk] = []
        chunk_counter = 1

        heading_stack: List[str] = []
        current_section_path = ""

        tables_by_page: Dict[int, List[ExtractedTable]] = {}
        if tables:
            for t in tables:
                tables_by_page.setdefault(t.page_number, []).append(t)

        current_buffer: List[str] = []
        buffer_page_start = 1
        buffer_page_end = 1

        def flush_buffer():
            nonlocal current_buffer, buffer_page_start, buffer_page_end, chunk_counter
            if not current_buffer:
                return

            combined_text = "\n\n".join(current_buffer).strip()
            current_buffer = []

            if not combined_text:
                return

            if len(combined_text) > self.chunk_size:
                sub_segments = self.fallback_chunker.split_text(combined_text)
            else:
                sub_segments = [combined_text]

            for seg in sub_segments:
                seg_text = seg
                if self.include_section_in_text and current_section_path:
                    seg_text = f"[{current_section_path}]\n\n{seg}"

                c_hash = compute_chunk_hash(seg_text)
                c_id = generate_chunk_id(
                    document_id=doc_id,
                    version=version,
                    chunk_type="paragraph",
                    index_or_hash=f"{chunk_counter:04d}"
                )
                chunk_counter += 1

                all_chunks.append(
                    Chunk(
                        chunk_id=c_id,
                        document_id=doc_id,
                        version=version,
                        chunk_type="paragraph",
                        section_path=current_section_path,
                        page_start=buffer_page_start,
                        page_end=buffer_page_end,
                        text=seg_text,
                        content_hash=c_hash,
                        source_uri=source_uri
                    )
                )

        for page in extracted_doc.pages:
            p_num = page.page_number
            page_tables = tables_by_page.get(p_num, [])

            for block in page.blocks:
                block_text = self._clean_text(block.text)
                if not block_text:
                    continue

                if self._is_heading(block):
                    flush_buffer()

                    if re.match(r"^(?:CHAPTER|PART)\s+", block_text, re.IGNORECASE):
                        heading_stack = [block_text]
                    elif re.match(r"^\d+\.\s+", block_text) and len(heading_stack) > 1:
                        heading_stack = [heading_stack[0], block_text]
                    else:
                        if len(heading_stack) >= 3:
                            heading_stack[-1] = block_text
                        else:
                            heading_stack.append(block_text)

                    current_section_path = " > ".join(heading_stack)
                    buffer_page_start = p_num
                    buffer_page_end = p_num
                else:
                    if not current_buffer:
                        buffer_page_start = p_num
                    buffer_page_end = p_num

                    projected_size = sum(len(b) for b in current_buffer) + len(block_text)
                    if projected_size > self.chunk_size and current_buffer:
                        flush_buffer()
                        buffer_page_start = p_num
                        buffer_page_end = p_num

                    current_buffer.append(block_text)

            if page_tables:
                flush_buffer()
                for table in page_tables:
                    table_chunks = self.chunk_table(
                        table=table,
                        document_id=doc_id,
                        version=version,
                        section_path=current_section_path,
                        source_uri=source_uri,
                        chunk_index_start=chunk_counter
                    )
                    chunk_counter += len(table_chunks)
                    all_chunks.extend(table_chunks)

        flush_buffer()
        return all_chunks
