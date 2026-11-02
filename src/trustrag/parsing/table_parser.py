from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class ExtractedTable(BaseModel):
    """Structured representation of an extracted regulatory table."""
    table_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    markdown: str = ""
    caption: Optional[str] = None


def format_table_as_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Format headers and row cells into clean GitHub-flavored markdown table syntax."""
    if not headers and not rows:
        return ""

    num_cols = max(len(headers), max((len(r) for r in rows), default=0))
    if num_cols == 0:
        return ""

    # Normalize headers
    normalized_headers = [str(h).strip().replace("\n", " ") if h is not None else "" for h in headers]
    while len(normalized_headers) < num_cols:
        normalized_headers.append(f"Col {len(normalized_headers)+1}")

    header_line = "| " + " | ".join(normalized_headers) + " |"
    sep_line = "| " + " | ".join(["---"] * num_cols) + " |"

    row_lines = []
    for row in rows:
        norm_row = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
        while len(norm_row) < num_cols:
            norm_row.append("")
        row_lines.append("| " + " | ".join(norm_row) + " |")

    return "\n".join([header_line, sep_line] + row_lines)


def extract_tables_from_fitz_page(page, page_num: int) -> List[ExtractedTable]:
    """Extract tables using PyMuPDF table detection."""
    tables = []
    try:
        tabs = page.find_tables()
        for idx, tab in enumerate(tabs.tables):
            extracted = tab.extract()
            if not extracted or len(extracted) == 0:
                continue

            headers = [str(c or "").strip() for c in extracted[0]]
            rows = [[str(c or "").strip() for c in r] for r in extracted[1:]]

            table_id = f"table_p{page_num}_{idx+1}"
            md = format_table_as_markdown(headers, rows)

            tables.append(
                ExtractedTable(
                    table_id=table_id,
                    page_number=page_num,
                    bbox=tab.bbox,
                    headers=headers,
                    rows=rows,
                    markdown=md
                )
            )
    except Exception:
        # Fall back gracefully if table detector encounters page anomalies
        pass

    return tables
