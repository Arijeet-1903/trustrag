# TrustRAG Architecture & Week 1 System Specification

## 1. Compliance Use Case & Domain Definition

### 1.1 Target Regulatory Domain
- **Regulator**: Reserve Bank of India (RBI)
- **Domain**: Indian Banking & Financial Services Regulatory Compliance
- **Initial Document Types**:
  1. **Master Directions (MD)**: Standing consolidated instructions issued to regulated entities (e.g., KYC Directions 2016, Internal Ombudsman Directions 2026).
  2. **Circulars & Amendment Notifications**: Point-in-time modifications that amend, introduce, or supersede specific paragraphs or schedules of Master Directions (e.g., KYC Amendment Directions 2025).

### 1.2 Supported Question Types
1. **Direct Regulatory Obligations & Definitions**: Exact obligations, designated directors, reporting thresholds, and regulatory definitions.
2. **Numeric Limits & Periodic Frequencies**: Thresholds (e.g., INR limits for cash transactions, timeline limits for complaint escalation).
3. **Table & Schedule Lookups**: Categorizations, document matrices (e.g., officially valid documents for individuals vs. legal entities).
4. **Temporal / Version-Specific Inquiries**: Identifying whether a clause is active as of a specific date, or how an amendment modifies a parent Master Direction.
5. **Unanswerable / Abstention Cases**: Detecting when an obligation is not governed by the indexed corpus and abstaining rather than hallucinating.

---

## 2. Ingestion & Provenance Architecture (Week 1 Deliverable)

```
[Public Regulatory Source / PDF]
              │
              ▼
   [hashing.py (SHA-256)]
              │
              ▼
   [manifest.py (sources.json)] ───► Stable Document ID, URLs, Dates
              │
              ▼
    [pdf_parser.py (PyMuPDF)]
     ├─► Page layout & bounding boxes
     ├─► Heading detection (regex + font size)
     ├─► quality_checks.py (scanned / low-density warnings)
     └─► table_parser.py (GFM Markdown tables & structured rows)
              │
              ▼
  [ExtractedDocument JSON] ───► data/processed/{doc_id}_extracted.json
```

### 2.1 Provenance Guarantees
- Every text block retains `page_number`, bounding box coordinates (`bbox`), and detected `block_type` (`heading`, `paragraph`, `table`).
- Table rows and headers are formatted as standard GitHub-Flavored Markdown to prevent losing column/row associations.
- Every document is hashed with SHA-256 before ingestion to ensure deterministic, idempotent processing.
- Scanned, corrupted, or low-density pages generate structured parser warnings rather than silently disappearing.

