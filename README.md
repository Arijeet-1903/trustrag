# TrustRAG: Financial Regulatory Compliance RAG System

> **Evaluation-first, table-aware, version-aware, and production-observable RAG system for Indian financial regulations (RBI).**

---

## 📌 Overview

TrustRAG is a document-grounded assistant designed for financial regulatory compliance research. It addresses the critical challenges in regulatory RAG:
1. **Regulatory Provenance**: Page, bounding box (`bbox`), and hierarchical section heading tracking for every extracted passage.
2. **Table-Aware Parsing**: Structured table extraction preserving rows, columns, and headers as Markdown and JSON representations.
3. **Temporal & Version Correctness**: Managing master directions, circulars, and amendments with effective date tracking.
4. **Claim-Level Evidence Verification**: Grounding every statement to verifiable source locations.

---

## 🏛️ Seed Regulatory Corpus (Reserve Bank of India)

| Document ID | Type | Official Reference | Title | Pages | Tables |
|---|---|---|---|---|---|
| `rbi_md_internal_ombudsman_2026` | Master Direction | RBI/CEPD/2025-26/381 | RBI (Commercial Banks - Internal Ombudsman) Directions, 2026 | 16 | 5 |
| `rbi_md_kyc_2016` | Master Direction | RBI/DBR/2015-16/18 | Master Direction - Know Your Customer (KYC) Direction, 2016 | 107 | 21 |
| `rbi_amend_kyc_2025_51` | Amendment | RBI/2025-26/51 | Reserve Bank of India (KYC) (Amendment) Directions, 2025 | 3 | 0 |

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Run Ingestion Pipeline
```bash
python scripts/build_manifest.py
python scripts/ingest_documents.py
```

### 4. Run Tests
```bash
pytest
```

---

## 📁 Repository Structure

```
trustrag/
├── configs/               # Application & ingestion YAML configurations
├── data/
│   ├── raw/rbi_docs/      # Original regulatory PDFs
│   ├── processed/         # Extracted structured JSON (pages, blocks, tables)
│   ├── eval/              # Evaluation questions, golden answers, qrels
│   └── sources.json       # Document provenance manifest with SHA-256 hashes
├── docs/                  # Architecture, data sources, and evaluation specs
├── scripts/               # CLI pipelines (manifest, ingestion, evaluation)
├── src/trustrag/
│   ├── ingestion/         # Source discovery, downloader, hashing, manifest
│   ├── parsing/           # Layout-aware PDF & table parsers, quality checks
│   ├── chunking/          # Fixed, recursive, and structure-aware chunkers
│   ├── indexing/          # BM25, vector store, and index management
│   ├── retrieval/         # Dense, sparse, hybrid search, rerankers, filters
│   ├── generation/        # Prompts, citations, claim verification, abstention
│   ├── temporal/          # Effective dates and regulatory version resolvers
│   ├── evaluation/        # Retrieval, citation, and faithfulness evaluators
│   ├── observability/     # Cost tracking, metrics, and tracing
│   └── api/               # FastAPI endpoints, schemas, dependencies
└── tests/                 # Unit and integration test suites
```

---

## 🧪 Current Status

- [x] **Week 1 — Domain corpus, parsing, and provenance**
  - [x] Seed corpus manifest with SHA-256 verification
  - [x] PyMuPDF layout-aware extraction with heading and table detection
  - [x] Scanned/empty page quality checks
  - [x] 126 pages and 26 tables extracted to structured JSON
  - [x] 5 unit tests passing
- [ ] **Week 2 — Structure-aware chunking, tables, and initial index**
- [ ] **Week 3 — Evaluation set and dense/lexical baseline**
- [ ] **Week 4 — Controlled chunking and retrieval experiments**
- [ ] **Week 5 — Grounded answers, citations, and claim-level verification**
- [ ] **Week 6 — Temporal correctness, reliability, and agentic RAG experiment**
- [ ] **Week 7 — Production-minded API, tracing, caching, and CI**
- [ ] **Week 8 — Final evaluation and portfolio release**
