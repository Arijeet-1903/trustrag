# Data Sources Manifest

This file documents the seed financial regulatory corpus used for TrustRAG development and baseline evaluation.

## Active Corpus (RBI Regulations)

### 1. Internal Ombudsman Directions (2026)
- **Document ID**: bi_md_internal_ombudsman_2026
- **Official Ref / Circular**: RBI/CEPD/2025-26/381 (CEPD.PRD.No.S1027/13.01.019/2025-26)
- **Title**: Reserve Bank of India (Commercial Banks - Internal Ombudsman) Directions, 2026
- **Issuing Regulator**: Reserve Bank of India (RBI)
- **Document Type**: Master Direction
- **Publication & Effective Date**: 2026-01-14
- **Local File**: data/raw/rbi_docs/380MD14012026DAD56B73DC944F02AEA4530EC779AD7F.pdf (16 pages)
- **Status**: Active
- **SHA-256**: cac4244ff5e1888f6159a987878a9a4f3d57e065770b3bbcce0dccee00deb422

---

### 2. Master Direction - Know Your Customer (KYC) Directions, 2016
- **Document ID**: bi_md_kyc_2016
- **Official Ref / Circular**: RBI/DBR/2015-16/18 (DBR.AML.BC.No.81/14.01.001/2015-16)
- **Title**: Master Direction - Know Your Customer (KYC) Direction, 2016 (Updated as on August 14, 2025)
- **Issuing Regulator**: Reserve Bank of India (RBI)
- **Document Type**: Master Direction
- **Publication Date**: 2016-02-25
- **Version**: Updated as on August 14, 2025
- **Local File**: data/raw/rbi_docs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.pdf (107 pages)
- **Status**: Active
- **SHA-256**: e525b18753fd08ffad04898202402cad37ad9be10ffa088b7ec2a37e288670ad

---

### 3. KYC Amendment Directions, 2025
- **Document ID**: bi_amend_kyc_2025_51
- **Official Ref / Circular**: RBI/2025-26/51 (DOR.AML.REC.30/14.01.001/2025-26)
- **Title**: Reserve Bank of India (Know Your Customer (KYC)) (Amendment) Directions, 2025
- **Issuing Regulator**: Reserve Bank of India (RBI)
- **Document Type**: Amendment Notification
- **Publication Date**: 2025-06-12
- **Effective From**: 2025-06-12
- **Amends Document**: bi_md_kyc_2016
- **Local File**: data/raw/rbi_docs/NT517A41C334FB0A42BC9D58A37DE3793F4E.pdf (3 pages)
- **Status**: Active
- **SHA-256**: eccb870209a187c6b4e98062f1af863a287428e08706ce17fe3d59597311242

---

## Machine-Readable Manifest

See [data/sources.json](../data/sources.json) for the JSON format parsed by src/trustrag/ingestion/manifest.py.
