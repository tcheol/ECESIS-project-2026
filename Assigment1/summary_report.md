# Power Market Constraint Mapping — Summary Report

**Date:** 2026-05-25
**Task:** Map (facility, contingency) constraints across PJM ISO Market, Dayzer, and Panorama

---

## Data Sources

| Source | Rows | Key Columns |
|--------|------|-------------|
| Market (PJM ISO) | 5,230 | CONSTRAINT, CONTINGENCY |
| Dayzer | 13,813 | CID, NAME |
| Panorama | 21,963 | Monitored Facility, Contingency Name |

- Dayzer line constraints (with `:`): 13,174
- Dayzer interface/aggregate entries (no `:`): 639

---

## Match Results

### Market ↔ Pano

| Metric | Count | % of Market |
|--------|-------|-------------|
| Matched to Pano (any method) | 4,619 | 88.3% |
| — Exact (CONSTRAINT key) | 4,533 | 86.7% |
| — Exact (REPORTEDNAME key) | 1,730 | — |
| — Fuzzy high confidence (≥ 92%) | 20 | 0.4% |
| — Fuzzy low confidence (78%–91%) | 51 | 1.0% |
| — Double fuzzy | 15 | 0.3% |
| Unmatched (Market only vs Pano) | 611 | 11.7% |

### Market ↔ Dayzer

| Metric | Count | % of Market |
|--------|-------|-------------|
| Matched to Dayzer (any method) | 4,005 | 76.6% |
| — Exact (CONSTRAINT key) | 3,876 | 74.1% |
| — Exact (REPORTEDNAME key) | 1,450 | — |
| — Fuzzy high confidence | 9 | 0.2% |
| — Fuzzy low confidence | 42 | 0.8% |
| — Double fuzzy | 78 | 1.5% |
| Unmatched (Market only vs Dayzer) | 1,225 | 23.4% |

### Three-Way Matches
| Matched in all 3 sources | 4,000 | 76.5% |

### Orphans (Constraints Not Anchored to Market)

| Source | Orphan Count | % of Source |
|--------|-------------|-------------|
| Pano not in Market | 17,670 | 80.5% |
| Dayzer line not in Market | 9,479 | 72.0% |
| Dayzer interface (no line match) | 639 | 4.6% |

---

## Key Findings & Insights

### 1. Naming Convention Differences
Each source uses a distinct naming dialect:
- **Market**: `STATION kV SHORTCODE` (e.g., `ELIMA 138 KV ELI-HAV1`)
- **Dayzer**: `STATION_kV_SHORTCODE:CONTINGENCY` (underscores, colon separator)
- **Pano**: `ENDPOINT1 kV - ENDPOINT2 kV (STATION kV SHORTCODE)` — parenthetical mirrors Market format

The Pano parenthetical is the closest analog to Market CONSTRAINT, making
Market↔Pano matching the highest-fidelity pair.

### 2. Contingency Equivalences
- `ACTUAL` (Market) = `BASE` (Pano) — both mean the constraint is binding under
  normal operating conditions (no line/transformer outage)
- Line contingency IDs (e.g., `.15502`, `.11215`) are globally unique and serve
  as reliable tiebreakers even when station names are abbreviated differently

### 3. Abbreviation & Annotation Discrepancies
Specific cases resolved by normalization:
- `ELECTRICJCT` ↔ `ELECTRICJUNCTION` (in L345 contingency IDs)
- Trailing region/state tags `[NYISO]`, `(MEC)`, `(Sctnlz)` stripped (short tags only; long parenthetical content is preserved because it carries topology info, e.g. `(MOROCCO-LULU-MILAN-MONROE)`)
- **Underscores treated as separators across all sources** — 804 Market rows use Pano-Format-B-style names like `GRACETON_230KVGRA-MANO_1_LN`; failing to canonicalize these was the largest single source of missed exact matches in the original pass
- Voltage normalization handles both leading and trailing concatenation (`BOONETWN230 KV`, `230KVGRA-MANO`)

### 4. Dayzer Interface Entries
639 Dayzer entries (4.6% of all Dayzer) are aggregate interface
constraints (e.g., `Eastern Interface`, `DPL_S ACTUAL`) with no individual line
counterpart in Market or Pano. These represent zone-level congestion metrics
used for portfolio monitoring but cannot be mapped to specific physical elements.

### 5. Market Rows Without Matches
611 Market rows have no Pano match and 1,225 have no Dayzer match.
Likely causes:
- Constraints recently added to Market not yet reflected in vendor systems
- Regional constraints outside the scope of Dayzer/Pano feeds
- Naming conventions that require additional domain-specific normalization

---

## Methodology

### Normalization Pipeline
1. Uppercase all text
2. Underscores → spaces (all sources)
3. Strip ` L/O ...` suffix from Market facility names
4. Voltage normalization — insert a space when the voltage is glued to alphanumerics on either side, then collapse `138KV` / `138  KV` to `138 KV`
5. Collapse whitespace
6. Pano: extract parenthetical short-form as primary match key
7. Contingency: map `BASE` → `ACTUAL`, strip short trailing tag annotations `[...]` and `(...)` (≤12 chars; long parenthetical content preserved)
8. Contingency dot-notation: expand abbreviations, normalize hyphens per segment

### Matching Algorithm
- **Pass 1** (exact): Hash-join on normalized `facility||contingency` key; also tries REPORTEDNAME as fallback
- **Pass 2** (fuzzy facility): Contingency-anchored; `rapidfuzz.token_sort_ratio` within matching contingency groups
- **Pass 3** (double fuzzy): Fuzzy contingency search + fuzzy facility for residual unmatched rows
- **Pass 4** (interface): Soft-match Dayzer interface entries against Market REPORTEDNAME

### Confidence Scores
- **1.0** — exact key match
- **0.92–0.99** — high-confidence fuzzy match (flagged `fuzzy_high`)
- **0.78–0.91** — low-confidence fuzzy match (flagged `fuzzy_low`, manual review recommended)
- **< 0.78** — below threshold, not matched
- **0.5** — interface soft hint (informational only)

---

## Recommendations

1. **Review low-confidence matches** (confidence < 0.92): approximately 93 rows
2. **Investigate Market-only rows**: 606 constraints have no match in either vendor system;
   these may represent newly-added constraints or data gaps
3. **Dayzer interface mapping**: Work with Dayzer to obtain line-level breakdowns for the
   639 interface entries — this would significantly increase three-way coverage
4. **Threshold calibration**: The current thresholds (HIGH=92, LOW=78) were set
   based on known-pair validation; consider re-calibrating as more ground-truth pairs are identified
