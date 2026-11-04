# Archive

Material from earlier stages, kept unchanged for reference. The current
pipeline in `src/` does not read anything here, except one consistency check
in `src/03_build_tables.py` that compares the first download with the current data.

- `phase1_submission/`: the Phase 1 deliverable as submitted on Oct 4, 2026:
  the workbook (Aug 21 - Sep 18, 20 days), the workbook with Task 6 statistics,
  Diego's `task6_statistics.py` that created those sheets, and the five report figures.
- `../data/archive/raw_phase1_25days/`: the first download (Aug 14 - Sep 18), unadjusted
  SIP bars, one CSV per stock (plus JBHT, excluded for coverage, and the 2-stock IEX pilot).
- `../data/archive/raw_iex_25days/`: the first attempt with the free IEX feed; kept as
  evidence for switching to SIP (several stocks had no complete trading days).
- `../data/archive/processed_phase1/`: the Phase 1 cleaned bars and tables.
