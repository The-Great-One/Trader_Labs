# Promotion workflow: Trader_Labs → Auto_Trader

Trader_Labs is where research changes live. Auto_Trader is where production/live
runtime lives.

## Promotion checklist

- [ ] Candidate uses only approved/Kite data unless explicitly marked exploratory.
- [ ] No look-ahead/leakage: signals use rows available at decision time;
      execution remains next-open or live-equivalent.
- [ ] Baseline-current comparison included.
- [ ] Walk-forward/OOS metrics included.
- [ ] Drawdown/trade-count/regime impact reviewed.
- [ ] Candidate patch into Auto_Trader is minimal and live-safe.
- [ ] Verify Auto_Trader tests/import checks after applying.
- [ ] Deploy flow remains: push → pull both servers → verify.

## Suggested promotion note format

Create `promotion_notes/YYYY-MM-DD_candidate-name.md` with:

- Candidate summary
- Files changed in Trader_Labs
- Evidence/reports
- OOS/walk-forward table
- Production patch plan
- Rollback plan
