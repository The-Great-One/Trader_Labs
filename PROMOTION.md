# Promotion workflow: Trader_Labs → Auto_Trader

Trader_Labs is where research changes live. Auto_Trader is where production/live
runtime lives.

## Promotion checklist

- [ ] Candidate uses only approved/Kite data unless explicitly marked exploratory.
- [ ] No look-ahead/leakage: signals use rows available at decision time;
      execution remains next-open or live-equivalent.
- [ ] Baseline-current comparison included.
- [ ] Walk-forward/OOS metrics included.
- [ ] Full-history winner is labelled `retrospective_only`; only a candidate passing all configured gates is `champion_ready`.
- [ ] Stateful D-close/D+1-open fills, explicit cash, actual-notional fees/turnover, and fail-closed OHLC diagnostics reviewed.
- [ ] Legacy history is archived and excluded; result/history rows carry the same schema and run ID.
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
