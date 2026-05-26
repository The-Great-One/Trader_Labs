# 2026-05-27 — RSI + Momentum 40% XIRR Candidate (PROMOTABLE)

## Summary

RSI 22/44/66 average-rank momentum rotation with a 1-month positive momentum overlay. Only stocks with positive 21-day returns are eligible for the RSI top-N ranking. Monthly rebalance (month-end), top-10 equal-weight positions, 10 bps cost model.

## Data

- 363 symbols from Kite Hist_Data cache
- 2021-04-20 → 2026-05-22 (5.1 years)
- Filtered: 129 derivatives, 41 too-short, 0 stale

## Performance

### In-sample full-period headline
| Metric | Value |
|--------|-------|
| CAGR | 43.32% |
| XIRR | 43.32% |
| Max Drawdown | -21.83% |
| Sharpe | 1.403 |
| Positive Years | 6/6 |
| Worst Year | +1.15% |

### Walk-forward validation (expanding window, 6-month test periods)
| Fold | Period | CAGR | Return | MaxDD | Result |
|------|--------|------|--------|-------|--------|
| 1 | Apr-Oct 2023 | +81.9% | +34.9% | -7.9% | ✅ |
| 2 | Oct 2023-Apr 2024 | +52.0% | +22.7% | -11.4% | ✅ |
| 3 | Apr-Oct 2024 | +320.6% | +103.9% | -12.4% | ✅ |
| 4 | Oct 2024-Apr 2025 | +17.8% | +8.2% | -13.8% | ✅ |
| 5 | Apr-Oct 2025 | +29.0% | +13.6% | -11.4% | ✅ |
| 6 | Oct 2025-Apr 2026 | +20.8% | +9.6% | -15.4% | ✅ |

- **6/6 positive folds**
- Mean WF CAGR: +87.0%
- Worst fold: +17.8%

### Comparison to alternatives
| Strategy | IS XIRR | WF Folds | Worst Fold | Verdict |
|----------|---------|----------|-----------|---------|
| RSI+Momentum ME_top10 | **43.32%** | ✅ 6/6 | +17.8% | **PROMOTABLE** |
| RSI+Momentum ME_top8 | 40.30% | ✅ 6/6 | +10.8% | Promotable |
| RSI+Momentum ME_top6 | 39.51% | ✅ 6/6 | +14.6% | Promotable |
| RSI Pure ME_top6 | 27.40% | ✅ 6/6 | +8.1% | Promotable |
| Equal-weight universe | 26.99% | — | — | Baseline |
| RS7/RS2 (production) | 0.05% | — | — | Current |

### Why this works

The momentum filter eliminates falling stocks from the rotation — stocks with negative 1-month returns are excluded. This prevents the value traps that caused the pure RSI rotation to crash in Oct 2024-Apr 2025 (-46.35% for weekly, -0.71% for monthly).

## Files changed in Trader_Labs

This is a NEW strategy file; no Auto_Trader production files have been modified yet.

- **New**: `scripts/rsi_momentum_report.py` — official research report script
- **New**: `scripts/rsi_momentum_quickscan.py` — quick IS headliner
- **New**: `scripts/rsi_momentum_wf_quick.py` — quick WF validation
- **Updated**: `reports/rsi_momentum_latest.json` — latest validated results
- **Updated**: `reports/rsi_momentum_report_20260527_011109.json` — timestamped report

## Evidence

- `reports/rsi_momentum_latest.json` — full IS + WF results
- Walk-forward: 6/6 positive folds, worst +17.8%, mean +87.0%

## Production patch plan

Productionization path TBD — this is a rotation strategy, not a signal-based one. Options:

1. **Paper trading first**: Add a paper-shadow rotation script that publishes monthly top-10 picks
2. **Hybrid approach**: Build a rotation layer in `WedThurs` that receives RSI+momentum rankings, rebalances monthly, and places orders accordingly
3. **Manual-first**: Run the monthly scan, review picks, and manually place orders for 1-2 months before automating

## Rollback

Remove the rotation script and revert to existing RS7/RS2 production behavior. No database or persistent state changes required.

## Next steps

- [ ] Test on server with live Kite data feed
- [ ] Build paper-shadow rotation script for live monitoring
- [ ] Run 1-2 months manual before full automation
