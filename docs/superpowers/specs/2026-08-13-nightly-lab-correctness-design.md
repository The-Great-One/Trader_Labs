# Nightly Lab Correctness and Paper-Parity Design

**Date:** 2026-08-13

**Status:** Reviewed design; ready for implementation planning

**Scope:** `Trader_Labs` nightly research and sibling `Auto_Trader` paper execution only

## 1. Problem and outcome

The adversarial audit found that the nightly lab's reported winners are not promotion-safe. Close-derived signals receive an unrealizable return before a fill, holdings are treated as fixed weights instead of stateful positions, fees depend on changed symbol count instead of traded capital, the configured walk-forward section is not enforced, and history identity/persistence/runner lifecycle are unsafe. The deployed paper ledger then breaks parity by discarding the signal's inverse-volatility target weights and rebuilding an equal-weight book at historical signal-date closes.

The corrected system will use one explicit timing and accounting contract:

1. Compute a strategy signal from the complete daily bar at trading day **D close**.
2. Model research execution at the next trading day's **D+1 open**.
3. Require sufficient point-in-time OHLC coverage for every required sell and buy; otherwise fail the rebalance closed without mutating state.
4. Maintain shares and cash through time. Holdings drift with market prices between rebalances; there is no cost-free daily return to target weights.
5. Charge fees on actual buy and sell notional.
6. Preserve residual cash.
7. Publish the strategy's target weights, including configured inverse-volatility weights.
8. In paper execution, consume those target weights and use fresh live quotes for actual fills. Record the historical/modelled D+1 open and observed slippage for comparison, but never create a retroactive fill.
9. Require configured walk-forward gates before any candidate can be `champion_ready`. A winner selected on full history is always `retrospective_only`.
10. Never place live orders automatically.

The change deliberately invalidates existing headline metrics. Old history remains available as an archive but cannot seed, qualify, rank, or deduplicate the corrected search.

## 2. Scope boundaries

### In scope

- A stateful, next-open portfolio simulator for the focused Trader_Labs closure.
- Point-in-time signal target weights and execution diagnostics.
- Fail-closed OHLC validation.
- Actual-notional turnover and fee accounting.
- Walk-forward evaluation and enforced champion readiness.
- Canonical parameter fingerprints and a clean schema boundary for history.
- Atomic report/history writes and single-run/process-tree-safe runner behavior.
- Auto_Trader shadow output contract and paper-ledger target-weight/live-quote parity.
- Unit and integration tests written test-first.
- Documentation updates after behavior is implemented.

### Out of scope

- Real/live broker orders, Kite reactivation, service deployment, or strategy promotion.
- Changing the strategy family, indicators, default champion parameters, universe rules, or consistency thresholds except where schema validation exposes an invalid configuration.
- Fractional shares in the paper ledger; it remains a whole-share simulator with residual cash.
- Solving current-universe survivorship bias or reconstructing delisted/historical membership data.
- Treating retrospective fixed-candidate folds as genuinely unseen model-selection evidence.
- Reusing or rewriting legacy report rows into the new schema.

## 3. Approaches considered

### A. Patch the current vectorized return loops

This would shift returns by one day, replace changed-name fee counts with weight deltas, and add walk-forward fields around the existing code. It is the smallest textual diff, but fixed weight matrices still make daily rebalancing implicit, cash/rounding behavior remains opaque, and research/paper parity stays fragile. **Rejected.**

### B. Introduce an event-driven stateful simulator and explicit cross-repo signal contract

Trader_Labs owns a small deterministic portfolio simulator. Signals contain target weights and timing metadata. Auto_Trader consumes that contract but retains its own whole-share paper state and fresh-quote fill logic. Shared fixtures assert both sides agree on target intent, while each repository remains independently deployable. **Selected.** It gives auditable timing, notional, holdings drift, and cash without coupling research to production imports.

### C. Create a shared package imported by both repositories

A common simulator package would reduce duplication, but it creates a deployment dependency across two independently versioned repositories and risks allowing research code into the paper runtime. It is unnecessary for the focused closure. **Rejected for now.** The schema and golden fixtures are the shared contract; a package can be reconsidered only if drift recurs.

## 4. Architecture

### 4.1 Trader_Labs modules

- `scripts/rsi_224466_rotation_lab.py` loads and validates point-in-time OHLCV and retains causal indicator/rebalance helpers.
- New `scripts/portfolio_simulator.py` owns immutable signal intents, mutable portfolio state, next-open fills, whole/fractional research units as explicitly configured, fees, cash, valuation, turnover, and execution diagnostics. For the nightly lab, fractional research units are acceptable so target-weight strategy comparisons are not distorted by an arbitrary nominal NAV; cash is nevertheless an explicit state field and remains when a target is infeasible or absent.
- `scripts/auto_iteration_lab.py` owns candidate generation, canonical fingerprints, full-history retrospective ranking, walk-forward folds, incumbent comparison, readiness verdicts, history schema, and atomic publication.
- `scripts/run_auto_iteration_nightly.sh` owns OS-level admission and worker process-group lifecycle, not business logic.

### 4.2 Auto_Trader modules

- `scripts/rsi_momentum_paper_shadow.py` computes the same D-close intent and publishes `target_weights`, `signal_date`, `modeled_execution_date`, model-open diagnostics, parameter fingerprint, and schema version. It does not fill a portfolio.
- `scripts/rsi_momentum_paper_ledger.py` validates an unconsumed signal, prices the whole current book and target book from one fresh live-quote snapshot, computes whole-share deltas from target weights, applies fees to actual notional, retains cash, and commits state atomically. Historical D+1 opens are diagnostics only.
- Existing read-only MTM behavior stays separate from rebalance execution.

No module may import or call a broker order API. The paper ledger remains a simulator.

## 5. Canonical timing contract

For each rebalance:

1. **Signal observation:** Indicators and universe filters use bars ending at D close only.
2. **Target intent:** Selection and configured weighting produce a normalized mapping `symbol -> target_weight` with finite, positive weights whose sum is at most 1. The remainder is target cash.
3. **Model execution:** The research simulator trades once at D+1 open. It cannot earn D close-to-D+1 open returns because it did not own the new basket yet.
4. **Valuation:** Existing D holdings participate in the overnight move through D+1 open. After the fill, new holdings drift through subsequent marks until the next execution.
5. **End of research sample:** In the historical simulator, a D signal without a later valid open is pending/non-actionable and creates no model fill or performance. This does not suppress paper intent: a newly published D-close signal may execute in the real D+1 session from a fresh quote even before the completed D+1 daily bar/model open exists.

Open is the required execution field. Close-only or synthetic OHLC must not be silently substituted.

## 6. OHLC coverage and fail-closed behavior

A proposed model rebalance is valid only when:

- D signal inputs meet existing breadth/freshness requirements;
- a later trading date exists for D+1;
- every held symbol requiring valuation/sale has a finite positive D+1 open;
- every target symbol requiring purchase has a finite positive D+1 open;
- the execution-date coverage threshold configured for the run is met; and
- no duplicate date/symbol or timezone ambiguity survives normalization.

Every daily research valuation also requires a finite positive same-day close for every held symbol. A held symbol may use at most the single, configured close-input forward-fill performed before simulation (default three trading rows); the mark records its source date. If any held mark exceeds that allowance or aggregate held-close coverage falls below the mandatory configured threshold, the candidate fails closed. A missing close is never converted to a zero return.

Validation happens before any cash, share, cost-basis, trade-log, or history mutation. A rejected execution returns a typed diagnostic containing signal date, proposed execution date, missing symbols, coverage, and reason. Candidate simulation fails closed rather than dropping missing names and redistributing their weights. The nightly report remains publishable as a failed run, but the candidate cannot qualify.

Auto_Trader applies the same atomicity rule to actual paper fills. All required held and target names must have a fresh quote from one quote snapshot. If not, it performs no partial rebalance and leaves `last_consumed_signal_id` unchanged so a later run may retry. Historical prices cannot rescue a live fill. The quote contract is `paper_quote_snapshot_v1`: one `snapshot_id`, timezone-aware `generated_at`, `prices`, and per-symbol timezone-aware `price_times`. The producer must subscribe/fetch the union of held and target symbols. Every fill quote must be finite, positive, from that snapshot, and no older than the configured maximum at execution time; the existing after-close unlimited-age exception is not valid for fills (same-day close ticks may remain usable for MTM only).

## 7. Stateful portfolio accounting

The research portfolio state contains:

- cash;
- symbol units/shares;
- last marks;
- cost basis (diagnostic, not needed for return calculation);
- cumulative fees and traded notional;
- executed signal ID/date;
- ordered trade and valuation events.

At D+1 open:

1. Mark the pre-trade portfolio at execution opens to determine NAV.
2. Convert target weights to target notionals from that NAV.
3. Calculate signed symbol deltas from current notionals.
4. Sell reductions first, then buy increases subject to post-fee cash.
5. Charge `abs(delta_units) * fill_price * cost_bps / 10_000` on each actual trade.
6. Retain residual cash; do not renormalize successful names after a missing or infeasible trade.
7. Record one execution event with gross traded notional, one-way turnover (`gross_traded_notional / pre_trade_nav`), fees, cash, and fill details.

Between executions, units remain unchanged. Daily portfolio returns come from changes in cash plus marked position value, so weights drift naturally. Exits are explicit trades at the next permitted execution price and must work for newly acquired positions after their entry event.

## 8. Weighting parity

The signal generator is the authority for target intent. Equal weighting and inverse-volatility weighting are both strategies, not ledger choices.

For configured inverse-volatility weighting:

- volatility uses data through D close only;
- non-finite/zero volatility is rejected or handled by one documented deterministic floor;
- raw inverse vol is normalized once across selected names;
- weights are serialized at sufficient precision and carry `weighting_method` and `vol_lookback` metadata;
- the same target mapping feeds the research simulator and paper signal.

The paper ledger sizes whole shares toward published weights using current fresh quotes and available NAV. Rounding creates observable tracking error and retained cash; the output reports target weight, actual post-fill weight, and deviation per symbol. It must not replace target weights with equal budgets.

## 9. Paper fill and slippage semantics

A signal has a stable `signal_id` derived from strategy schema, canonical parameter fingerprint, signal date, and target weights.

- If the ledger first sees the signal before/on the modeled D+1 session and has fresh quotes, it fills at those actual quote prices even when the completed D+1 daily bar/model open is not available yet.
- If it sees the signal later, it may fill at the current fresh quote only. The trade timestamp is the actual run time. It must not backdate the fill to D or D+1.
- If the same signal ID was consumed, the run is MTM-only and cannot fill again.
- The shadow may attach modeled D+1 opens once that bar exists. The signal carries target-symbol opens when known; for sells of dropped holdings, the ledger looks up the same D+1 OHLC dataset by `modeled_execution_date`. Each fill logs nullable `modeled_open`, `actual_fill`, and `slippage_bps = side * (actual_fill / modeled_open - 1) * 10_000` when both exist. Missing sell or buy model opens remain explicitly null.
- Missing modeled opens make slippage unavailable; they do not authorize a historical fill.

Actual paper fees use actual filled notional. Research fees use modeled filled notional.

## 10. Walk-forward governance

### 10.1 Two distinct labels

- `retrospective_only`: best qualified full-history result. It may guide research but was selected using the evaluated sample.
- `champion_ready`: a candidate that also passes every configured walk-forward gate and beats or retains the re-evaluated incumbent under the same data snapshot and schema.

A full-history winner can never become `champion_ready` merely from consistency gates.

### 10.2 Fold construction

The runtime consumes every key under `config/auto_iteration_lab.json::walk_forward`:

- expanding train windows with `min_train_years`; training starts at the first dataset session and ends on the trading session immediately before the test start;
- half-open calendar test intervals `[test_start, test_end)`, where `test_end = test_start + test_months`; the observed test rows are sessions in that interval, and the next fold advances by `step_months`; require `step_months >= test_months` so scored test intervals cannot overlap;
- at least `min_test_days` and `min_folds`;
- point-in-time calculations truncated to each fold's test end;
- fixed candidates only: the expanding train range supplies causal indicator warm-up and proves that the configured minimum history exists; it does not refit or reselect parameters. On the last training close, construct the candidate intent from data available then, initialize cash-only test capital, and permit its first fill at the first valid test-session open. Holdings then carry only within that fold and are liquidated/marked at the fold's final close; no state carries between folds. Stitched OOS returns concatenate test-session returns in chronological order and include cash days;
- worst fold, stitched Sharpe/drawdown, baseline outperformance ratio, and fold-to-median ratio gates.

For each fold, `candidate_return` and `baseline_return` are compounded decimal returns. `baseline_outperformance = candidate_return - baseline_return`; the configured `min_baseline_outperformance_ratio` is the fraction of valid folds with non-negative outperformance (denominator: all valid folds). `fold_to_median_ratio = best_positive_fold / median_positive_fold`; if there are no positive folds, or the positive median is non-finite/non-positive, the ratio is infinity and the gate fails. Strict worst-fold return remains independent, so relative outperformance cannot rescue a negative candidate fold.

Because the candidate list was generated from full history, these folds are retrospective stability evidence. Reports say so explicitly. A future unseen holdout is needed for a stronger label, but is not part of this change.

### 10.3 Incumbent comparison

Every run re-simulates baseline, prior eligible incumbent, and challengers on the identical data snapshot. `champion_after` is the higher selection-score result between the re-evaluated incumbent and best `champion_ready` challenger. If the incumbent fails current schema/gates, it cannot remain ready but remains visible with failure reasons. A challenger is never promoted merely for beating baseline.

## 11. Parameter identity and novelty

One canonicalization function:

- starts from validated defaults;
- overlays strategy parameters;
- excludes `enhancement`, labels, display names, run IDs, dates, and other metadata;
- normalizes equivalent numeric/list/boolean representations;
- rejects unknown parameters;
- emits sorted canonical JSON; and
- hashes it with SHA-256 as `params_fingerprint`.

The same fingerprint drives within-run deduplication, history lookup, untested-idea selection, incumbent identity, and signal metadata. Run reports separate `novel_configs`, `intentional_retests` (baseline/incumbent), and `accidental_retests` (which must be zero for a successful run).

## 12. Schema migration and persistence

Introduce a new explicit schema version (implementation should use a constant such as `nightly_lab_v6_next_open_stateful`). On first new-schema run:

1. Acquire the run lock.
2. If legacy `reports/auto_iteration_history.jsonl` exists and is not already archived, atomically rename it to a timestamped file under `reports/archive/`.
3. Start a new history file containing only new-schema rows.
4. Never translate legacy metrics into the new schema.
5. Record archive path and row counts in the latest report.

Malformed new history is a visible error with line number; it is never silently skipped. History update is a read/validate/write of the complete next file to a same-filesystem temporary path followed by `fsync` and `os.replace`. Latest reports and shadow/ledger state/output use the same atomic-write helper. For Auto_Trader, state is the sole authoritative commit record. Each committed state carries a monotonically increasing `state_revision`; it is atomically replaced first. The latest output is a rebuildable projection carrying the same revision and is atomically replaced second. If output publication fails, the process returns non-zero; the next run loads authoritative state and regenerates output before considering a new signal. It must never roll state back or replay the already consumed signal. Crash-boundary tests cover failure before state replace, after state replace/before output replace, and after output replace. Thus readers reject mismatched output revisions and no rebalance can be duplicated.

## 13. Runner lifecycle

On the Linux runtime, `run_auto_iteration_nightly.sh` uses `/usr/bin/flock` for atomic single-run admission and launches a small Python `subprocess.Popen(..., start_new_session=True)` supervisor so the worker has a dedicated process group while its exit status is preserved. The shell creates `run_id` once, exports it as `AT_LAB_RUN_ID`, and the lab requires that value in every history row and latest report. The runner records PID, PGID, command identity, run ID, and start time atomically. `stop` sends TERM to the recorded worker group, waits with a timeout, then sends KILL only if needed. `status` validates command identity, PGID, and run ID, not merely PID existence; terminal states clear admission metadata after status delivery.

macOS does not ship `flock` or `setsid`, so local unit tests do not invoke those binaries. They test the Python process-group supervisor directly and test shell state/status logic with a temporary injected lock-command shim. A Linux-only smoke test (or CI/container command with util-linux installed) verifies real `/usr/bin/flock` admission before deployment.

A successful status delivery requires a latest report whose `run_id` and generation time match the current run. Delivery markers are written only after rendering succeeds; failures remain retryable. Runner tests use a temporary fake worker and never start the expensive lab.

## 14. Data contracts

### Nightly result essentials

```json
{
  "schema_version": "nightly_lab_v6_next_open_stateful",
  "run_id": "...",
  "params_fingerprint": "sha256:...",
  "evidence_label": "retrospective_only",
  "champion_ready": false,
  "execution_model": "signal_close_D_fill_open_D_plus_1",
  "accounting": {
    "traded_notional": 0.0,
    "fees": 0.0,
    "cash_final": 0.0,
    "avg_one_way_turnover": 0.0
  },
  "walk_forward": {
    "qualified": false,
    "failures": []
  }
}
```

### Paper signal essentials

```json
{
  "schema_version": "paper_signal_v2_target_weights",
  "signal_id": "sha256:...",
  "signal_date": "YYYY-MM-DD",
  "modeled_execution_date": "YYYY-MM-DD",
  "params_fingerprint": "sha256:...",
  "target_weights": {"AAA": 0.18, "BBB": 0.12},
  "target_cash_weight": 0.0,
  "weighting_method": "inverse_volatility",
  "modeled_open_prices": {"AAA": 100.0, "BBB": 200.0}
}
```

### Paper trade essentials

Each fill records actual timestamp, side, quantity, actual quote, actual notional, fee, modeled open (nullable), and slippage bps (nullable). Ledger state records `last_consumed_signal_id`; outputs report target-versus-actual weights and retained cash.

## 15. Error handling

- Configuration validation occurs before market-data loading and rejects missing/unknown keys, non-finite values, invalid enums, incomplete weights, and cross-field contradictions.
- Data and execution failures are typed and contextual; broad exception swallowing is removed from governance and persistence paths.
- A candidate with an invalid execution is unqualified and reports why.
- An Auto_Trader rebalance failure performs no trades, writes no consumed signal marker, and returns non-zero so cron can retry.
- MTM can use the existing historical fallback for valuation reporting, but a rebalance cannot use fallback/stale prices for actual paper fills.
- Existing output is preserved when a new shadow or ledger candidate is invalid.

## 16. Testing strategy

Implementation follows red-green-refactor. Deterministic fixtures cover:

- D-close selection and first return only after D+1-open fill;
- overnight ownership split between old and new baskets;
- fail-closed missing open for held/target names;
- stateful weight drift and no daily rebalancing;
- notional-based fees invariant to top-N label/count;
- retained cash and inverse-vol target intent;
- drawdown exits after a new entry;
- RSI monotonic gains returning 100 rather than NaN;
- exactly one forward-fill pass;
- all walk-forward config keys affecting/guarding verdicts;
- full-history winner remaining `retrospective_only`;
- incumbent versus challenger selection;
- canonical fingerprints ignoring labels and normalizing defaults;
- legacy archive/new-schema isolation and malformed-history visibility;
- atomic write failure preserving the old file;
- concurrent runner admission, child-tree stop, freshness/run-ID checks, retryable delivery;
- paper target-weight sizing from fresh quotes, no retroactive fills, actual fees, idempotent signal consumption, and slippage logging;
- cross-repo golden signal contract.

No test invokes production shadow/ledger state paths, network APIs, Telegram, server cron, or live orders.

## 17. Rollout and safety

1. Implement and test Trader_Labs simulator/governance locally.
2. Generate a new-schema report against fixtures, then against a copied/local dataset only.
3. Implement and test Auto_Trader signal/ledger parity locally with temporary state and quote fixtures.
4. Review both diffs and run each full suite.
5. Commit separately per repository when implementation is authorized.
6. Do not push, deploy, run production shadow/ledger, or change cron as part of the implementation unless separately requested.
7. Any future paper deployment starts with observation and dry-run comparison; live broker orders remain prohibited.

## 18. Self-review record

Reviewed against the 2026-08-13 adversarial findings and authoritative decisions:

- No placeholders or unresolved timing choices remain.
- Signal/fill timing, cash, fees, state drift, inverse-vol weights, live quote fills, and no-retro-fill rules are explicit.
- Walk-forward readiness and retrospective labeling cannot be conflated.
- Legacy history cannot contaminate new rankings.
- Locking, atomicity, process-tree lifecycle, malformed history, and stale-result delivery are covered.
- Scope stays paper-only and contains no automatic live-order path.
