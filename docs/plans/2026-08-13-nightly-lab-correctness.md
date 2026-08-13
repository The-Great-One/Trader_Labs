# Nightly Lab Correctness and Paper-Parity Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Keep the two repositories independently reviewable and do not push or deploy without explicit authorization.

**Goal:** Replace optimistic nightly-lab accounting with a fail-closed D-close/D+1-open stateful simulator, enforce walk-forward champion readiness, and make Auto_Trader paper fills honor published target weights using fresh quotes without retroactive execution.

**Architecture:** Trader_Labs gains a deterministic event-driven portfolio simulator plus versioned governance/persistence contracts. Auto_Trader keeps an independent whole-share ledger but consumes the shadow's versioned target-weight signal and actual quote snapshot. Golden fixtures test the contract across repositories without creating a runtime import dependency.

**Tech Stack:** Python 3, pandas, NumPy, pytest/unittest, JSON/JSONL, POSIX `flock`, Bash, Git.

**Design:** `docs/superpowers/specs/2026-08-13-nightly-lab-correctness-design.md`

---

## Safety and execution rules

- Work in `/Users/sahilgoel/Desktop/Projects/trading/Trader_Labs` and sibling `/Users/sahilgoel/Desktop/Projects/trading/Auto_Trader`.
- Read both `.hermes.md` files before starting and recheck each repository's status before every commit.
- Use TDD: write one focused failing test, run it and observe the intended failure, implement minimally, rerun, then refactor.
- Use temporary directories/files in tests. Never run production shadow or ledger commands: they write live paper state.
- Do not start the expensive nightly job during routine checks.
- Do not push, deploy, edit server cron, restart services, unmask Kite, or add any live-order path.
- Preserve unrelated working-tree changes and stage only named files.

## Task 1: Establish next-open OHLC and RSI primitives

**Objective:** Make data inputs and indicator behavior explicit before replacing portfolio accounting.

**Files:**
- Modify: `Trader_Labs/scripts/rsi_224466_rotation_lab.py`
- Modify: `Trader_Labs/tests/test_rsi_rotation_lab.py`
- Modify: `Auto_Trader/scripts/rsi_224466_rotation_lab.py`
- Create: `Auto_Trader/tests/test_rsi_rotation_lab.py`

**Step 1: Write failing Trader_Labs tests**

Add fixtures proving:

- `load_ohlc_prices(...)` returns aligned `open` and `close` frames from mixed-case feather columns after timezone normalization;
- duplicate dates are rejected or deterministically deduplicated before alignment;
- no synthetic open is created from close;
- `rsi_dataframe` returns `100.0` for a continuous all-gain window and `0.0` for all-loss;
- the existing XIRR test is updated to the new complete metric contract rather than referencing a missing dataclass field.

**Step 2: Run the focused tests and verify RED**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
pytest -q tests/test_rsi_rotation_lab.py
```

Expected: failures for missing OHLC loader and monotonic RSI semantics (plus the pre-existing `xirr_pct` contract failure until resolved).

**Step 3: Implement the minimum causal primitives**

Add an OHLC loader that requires real date/open/close columns, normalizes dates once, reports skip reasons, and returns raw aligned frames without forward filling. Fix RSI zero-loss/zero-gain cases explicitly. Decide and document the metric API: either add calendar-inclusive `xirr_pct` or replace the stale test and all callers with the versioned metric field; do not leave a red test.

Copy the same audited primitives to Auto_Trader or apply an equivalent minimal patch there; keep file hashes equal if the full helper files remain intentionally mirrored.

**Step 4: Verify GREEN in both repositories**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
pytest -q tests/test_rsi_rotation_lab.py
python3 -m py_compile scripts/rsi_224466_rotation_lab.py

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
pytest -q tests/test_rsi_rotation_lab.py
python3 -m py_compile scripts/rsi_224466_rotation_lab.py
```

Expected: focused suites pass; syntax checks exit 0.

**Step 5: Commit per repository**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
git add scripts/rsi_224466_rotation_lab.py tests/test_rsi_rotation_lab.py
git commit -m "fix: require real OHLC for next-open research"

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
git add scripts/rsi_224466_rotation_lab.py tests/test_rsi_rotation_lab.py
git commit -m "fix: align causal OHLC and RSI primitives"
```

## Task 2: Build the stateful research portfolio simulator

**Objective:** Model shares/units, cash, next-open fills, drift, notional turnover, and fees independently of strategy selection.

**Files:**
- Create: `Trader_Labs/scripts/portfolio_simulator.py`
- Create: `Trader_Labs/tests/test_portfolio_simulator.py`

**Step 1: Write failing event-accounting tests**

Create small deterministic open/close fixtures asserting:

- a D-close signal fills only at D+1 open;
- the new basket receives no D-close-to-D+1-open return;
- the old basket receives the overnight move before its sale;
- units do not change between fills and weights drift with prices;
- cash is retained;
- fees equal `sum(abs(quantity_delta) * fill_price) * bps / 10_000`;
- equal traded notional incurs equal fees regardless of top-5/top-10 labels;
- turnover is actual gross traded notional divided by pre-trade NAV;
- a D signal at sample end is pending/non-actionable.
- every held-position close mark is finite, positive, within the single configured forward-fill allowance, and missing/stale close coverage fails the candidate rather than creating a zero return.

**Step 2: Run and verify RED**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
pytest -q tests/test_portfolio_simulator.py
```

Expected: import failure because `scripts.portfolio_simulator` does not exist.

**Step 3: Implement minimal event model**

Define typed records such as `SignalIntent`, `Fill`, `PortfolioState`, `ExecutionEvent`, and `SimulationResult`. Keep selection logic out. Implement pre-trade open valuation, sell-first/buy-second target deltas, explicit cash, per-fill fees, close marks, and daily NAV/return output. Use one documented volatility floor only at signal construction, not inside the simulator.

**Step 4: Verify GREEN and invariants**

```bash
pytest -q tests/test_portfolio_simulator.py
python3 -m py_compile scripts/portfolio_simulator.py
```

Expected: all event-accounting tests pass and syntax exits 0.

**Step 5: Commit**

```bash
git add scripts/portfolio_simulator.py tests/test_portfolio_simulator.py
git commit -m "feat: add stateful next-open research simulator"
```

## Task 3: Fail closed on incomplete execution OHLC

**Objective:** Prevent partial or silently redistributed research rebalances.

**Files:**
- Modify: `Trader_Labs/scripts/portfolio_simulator.py`
- Modify: `Trader_Labs/tests/test_portfolio_simulator.py`

**Step 1: Write failing tests**

Cover missing/non-finite/non-positive D+1 opens for held sells and target buys, insufficient mandatory execution coverage, missing/stale daily closes for held positions, insufficient mandatory held-close coverage, duplicate signal IDs, and mutation rollback. Assert the full state/event log is byte-for-byte unchanged on failure and diagnostics identify signal date, execution/valuation date, missing symbols, source-mark age, and coverage.

**Step 2: Verify RED**

```bash
pytest -q tests/test_portfolio_simulator.py -k "missing or coverage or atomic"
```

Expected: failures showing partial execution or missing typed diagnostics.

**Step 3: Add preflight validation**

Validate every required price and threshold before copying/committing state. Return/raise one typed `ExecutionDataError`. Do not drop a name, substitute close, forward-fill open, or renormalize remaining weights.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_portfolio_simulator.py
```

Expected: complete simulator suite passes.

**Step 5: Commit**

```bash
git add scripts/portfolio_simulator.py tests/test_portfolio_simulator.py
git commit -m "fix: fail closed on incomplete model fills"
```

## Task 4: Route nightly candidates through the new simulator

**Objective:** Replace the optimistic `_simulate` return loop while preserving causal selection and configured target weighting.

**Files:**
- Modify: `Trader_Labs/scripts/auto_iteration_lab.py`
- Modify: `Trader_Labs/tests/test_auto_iteration_consistency.py`
- Create: `Trader_Labs/tests/test_auto_iteration_execution.py`

**Step 1: Write failing integration tests**

Use a deterministic OHLC fixture and assert:

- selections use D close and execute at D+1 open;
- `_simulate` returns simulator-derived NAV, cash, fills, traded notional, fees, and one-way turnover;
- inverse-volatility target weights are computed from closes through D only and passed unchanged into `SignalIntent`;
- only one `ffill(limit=3)` transformation exists at the designated close-input boundary;
- drawdown exits can execute for positions bought at the prior fill and create actual sell fees;
- missing required opens disqualify the candidate with a visible reason.

**Step 2: Verify RED**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
pytest -q tests/test_auto_iteration_execution.py
```

Expected: current `_simulate` earns close-to-close returns, counts names for fees, fixes weights, and lacks execution diagnostics.

**Step 3: Refactor strategy intent construction**

Extract a pure function that produces picks and target weights at D close, including sector cap, inverse vol, thresholds, and exits. Feed intents plus OHLC to `portfolio_simulator`. Remove `prev_picks` name-count turnover, fee amortization, fixed daily weights, and duplicate forward-fill. Keep CAGR report-only.

**Step 4: Verify GREEN plus existing consistency tests**

```bash
pytest -q tests/test_auto_iteration_execution.py tests/test_auto_iteration_consistency.py tests/test_portfolio_simulator.py
python3 -m py_compile scripts/auto_iteration_lab.py
```

Expected: all focused tests pass and syntax exits 0.

**Step 5: Commit**

```bash
git add scripts/auto_iteration_lab.py tests/test_auto_iteration_consistency.py tests/test_auto_iteration_execution.py
git commit -m "fix: use stateful next-open nightly accounting"
```

## Task 5: Validate all governance configuration

**Objective:** Reject invalid/dead configuration before loading market data and prove every governance key has a runtime consumer.

**Files:**
- Modify: `Trader_Labs/config/auto_iteration_lab.json`
- Modify: `Trader_Labs/scripts/auto_iteration_lab.py`
- Create: `Trader_Labs/tests/test_auto_iteration_config.py`

**Step 1: Write failing tests**

Parameterize missing sections, unknown keys, wrong types, NaN/infinity, invalid rebalance enums, incomplete score weights, impossible bounds, overlapping/non-progressing folds, and an intentionally altered value for each consistency/walk-forward key. Assert validation runs before `lab_load_prices` using a mock.

Add mandatory `min_execution_open_coverage`, `min_held_close_coverage`, and `max_close_ffill_rows` settings plus schema settings; give each a test and runtime read. The loader performs that close forward-fill once and preserves source-date metadata for the simulator.

**Step 2: Verify RED**

```bash
pytest -q tests/test_auto_iteration_config.py
```

Expected: unknown/dead/invalid keys currently pass or fail late.

**Step 3: Implement strict config parsing**

Create validated configuration dataclasses or equivalent pure validators. Reject unknown keys. Check finite numeric bounds and cross-field rules. Do not introduce optional unconsumed governance fields.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_auto_iteration_config.py
```

Expected: all config mutations produce the expected verdict and data loading is not called on invalid config.

**Step 5: Commit**

```bash
git add config/auto_iteration_lab.json scripts/auto_iteration_lab.py tests/test_auto_iteration_config.py
git commit -m "fix: enforce nightly governance configuration"
```

## Task 6: Implement and gate walk-forward evidence

**Objective:** Make `walk_forward` executable and prevent full-history winners from being called champion-ready without it.

**Files:**
- Create: `Trader_Labs/scripts/walk_forward.py`
- Create: `Trader_Labs/tests/test_walk_forward.py`
- Modify: `Trader_Labs/scripts/auto_iteration_lab.py`
- Modify: `Trader_Labs/tests/test_auto_iteration_consistency.py`

**Step 1: Write failing fold/verdict tests**

Cover expanding train ranges whose train end is the session before test start; half-open `[test_start, test_end)` intervals; `step_months >= test_months`; minimum train years/test days/folds; truncation at each test end; cash-only fold initialization from a last-training-close intent with first valid test-open fill; holdings carried within but never between folds; final-fold close marking; stitched returns including cash days; worst-fold return; stitched Sharpe/drawdown; baseline-outperformance fraction over all valid folds; positive-best/positive-median fold ratio including zero/no-positive handling; and strict all-positive behavior. Add a candidate that wins full history but fails one fold; assert `evidence_label == "retrospective_only"` and `champion_ready is False`.

**Step 2: Verify RED**

```bash
pytest -q tests/test_walk_forward.py tests/test_auto_iteration_consistency.py -k "walk_forward or retrospective or champion_ready"
```

Expected: no runtime walk-forward verdict exists.

**Step 3: Implement fold engine and readiness gate**

Build folds from config with the exact boundaries above. Candidates are fixed: train history is causal indicator warm-up/minimum-history proof, not refitting or selection. Compute `baseline_outperformance = candidate_return - baseline_return`; gate on the fraction of all valid folds where it is non-negative. Compute `best_positive_fold / median_positive_fold`, treating absent/non-positive medians as infinity/failure. Run the same stateful simulator, stitch test-only NAV returns, and calculate every configured gate. State explicitly in the report that the evidence is retrospective because candidate discovery used full history.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_walk_forward.py tests/test_auto_iteration_consistency.py
python3 -m py_compile scripts/walk_forward.py scripts/auto_iteration_lab.py
```

Expected: all fold and evidence-label tests pass.

**Step 5: Commit**

```bash
git add scripts/walk_forward.py scripts/auto_iteration_lab.py tests/test_walk_forward.py tests/test_auto_iteration_consistency.py
git commit -m "feat: gate research champions on walk-forward evidence"
```

## Task 7: Canonicalize parameter identity and compare the incumbent

**Objective:** Remove labels from identity, measure novelty honestly, and retain the better ready strategy on the same snapshot.

**Files:**
- Create: `Trader_Labs/scripts/lab_schema.py`
- Create: `Trader_Labs/tests/test_lab_schema.py`
- Modify: `Trader_Labs/scripts/auto_iteration_lab.py`
- Modify: `Trader_Labs/tests/test_auto_iteration_consistency.py`

**Step 1: Write failing identity and selection tests**

Assert:

- baseline, `champion_baseline`, and a mutation label with identical strategy values share one SHA-256 fingerprint;
- omitted defaults and explicit defaults share a fingerprint;
- unknown strategy keys fail validation;
- list/numeric/boolean normalization is stable;
- within-run duplicate configs are removed;
- report counts distinguish novel configs, baseline/incumbent intentional retests, and accidental retests;
- a weaker challenger cannot replace a stronger re-evaluated incumbent;
- a stale/unready incumbent cannot remain `champion_ready`.

**Step 2: Verify RED**

```bash
pytest -q tests/test_lab_schema.py tests/test_auto_iteration_consistency.py -k "fingerprint or novelty or incumbent"
```

Expected: label-contaminated history identity and baseline-only champion comparison fail.

**Step 3: Implement one canonical path**

Centralize schema version, defaults, allowed keys, canonical JSON, and fingerprint. Use it in grid construction, history, result rows, signal metadata, and champion lookup. Re-evaluate incumbent and compare ready candidates by current-schema selection score on the same data snapshot.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_lab_schema.py tests/test_auto_iteration_consistency.py
```

Expected: identity, novelty, and incumbent tests pass.

**Step 5: Commit**

```bash
git add scripts/lab_schema.py scripts/auto_iteration_lab.py tests/test_lab_schema.py tests/test_auto_iteration_consistency.py
git commit -m "fix: canonicalize candidates and retain valid incumbent"
```

## Task 8: Add schema-break archive and atomic persistence

**Objective:** Preserve legacy evidence without allowing it into corrected rankings, and make writes crash-safe.

**Files:**
- Create: `Trader_Labs/scripts/atomic_io.py`
- Create: `Trader_Labs/tests/test_atomic_io.py`
- Create: `Trader_Labs/tests/test_history_migration.py`
- Modify: `Trader_Labs/scripts/auto_iteration_lab.py`

**Step 1: Write failing tests**

Using temporary report roots, assert:

- legacy history is atomically moved once to `reports/archive/auto_iteration_history.<timestamp>.legacy.jsonl`;
- the new history contains only current-schema rows;
- legacy rows never seed champion/dedup;
- malformed new JSONL fails with exact line number and leaves files unchanged;
- simulated write/fsync/replace failures preserve prior complete latest/history files;
- latest report and history rows carry matching schema/run IDs.

**Step 2: Verify RED**

```bash
pytest -q tests/test_atomic_io.py tests/test_history_migration.py
```

Expected: current append/write-text and silent malformed-line skipping violate tests.

**Step 3: Implement atomic helpers and migration**

Write same-directory temp files, flush/fsync, `os.replace`, and fsync the parent directory where supported. Serialize a complete next JSONL under the run lock. Archive rather than convert legacy history. Make malformed current history fatal and visible.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_atomic_io.py tests/test_history_migration.py
python3 -m py_compile scripts/atomic_io.py scripts/auto_iteration_lab.py
```

Expected: persistence and migration tests pass.

**Step 5: Commit**

```bash
git add scripts/atomic_io.py scripts/auto_iteration_lab.py tests/test_atomic_io.py tests/test_history_migration.py
git commit -m "fix: version and atomically persist nightly history"
```

## Task 9: Make the nightly runner single-instance and process-tree safe

**Objective:** Replace PID check-then-start behavior with flock admission and reliable start/status/stop semantics.

**Files:**
- Modify: `Trader_Labs/scripts/run_auto_iteration_nightly.sh`
- Create: `Trader_Labs/scripts/run_lab_worker.py`
- Create: `Trader_Labs/tests/test_nightly_runner.py`

**Step 1: Write failing runner tests**

Launch a temporary fake worker through the Python supervisor and assert:

- two simultaneous starts admit exactly one worker;
- recorded identity belongs to the actual worker/process group;
- stop terminates worker and spawned child, waits, and clears state;
- stale/reused PID cannot report running;
- an old latest report cannot satisfy the new run;
- delivery is marked only after successful parsing/rendering and remains retryable after failure.
- the shell-generated run ID is exported as `AT_LAB_RUN_ID`, required by the lab, and appears identically in every history/result row and latest report;
- terminal status delivery clears stale admission metadata.

**Step 2: Verify RED**

```bash
pytest -q tests/test_nightly_runner.py
```

Expected: current wrapper PID, child survival, stale report, and delivery behavior fail.

**Step 3: Implement Linux flock, run-ID propagation, and process-group lifecycle**

On Linux, hold a nonblocking `/usr/bin/flock` FD for the worker lifetime. The shell creates one run ID and exports `AT_LAB_RUN_ID`; `auto_iteration_lab.py` rejects a missing ID and writes it into every history row and latest report. `run_lab_worker.py` starts the real lab with `subprocess.Popen(..., start_new_session=True)`, writes PID/PGID/command/run-ID/start-time metadata atomically, waits, and preserves the worker exit status. TERM the recorded group, wait with bounded polling, KILL only after timeout. Require matching command identity, PGID, run ID, and report freshness before successful status delivery, then clear terminal metadata.

macOS unit tests must not depend on absent `flock`/`setsid`: test `run_lab_worker.py` directly and inject a temporary lock-command shim into shell tests. Add a Linux smoke command using a container or host with util-linux to exercise real `/usr/bin/flock` before deployment; do not make that deployment-only smoke a mandatory Mac gate.

**Step 4: Verify GREEN and shell syntax**

```bash
pytest -q tests/test_nightly_runner.py
bash -n scripts/run_auto_iteration_nightly.sh
python3 -m py_compile scripts/run_lab_worker.py
```

Expected: runner tests pass; shell syntax exits 0. Do not invoke `start` against the real lab.

**Step 5: Commit**

```bash
git add scripts/run_auto_iteration_nightly.sh scripts/run_lab_worker.py tests/test_nightly_runner.py
git commit -m "fix: lock and stop the complete nightly process tree"
```

## Task 10: Publish a versioned target-weight paper signal

**Objective:** Make Auto_Trader shadow output the exact D-close strategy intent and modeled D+1-open diagnostics atomically.

**Files:**
- Create: `Auto_Trader/scripts/atomic_io.py`
- Create: `Auto_Trader/scripts/signal_schema.py`
- Create: `Auto_Trader/tests/fixtures/paper_signal_v2.json`
- Modify: `Auto_Trader/scripts/rsi_momentum_paper_shadow.py`
- Modify: `Auto_Trader/tests/test_rsi_momentum_paper_shadow_safety.py`

**Step 1: Write failing shadow-contract tests**

Assert output includes schema version, stable signal ID, canonical parameter fingerprint, signal date, expected next trading session when calendar data permits, normalized `target_weights`, target cash weight, weighting method/lookback, and real modeled opens when available. Verify inverse-vol weights use only closes through D. Verify a latest D-close paper signal remains actionable for a fresh-quote D+1 fill even when the completed D+1 bar/model open is not available; only the historical research simulator treats that signal as pending. Simulate replace failure and assert prior valid output survives.

**Step 2: Verify RED**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
pytest -q tests/test_rsi_momentum_paper_shadow_safety.py
```

Expected: current output lacks the versioned intent fields and writes non-atomically.

**Step 3: Implement pure signal construction and atomic publication**

Separate intent construction from historical report metrics. Validate finite positive normalized weights. Derive stable IDs from schema, canonical params, date, and weights. Attach real D+1 open diagnostics only when present; never substitute close. Remove the current optimistic shadow backtest from the active signal payload, or route it through the corrected stateful simulator; no legacy close-return/fixed-weight/name-count-cost metric may be published without `legacy_non_promotion_safe` labeling and schema isolation. Preserve `No real orders placed` boundary.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_rsi_momentum_paper_shadow_safety.py
python3 -m py_compile scripts/atomic_io.py scripts/signal_schema.py scripts/rsi_momentum_paper_shadow.py
```

Expected: shadow safety/contract tests pass.

**Step 5: Commit**

```bash
git add scripts/atomic_io.py scripts/signal_schema.py scripts/rsi_momentum_paper_shadow.py \
  tests/test_rsi_momentum_paper_shadow_safety.py tests/fixtures/paper_signal_v2.json
git commit -m "feat: publish versioned paper target weights"
```

## Task 11: Execute paper target weights from one fresh quote snapshot

**Objective:** Replace sell-all/equal-budget historical fills with atomic target-delta whole-share fills at actual fresh quotes.

**Files:**
- Modify: `Auto_Trader/scripts/rsi_momentum_paper_ledger.py`
- Modify: `Auto_Trader/tests/test_rsi_momentum_paper_ledger_safety.py`
- Create: `Auto_Trader/tests/test_rsi_momentum_paper_ledger_parity.py`
- Create: `Auto_Trader/tests/fixtures/live_prices_v1.json`

**Step 1: Write failing ledger tests**

Cover:

- inverse-vol target weights produce non-equal target notionals;
- unchanged holdings trade only the delta, not sell-all/buy-all;
- all held/target prices come from one fresh quote snapshot;
- `paper_quote_snapshot_v1` requires one snapshot ID, timezone-aware generation time, finite positive prices, timezone-aware per-symbol times, and the union of held/target symbols; fill freshness never uses the current after-close unlimited-age exception;
- stale/missing quote aborts without state/output/signal-consumption mutation;
- historical signal prices are never actual fills;
- delayed signals fill now at fresh quotes with actual timestamp and no backdating;
- replayed signal ID is idempotent;
- actual fees derive from actual notional;
- rounding retains cash and reports target/actual weight deviation;
- modeled open and side-adjusted slippage are logged when available;
- target buys obtain modeled opens from the signal when present; dropped-holding sells look up the same D+1 OHLC dataset; either side without a model open yields `null` slippage, not a substituted price;
- state revisions are authoritative: failures before state replace change nothing; failures after state replace/before output replace do not replay the signal; the next run regenerates output from state; readers reject output/state revision mismatch.

**Step 2: Verify RED**

```bash
pytest -q tests/test_rsi_momentum_paper_ledger_safety.py tests/test_rsi_momentum_paper_ledger_parity.py
```

Expected: current equal-weight sell-all, signal-date historical pricing, and non-versioned consumption fail.

**Step 3: Implement atomic target-delta execution**

Extend state with schema, monotonically increasing `state_revision`, and `last_consumed_signal_id`. Define and validate `paper_quote_snapshot_v1` before mutation. The external Mac quote producer documented in `PROJECT_MAP.md` must publish the union of held and target symbols under that schema before the ledger can rebalance; if changing that automation is not available in the implementation session, fail closed and record it as a deployment prerequisite rather than weakening coverage. Mark current NAV from one snapshot, compute whole-share targets from published weights, sell reductions, buy increases within cash after fees, and commit copy-on-success. Fill quotes always obey `RSI_LEDGER_LIVE_MAX_AGE_SEC`; the after-close same-day exception is MTM-only. Keep historical fallback for MTM only, never rebalance. Remove or isolate dead SuperTrend execution from the champion path without expanding scope.

Commit protocol: atomically replace authoritative state first, then atomically write the rebuildable latest-output projection with the same revision. If projection write fails, return non-zero. On the next run, detect missing/mismatched output, regenerate it from state before processing a new signal, and never replay `last_consumed_signal_id`. Add injected failures at every boundary.

**Step 4: Verify GREEN**

```bash
pytest -q tests/test_rsi_momentum_paper_ledger_safety.py tests/test_rsi_momentum_paper_ledger_parity.py
python3 -m py_compile scripts/rsi_momentum_paper_ledger.py
```

Expected: safety and parity tests pass.

**Step 5: Commit**

```bash
git add scripts/rsi_momentum_paper_ledger.py \
  tests/test_rsi_momentum_paper_ledger_safety.py tests/test_rsi_momentum_paper_ledger_parity.py \
  tests/fixtures/live_prices_v1.json
git commit -m "fix: fill paper target weights from fresh quotes"
```

## Task 12: Add cross-repo golden parity fixtures

**Objective:** Detect future drift in target intent without coupling repository imports at runtime.

**Files:**
- Create: `Trader_Labs/tests/fixtures/parity_ohlc.json`
- Create: `Trader_Labs/tests/fixtures/expected_signal_v2.json`
- Create: `Trader_Labs/tests/test_paper_signal_contract.py`
- Create: `Auto_Trader/tests/fixtures/parity_ohlc.json`
- Create: `Auto_Trader/tests/fixtures/expected_signal_v2.json`
- Create: `Auto_Trader/tests/test_paper_signal_contract.py`

**Step 1: Write failing contract tests**

Use identical tiny OHLC/instrument fixtures in both repos. Assert the same picks, target weights within explicit tolerance, signal date, modeled execution date/open, parameter fingerprint, and signal ID. Add fixture hash/version checks so one side cannot silently change alone.

**Step 2: Verify RED**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
pytest -q tests/test_paper_signal_contract.py
cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
pytest -q tests/test_paper_signal_contract.py
```

Expected: one or both contract adapters/fixtures are missing.

**Step 3: Add thin test adapters, not runtime cross-imports**

Call each repository's pure signal-intent function with its local fixture. Keep copied fixture JSON byte-identical and document its schema version in the test.

**Step 4: Verify GREEN and fixture equality**

```bash
cmp /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs/tests/fixtures/parity_ohlc.json \
    /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader/tests/fixtures/parity_ohlc.json
cmp /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs/tests/fixtures/expected_signal_v2.json \
    /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader/tests/fixtures/expected_signal_v2.json
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs && pytest -q tests/test_paper_signal_contract.py
cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader && pytest -q tests/test_paper_signal_contract.py
```

Expected: `cmp` exits 0 and both contract tests pass.

**Step 5: Commit per repository**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
git add tests/fixtures/parity_ohlc.json tests/fixtures/expected_signal_v2.json tests/test_paper_signal_contract.py
git commit -m "test: lock paper signal parity contract"

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
git add tests/fixtures/parity_ohlc.json tests/fixtures/expected_signal_v2.json tests/test_paper_signal_contract.py
git commit -m "test: lock research signal parity contract"
```

## Task 13: Update operator documentation and continuity

**Objective:** Replace legacy close/equal-weight/history claims with the implemented contract and preserve exact next actions.

**Files:**
- Modify: `Trader_Labs/README.md`
- Modify: `Trader_Labs/PROJECT_MAP.md`
- Modify: `Trader_Labs/PROMOTION.md`
- Modify: `Trader_Labs/.hermes.md` (if project policy keeps it untracked, update but do not force-add)
- Modify: `Auto_Trader/README.md`
- Modify: `Auto_Trader/PROJECT_MAP.md`
- Modify: `Auto_Trader/.hermes.md`

**Step 1: Write a documentation checklist before editing**

Checklist exact semantics: D close, D+1 model open, real OHLC fail-close, stateful drift/cash, actual-notional fees, inverse-vol targets, fresh actual paper quotes, slippage diagnostic, no retro fills, retrospective-only full-history winner, walk-forward `champion_ready`, legacy archive, atomic persistence, flock/process group, no live orders.

**Step 2: Update docs and remove contradictory language**

Search for `equal-weight`, `signal-date prices`, `append-only`, `champion`, `month-end`, and old performance claims. Replace only statements affected by the implementation; do not claim new performance before a new-schema run.

**Step 3: Verify documentation references**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
rg -n "equal-weight|signal-date prices|append-only|champion_ready|retrospective_only|D\+1" README.md PROJECT_MAP.md PROMOTION.md .hermes.md docs

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
rg -n "equal-weight|signal-date prices|month-end|target_weights|retroactive|live orders" README.md PROJECT_MAP.md .hermes.md
```

Expected: no contradictory active-runtime claims; new contract terms are discoverable.

**Step 4: Commit per repository**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
git add README.md PROJECT_MAP.md PROMOTION.md
git commit -m "docs: describe corrected nightly evidence contract"

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
git add README.md PROJECT_MAP.md .hermes.md
git commit -m "docs: describe target-weight paper execution"
```

## Task 14: Run the complete local release-candidate gate

**Objective:** Prove both repositories are internally green without mutating production state.

**Files:** No source changes expected; if a check requires a fix, return to RED/GREEN in the owning task and rerun this entire gate.

**Step 1: Run Trader_Labs full gate**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
set -euo pipefail
pytest -q
python3 -m py_compile scripts/*.py
auto_files="$(git ls-files '*.py')"
python3 -m compileall -q $auto_files
bash -n scripts/run_auto_iteration_nightly.sh
git diff --check
git status --short
```

Expected: all tests pass, syntax checks exit 0, no whitespace errors. Do not run `scripts/run_auto_iteration_nightly.sh start`.

**Step 2: Run Auto_Trader full gate**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
set -euo pipefail
pytest -q
python3 -m py_compile scripts/*.py
git diff --check
git status --short
```

Expected: all tests pass, syntax checks exit 0, no whitespace errors. Do not run `rsi_momentum_paper_shadow.py` or `rsi_momentum_paper_ledger.py` against real report paths.

**Step 3: Review contracts and prohibition mechanically**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading
rg -n "place_order|orders\.place|KiteConnect|auto_trade\.service" Trader_Labs/scripts Auto_Trader/scripts
```

Expected: no new live-order path. Any legacy hit must be reviewed and documented, not waved through.

**Step 4: Review final diffs**

```bash
cd /Users/sahilgoel/Desktop/Projects/trading/Trader_Labs
git log --oneline --decorate -15
git status --short

cd /Users/sahilgoel/Desktop/Projects/trading/Auto_Trader
git log --oneline --decorate -15
git status --short
```

Expected: only intended implementation/docs changes, all committed. Do not push or deploy.

## Acceptance criteria

- Research signals use D close and model execution only at a valid D+1 open.
- Missing required OHLC aborts a rebalance before mutation.
- Portfolio units drift; fees and turnover use actual modeled notional; cash is explicit.
- Signal target weights preserve configured inverse-volatility weighting.
- Paper fills use one fresh live quote snapshot, size toward published weights, charge actual-notional fees, retain cash, and are idempotent.
- Delayed paper execution is timestamped now; no retroactive fill is created. Modeled open/slippage is diagnostic only.
- Every walk-forward configuration key gates a runtime verdict.
- Full-history winners are `retrospective_only`; only candidates passing all configured consistency and walk-forward gates can be `champion_ready`.
- Incumbent and challenger are re-evaluated on the same snapshot and the weaker ready result cannot replace the stronger one.
- Canonical parameter fingerprints exclude labels and normalize defaults.
- Legacy history is archived, not converted; only new-schema history participates in selection.
- Latest/history/shadow/state/output writes are atomic; malformed current history is visible.
- `flock` admits one nightly run and stop terminates the complete worker tree.
- Both full suites and syntax gates pass.
- No live orders, push, deployment, cron changes, or service actions occur.
