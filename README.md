<div align="center">

<img src="https://via.placeholder.com/120x120.png?text=TL" alt="Trader_Labs logo" width="120" height="120" />

# Trader_Labs

**Nightly RSI-momentum auto-iteration research lab. Does not place orders.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/The-Great-One/Trader_Labs)](https://github.com/The-Great-One/Trader_Labs/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/The-Great-One/Trader_Labs)](https://github.com/The-Great-One/Trader_Labs/pulls)
[![GitHub stars](https://img.shields.io/github/stars/The-Great-One/Trader_Labs?style=social)](https://github.com/The-Great-One/Trader_Labs/stargazers)

[Report Bug](https://github.com/The-Great-One/Trader_Labs/issues/new?template=bug_report.md) · [Request Feature](https://github.com/The-Great-One/Trader_Labs/issues/new?template=feature_request.md)

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About

`Trader_Labs` is the focused research deployment for the nightly RSI-momentum
auto-iteration lab. It does not place orders. Strategy promotion requires
reviewed evidence and a separate `Auto_Trader` change.

## Features

- ✅ Nightly auto-iteration of RSI-momentum strategies
- ✅ Walk-forward / OOS validation gates
- ✅ Stateful D-close/D+1-open accounting
- ✅ Corrected history (current-schema only, atomic rewrite)
- ✅ Process-group lifecycle via flock
- 🚧 Automatic live promotion (not enabled)

## Getting Started

### Prerequisites

- Python 3.10+
- Access to NSE data

### Installation

```bash
git clone https://github.com/The-Great-One/Trader_Labs.git
cd Trader_Labs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Live script closure

```text
scripts/auto_iteration_lab.py
└── scripts/rsi_224466_rotation_lab.py
```

`scripts/portfolio_simulator.py` owns stateful D-close/D+1-open accounting.
`scripts/walk_forward.py` owns fixed-candidate retrospective folds.
`scripts/run_auto_iteration_nightly.sh` uses flock plus `run_lab_worker.py` for
single admission and complete process-group lifecycle.

### Nightly auto-iteration

The Hermes nightly chain is:

```text
Hermes cron 7af4ce25de03
→ run_auto_iteration.sh
→ scripts/run_auto_iteration_nightly.sh start
→ scripts/auto_iteration_lab.py
```

The live baseline in `auto_iteration_lab.py` is RSI 22/44/66, momentum 63,
`3W-FRI`, top 8, SMA100 regime, MACD, momentum-rank blend 0.3, and
inverse-volatility weighting with a 10-day lookback. Keep it synchronized with
the deployed `Auto_Trader` paper-shadow parameters.

Qualification gates and scoring weights are read from
`config/auto_iteration_lab.json`. The lab writes:

- `reports/auto_iteration_latest.json`
- `reports/auto_iteration_history.jsonl`
- `reports/auto_iteration_runs/`

Corrected history is current-schema-only and atomically rewritten;
incompatible legacy rows are archived rather than ranked or deduplicated.
Full-history winners are `retrospective_only`. Only candidates passing every
consistency and walk-forward gate can be `champion_ready`, and even that remains
retrospective evidence rather than automatic live promotion.

Signals use D close and model fills only at a real D+1 open. Holdings and cash
persist and drift; fees and turnover use actual modeled notional. Missing
required opens or held-position marks fail the candidate closed.

### Lifecycle commands

```bash
cd /home/ubuntu/Trader_Labs
scripts/run_auto_iteration_nightly.sh start
scripts/run_auto_iteration_nightly.sh status
scripts/run_auto_iteration_nightly.sh stop
```

Do not start the nightly lab during routine verification.

### Runtime and promotion boundary

- Runtime host: `ubuntu@144.24.112.62` (`arunnet`)
- Deployment root: `/home/ubuntu/Trader_Labs`
- `Auto_Trader` dependency: `/home/ubuntu/Auto_Trader` (runtime venv only; no source imports)
- Generated `reports/` and market-data caches remain gitignored.
- Do not edit live trading rules from this repository.
- Require consistency qualification, walk-forward/OOS review, execution/data-parity checks, and a separate reviewed `Auto_Trader` patch before promotion.

## Roadmap

- [x] v0.1 – Nightly auto-iteration lab
- [ ] v0.2 – Champion promotion workflow
- [ ] v1.0 – Validated strategy promotion

See the open issues for the full list of proposed features and known bugs.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Sahil Goel – [@The-Great-One](https://github.com/The-Great-One) –
sahilgoel187@gmail.com

Project link: [https://github.com/The-Great-One/Trader_Labs](https://github.com/The-Great-One/Trader_Labs)

## Acknowledgments

- [Img Shields](https://shields.io)
- [Contributor Covenant](https://www.contributor-covenant.org)
