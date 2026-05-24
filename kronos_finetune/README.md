# Kronos fine-tuning for Kite/NSE data

This folder contains the Auto_Trader/Trader_Labs fine-tuning workflow for Kronos.

## 1. Export Kite feather history

```bash
source ../Stocks/venv/bin/activate
python scripts/export_kite_to_kronos_csv.py \
  --output-dir kronos_finetune/data \
  --output-name kite_nse_daily_kronos.csv
```

For a quick smoke test:

```bash
python scripts/export_kite_to_kronos_csv.py --limit 5 --min-rows 200 \
  --output-dir /tmp/kronos_export_smoke
```

Output columns:

- `symbol` audit metadata
- `timestamps`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`
- `split` audit metadata using chronological train/val/test allocation

Kronos expects `timestamps/open/high/low/close/volume/amount`. Extra columns are kept for auditability.

## 2. Fine-tune Kronos-small

Install Kronos dependencies, then run from the Kronos CSV fine-tune folder:

```bash
cd external/Kronos/finetune_csv
python train_sequential.py --config ../../kronos_finetune/configs/kite_nse_daily_kronos_small.yaml
```

The first pass intentionally uses small CPU-friendly defaults. Increase epochs/batch/device only after the end-to-end path works.

## 3. Evaluate

After a checkpoint is produced, run the existing Kronos pilot with the fine-tuned model/tokenizer paths and compare:

- baseline RS7/RS2
- generic Kronos
- fine-tuned Kronos
- ranking overlay
- sizing/risk overlay
- exit overlay

Promotion requires out-of-sample improvement, not in-sample loss reduction.
