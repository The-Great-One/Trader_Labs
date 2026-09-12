from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stonkfly.actions import StonkflyActions
from stonkfly.broker import PaperBroker
from stonkfly.config import D, Settings
from stonkfly.display import market_frame
from stonkfly.ledger import Ledger
from stonkfly.market import Quote
from stonkfly.neural.controller import FlyController
from stonkfly.reinforcement import reinforcement
from stonkfly.risk import Guard


CSV = Path(os.environ.get('STONKFLY_NSE_CSV', '/tmp/nse_clean_ohlcv.csv'))


class ReplayMarket:
    def __init__(self, symbol: str, closes: list[tuple[int, float]], lookback: int):
        self.symbol = symbol
        self.product = 'BTC-USDC'  # Stonkfly's internal allowlisted paper symbol.
        self.rows = closes
        self.index = lookback
        self.history = [price for _, price in closes[:lookback]]

    def snapshot(self):
        ts, price = self.rows[self.index]
        mid = D(price)
        self.current_timestamp = float(ts)
        return {
            self.product: Quote(
                self.product, mid * D('0.9995'), mid * D('1.0005'), float(ts),
                D('0.00000001'), D('0.01'), D('0.01'), D('1'), D('0.00000001')
            )
        }

    def record(self, quotes):
        self.history.append(float((quotes[self.product].bid + quotes[self.product].ask) / 2))
        self.index += 1


class ReplayGuard(Guard):
    def __init__(self, settings, ledger, stop_file):
        super().__init__(settings, ledger, stop_file)
        self.now = 0.0

    def plan(self, product, side, quotes, now=None):
        return super().plan(product, side, quotes, now=self.now)

    def before_submit(self, plan):
        if self.stop_file.exists() or self.l.get('halted'):
            raise RuntimeError('Execution stopped')
        if not -0.5 <= self.now - plan['quote_timestamp'] <= self.s.max_quote_age:
            raise RuntimeError('Historical quote timing mismatch')
        pending = self.l.pending()
        if len(pending) != 1 or pending[0]['id'] != plan['client_order_id'] or pending[0]['status'] != 'PREPARED':
            raise RuntimeError('Intent ownership mismatch')


def load_path(symbol: str, days: int, lookback: int = 120):
    df = pd.read_csv(CSV, usecols=['symbol', 'timestamps', 'close'])
    df = df[df['symbol'].astype(str).str.upper() == symbol.upper()].copy()
    if df.empty:
        raise SystemExit(f'No NSE data for {symbol}')
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').dropna(subset=['close'])
    if len(df) < lookback + days:
        raise SystemExit(f'{symbol}: only {len(df)} rows; need {lookback + days}')
    df = df.tail(lookback + days)
    base = float(df.iloc[lookback]['close'])
    rows = [(int(ts.timestamp()), float(close) / base * 100.0) for ts, close in zip(df['timestamps'], df['close'])]
    return rows, df


def replay(symbol: str, days: int, out: Path, frozen: bool = False, shuffle_reward: bool = False):
    rows, raw = load_path(symbol, days)
    settings = Settings(products=('BTC-USDC',), neural_ms=100, pulse_ms=100, learning=not frozen, daily_orders=100)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / 'ledger.sqlite', settings, 'paper')
    broker = PaperBroker(settings, ledger)
    market = ReplayMarket(symbol, rows, 120)
    controller = FlyController(settings)
    guard = ReplayGuard(settings, ledger, out / 'STOP')
    actions = StonkflyActions(guard, broker)
    events = []
    try:
        for _ in range(len(rows) - 120):
            quotes = market.snapshot()
            q = quotes['BTC-USDC']
            guard.now = q.timestamp
            ledger.put('last_attempt', 0)
            equity_before = ledger.equity(quotes)
            kind, _delta = reinforcement(equity_before, ledger.get('anchor'), settings.reward_deadband)
            if shuffle_reward:
                kind = random.Random(7 + len(events)).choice(['none', 'reward', 'aversive'])
            frame = market_frame('BTC-USDC', market.history, q.bid, q.ask)
            neural = controller.observe(frame, kind)
            ledger.commit_tick(equity_before, {'file': 'nse-replay', 'sha256': symbol}, {'market_history': {'BTC-USDC': market.history}})
            order = {'status': 'HOLD'}
            if neural['side'] != 'HOLD':
                actions.quotes = quotes
                try:
                    order = actions.invoke({'product': 'BTC-USDC', 'side': neural['side']})
                except Exception as exc:
                    order = {'status': 'VETO', 'reason': type(exc).__name__}
            equity_after = ledger.equity(quotes)
            events.append({
                'timestamp': q.timestamp, 'price': str((q.bid + q.ask) / 2),
                'side': neural['side'], 'execution': order['status'],
                'equity': str(equity_after), 'stimulus': kind,
            })
            market.record(quotes)
        final_equity = ledger.equity(quotes)
        baseline = rows[-1][1] / rows[120][1] * 100
        summary = {
            'symbol': symbol.upper(), 'mode': 'frozen' if frozen else 'shuffled_reward' if shuffle_reward else 'learning', 'nse_rows': len(raw), 'evaluated_sessions': len(events),
            'start': str(raw.iloc[120]['timestamps'].date()), 'end': str(raw.iloc[-1]['timestamps'].date()),
            'stonkfly_final_equity': str(final_equity),
            'stonkfly_return_pct': float((final_equity / D('100') - 1) * 100),
            'buy_hold_return_pct': float(baseline - 100),
            'buy_proposals': sum(e['side'] == 'BUY' for e in events),
            'sell_proposals': sum(e['side'] == 'SELL' for e in events),
            'hold_proposals': sum(e['side'] == 'HOLD' for e in events),
            'filled': sum(e['execution'] == 'FILLED' for e in events),
            'vetoed': sum(e['execution'] == 'VETO' for e in events),
            'model_note': 'NSE close path normalized to a paper ₹100 base; Stonkfly internal BTC-USDC label is not an NSE trading recommendation.',
        }
        (out / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    finally:
        ledger.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='HDFCBANK')
    ap.add_argument('--days', type=int, default=20)
    ap.add_argument('--out', type=Path, default=Path('runs/nse-hdfcbank-20d'))
    ap.add_argument('--frozen', action='store_true')
    ap.add_argument('--shuffle-reward', action='store_true')
    args = ap.parse_args()
    if args.frozen and args.shuffle_reward:
        ap.error('--frozen and --shuffle-reward are mutually exclusive')
    replay(args.symbol, args.days, args.out, frozen=args.frozen, shuffle_reward=args.shuffle_reward)
