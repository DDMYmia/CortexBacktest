# Competitor comparison — 2026-11-05

CortexBacktest against fifteen other Python engines, one interpreter per engine,
one machine (CPython 3.11/3.13, arm64). `results.csv` and `report.md` come from
`benchmarks/real_data_comparison/driver.py`: 252 rows, 0 unexplained P&L
differences, 21 differences explained by engine price precision.

## Scenarios
- **SMA crossover**: SMA(20/50), 100 shares, on recent (34 x 563), Yahoo deep
  and wide shapes up to 596 x 2,932.
- **Multi-timeframe strategy** (`common.Mtf`): weekly regime, daily RSI /
  Bollinger / ATR, ATR-sized share counts; computed in each engine's callback.
- **Resting orders**: a GTC buy limit at 0.97 x close on every bar, 34 tickers.

## Data
- `recent.npz`: daily bars aggregated from a 1-minute store, 2024-01-02..2026-03-31,
  34 full-calendar tickers without a >40% one-day move (the store is unadjusted).
- `deep.npz` / `wide.npz`: Yahoo daily bars 2015–2026.

## Explained differences
- zipline-reloaded: its daily bundle stores prices at 0.001 (up to +$2.81).
- nautilus_trader: 2-decimal instrument price precision (up to −$124 on the
  multi-timeframe strategy, whose share counts amplify each rounded price).
- qlib: float32 price storage (up to +$0.03).

## Fixes found while running
- The strategy summed with the built-in `sum()`, which changed in Python 3.12;
  engines on 3.11 computed different targets. It now uses `math.fsum`, and both
  interpreters produce byte-identical targets.
- pyalgotrade caps fills at 25% of bar volume by default; the harness lifts the
  cap, as no other engine or the ledger applies one.
- pybroker rounds its reported results to cents by default; turned off.
- freqtrade rejects orders below exchange minimums and closes trades still open
  at the end at the last open: the runner uses pairs with a 1-unit minimum and
  re-marks those exits at the last close.

## Not covered
LEAN (C# through Docker), China-market engines (abu, rqalpha, wondertrader,
hikyuu), hummingbot and FinRL (not bar backtesters). Costs, intraday bars and
corporate actions are not part of these scenarios.
