
# Backtest on Real Historical Data (Yahoo Finance)

Fetch real prices from Yahoo Finance via `yfinance` and backtest an **SMA crossover** strategy.
Falls back to **synthetic data** if download fails, so you can run it offline too.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python backtest_real_data.py --ticker AAPL --start 2020-01-01 --end 2024-12-31
```

Change parameters:
```bash
python backtest_real_data.py --ticker MSFT --fast 10 --slow 40 --fee_bps 2 --out_dir msft_outputs
```

## Outputs
- `metrics.json`, `trade_log.csv`, `price_signals.png`, `equity_curve.png`
