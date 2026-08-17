"""
Compute beta for a given stock using Alpaca's historical market data.

Alpaca does not expose a "beta" field directly — beta is a derived
statistic, so this script pulls historical daily bars for your target
symbol and a benchmark (default: SPY), computes daily returns, and
calculates beta = Cov(stock_returns, benchmark_returns) / Var(benchmark_returns).

Requirements:
    pip install alpaca-py pandas numpy

Setup (local):
    Set your Alpaca API credentials as environment variables:
        export ALPACA_API_KEY="your_key_id"
        export ALPACA_SECRET_KEY="your_secret_key"

    (Paper trading keys work fine here — this only pulls market data,
    it does not place any orders.)

Setup (GitHub Actions):
    This script reads credentials from the environment, so in a workflow
    you just map repo/org Secrets into env vars for the step that runs it:

        - name: Compute beta
          env:
            ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
            ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
            SYMBOL: ${{ vars.SYMBOL }}          # repo/org "Variables", not secret
            BENCHMARK: ${{ vars.BENCHMARK }}
          run: python alpaca_beta.py "$SYMBOL" --benchmark "$BENCHMARK"

    Never put the actual key values in the workflow file or in repo
    "Variables" — only in Settings > Secrets and variables > Actions > Secrets.
    See sample_workflow.yml for a full working example.

Usage:
    python alpaca_beta.py AAPL
    python alpaca_beta.py AAPL --benchmark QQQ --lookback 252

    Symbol/benchmark can also come from env vars SYMBOL / BENCHMARK if you
    don't pass them as CLI args (useful for workflow_dispatch inputs).
"""

import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def get_daily_returns(client: StockHistoricalDataClient, symbol: str, start: datetime, end: datetime) -> pd.Series:
    """Fetch daily bars for a symbol and return a series of daily pct returns."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request).df

    # bars is a MultiIndex DataFrame (symbol, timestamp) when using the alpaca-py SDK
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)

    closes = bars["close"]
    returns = closes.pct_change().dropna()
    return returns


def compute_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Align two return series and compute beta via covariance / variance."""
    df = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner")
    df.columns = ["stock", "benchmark"]

    covariance = df["stock"].cov(df["benchmark"])
    variance = df["benchmark"].var()

    if variance == 0:
        raise ValueError("Benchmark variance is zero — cannot compute beta.")

    return covariance / variance


def main():
    parser = argparse.ArgumentParser(description="Compute beta for a stock using Alpaca historical data.")
    parser.add_argument("symbol", nargs="?", default=os.environ.get("SYMBOL"),
                         help="Ticker symbol to compute beta for, e.g. AAPL (falls back to SYMBOL env var)")
    parser.add_argument("--benchmark", default=os.environ.get("BENCHMARK", "SPY"),
                         help="Benchmark symbol (falls back to BENCHMARK env var, default: SPY)")
    parser.add_argument("--lookback", type=int, default=int(os.environ.get("LOOKBACK", 252)),
                         help="Number of trading days to look back (default: 252, ~1 year)")
    args = parser.parse_args()

    if not args.symbol:
        raise SystemExit(
            "No symbol provided. Pass it as an argument (e.g. `python alpaca_beta.py AAPL`) "
            "or set the SYMBOL environment variable."
        )

    # Credentials must come from Secrets (never from repo Variables or the
    # workflow file itself) — GitHub masks secret values in logs automatically.
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise SystemExit(
            "Missing credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
            "(locally via `export`, or in GitHub Actions via secrets.ALPACA_API_KEY / secrets.ALPACA_SECRET_KEY)."
        )

    client = StockHistoricalDataClient(api_key, secret_key)

    # Pad calendar days to comfortably cover the requested trading-day lookback
    end = datetime.now()
    start = end - timedelta(days=int(args.lookback * 1.6) + 10)

    print(f"Fetching {args.lookback} trading days of data for {args.symbol} and {args.benchmark}...")

    stock_returns = get_daily_returns(client, args.symbol, start, end).tail(args.lookback)
    benchmark_returns = get_daily_returns(client, args.benchmark, start, end).tail(args.lookback)

    beta = compute_beta(stock_returns, benchmark_returns)

    print(f"\nSymbol:      {args.symbol}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Lookback:    {len(stock_returns)} trading days")
    print(f"Beta:        {beta:.4f}")


if __name__ == "__main__":
    main()