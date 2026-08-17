"""
Vercel serverless function: GET /api/beta?symbol=AAPL[&benchmark=SPY&lookback=252]

Wraps the beta calculation from alpaca_beta.py in an HTTP handler using
Vercel's Python runtime (BaseHTTPRequestHandler convention).

Env vars required (set in Vercel Project Settings > Environment Variables):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os
import json
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def get_daily_returns(client, symbol: str, start: datetime, end: datetime) -> pd.Series:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)

    closes = bars["close"]
    return closes.pct_change().dropna()


def compute_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner")
    df.columns = ["stock", "benchmark"]

    covariance = df["stock"].cov(df["benchmark"])
    variance = df["benchmark"].var()

    if variance == 0:
        raise ValueError("Benchmark variance is zero — cannot compute beta.")

    return covariance / variance


def calculate(symbol: str, benchmark: str, lookback: int) -> dict:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.")

    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now()
    start = end - timedelta(days=int(lookback * 1.6) + 10)

    stock_returns = get_daily_returns(client, symbol, start, end).tail(lookback)
    benchmark_returns = get_daily_returns(client, benchmark, start, end).tail(lookback)

    beta = compute_beta(stock_returns, benchmark_returns)

    return {
        "symbol": symbol,
        "benchmark": benchmark,
        "lookback_days": len(stock_returns),
        "beta": round(float(beta), 4),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)

        symbol = query.get("symbol", [None])[0]
        benchmark = query.get("benchmark", ["SPY"])[0]
        lookback = int(query.get("lookback", [252])[0])

        if not symbol:
            self._send_json({"error": "Missing required query param: symbol"}, status=400)
            return

        try:
            result = calculate(symbol.upper(), benchmark.upper(), lookback)
            self._send_json(result, status=200)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)