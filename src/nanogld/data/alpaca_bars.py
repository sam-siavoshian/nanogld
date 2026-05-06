"""Alpaca historical 30min GLD bars — Source 1.

Spec hard rules (plan/02-DATA-PIPELINE.md "Source 1"):
- TimeFrame(30, TimeFrameUnit.Minute) — NOT TimeFrame.Minute_30 (does not exist).
- adjustment='all' (split + dividend adjusted).
- feed='sip' (paper free + SIP — verified 2026-05-05: returns full 10y of
  30min bars 2016-01-04+. IEX-only is more restricted on paper paid tier;
  SIP works on paper free with longer history).
- limit=None — SDK auto-paginates via next_page_token; drain all pages.
- Bar timestamp is bar START. t_visible = timestamp + 30min (bar END).
- Free tier 200 req/min, IEX-only ~2.5% of US volume — expect occasional gaps.
- 5y window: 2021-04-24 to 2026-04-24 (matches START_DATE_UTC / END_DATE_UTC).
- Latest 15min of intraday is gated; ok for backtest, irrelevant for historical pull.

Owner runs after .env.paper is filled. Keys read from environment via
python-dotenv (loaded by cli.py).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from nanogld.data.schema import BARS_MANIFEST, validate
from nanogld.data.utils import END_DATE_NAIVE, START_DATE_NAIVE, get_logger, raw_dir

LOG = get_logger("nanogld.data.alpaca_bars")

DEFAULT_SYMBOL = "GLD"
DEFAULT_TIMEFRAME = TimeFrame(30, TimeFrameUnit.Minute)
BAR_DURATION = pd.Timedelta(minutes=30)


def _client() -> StockHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_API_SECRET")
    if not key or not sec or "FILL_ME" in str(key):
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_API_SECRET missing — populate ~/.config/nanogld/.env.paper"
        )
    return StockHistoricalDataClient(key, sec)


def fetch_bars(
    symbols: str | Iterable[str] = DEFAULT_SYMBOL,
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
    feed: str = "sip",
) -> pd.DataFrame:
    """Pull 30min bars for one or many symbols. Returns long tidy frame.

    SDK auto-paginates so a single call drains all pages even for the 5y
    multi-symbol case. Multi-symbol response is symbol-then-timestamp ordered
    (V4 hard rule §12 — pages interleave; we drain all before constructing).
    """
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    LOG.info("fetching Alpaca bars: %s, %s -> %s, feed=%s", syms, start, end, feed)
    req = StockBarsRequest(
        symbol_or_symbols=syms,
        timeframe=DEFAULT_TIMEFRAME,
        start=start,
        end=end,
        adjustment="all",
        feed=feed,
        limit=None,
    )
    bars = _client().get_stock_bars(req)
    df = bars.df.reset_index()
    if df.empty:
        LOG.warning("Alpaca returned 0 rows for %s", syms)
        return pd.DataFrame()

    # SDK returns columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
    df = df.rename(columns={"timestamp": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype("string")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("trade_count", "vwap"):
        df[c] = (
            pd.to_numeric(df[c], errors="coerce").astype("float64") if c in df.columns else pd.NA
        )

    # Bar visibility = bar END (V4 hard rule §1)
    df["release_ts"] = df["timestamp"] + BAR_DURATION
    df["t_visible"] = df["release_ts"]
    return (
        df[[c.name for c in BARS_MANIFEST.columns]]
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def write_gld_parquet(
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
) -> tuple[pd.DataFrame, str]:
    df = fetch_bars(DEFAULT_SYMBOL, start=start, end=end)
    if df.empty:
        return df, ""
    validate(df, BARS_MANIFEST)
    out_path = raw_dir() / f"alpaca_bars_{DEFAULT_SYMBOL}_30min.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d %s bars -> %s", len(df), DEFAULT_SYMBOL, out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_gld_parquet()
    print(f"GLD bars: {len(df)} rows -> {p}")
