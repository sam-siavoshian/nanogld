"""Alpaca historical 30min ETF basket — Source 8 (V1 expansion 2026-05-04).

Same SDK + timeframe as Source 1; multi-symbol single batched call. Captures
risk-on/off (SPY/QQQ/IWM), gold cross-correlations (GDX miners, SLV silver),
and sector regime (XLF/XLE/XLK/XLU).

Spec hard rules:
- Multi-symbol pagination INTERLEAVES (V4 §12) — drain all pages first.
- Per-ETF parquet on disk: data/raw/alpaca_bars_<SYM>_30min.parquet.
- Free tier rate limit 200 req/min applies across symbols, not per-symbol.
- IEX-only on free tier — GDX/SLV may have low-volume gap bars; resilience same as GLD.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from nanogld.data.alpaca_bars import fetch_bars
from nanogld.data.schema import BARS_MANIFEST, validate
from nanogld.data.utils import END_DATE_NAIVE, START_DATE_NAIVE, get_logger, raw_dir

LOG = get_logger("nanogld.data.alpaca_etfs")

ETF_BASKET: dict[str, str] = {
    # Broad equity (risk-on/off)
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000 (small caps)",
    # Gold-specific cross-references
    "GDX": "VanEck Gold Miners (direct gold cross-correlation)",
    "SLV": "iShares Silver Trust (gold-silver ratio numerator)",
    # Sector ETFs (factor regime)
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLU": "Utilities (rate-sensitive defensive)",
}


def fetch_etf_basket(
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    syms = list(symbols) if symbols else list(ETF_BASKET.keys())
    return fetch_bars(syms, start=start, end=end)


def write_etf_parquets(
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
) -> tuple[pd.DataFrame, list[str]]:
    df = fetch_etf_basket(start=start, end=end)
    if df.empty:
        return df, []
    validate(df, BARS_MANIFEST)

    paths: list[str] = []
    for sym, sub in df.groupby("symbol"):
        out = raw_dir() / f"alpaca_bars_{sym}_30min.parquet"
        sub.to_parquet(out, compression="zstd", index=False)
        paths.append(str(out))
        LOG.info("wrote %d %s bars -> %s", len(sub), sym, out)
    return df, paths


if __name__ == "__main__":
    df, paths = write_etf_parquets()
    print(
        f"ETF basket: {len(df)} rows across {df['symbol'].nunique()} symbols -> {len(paths)} files"
    )
