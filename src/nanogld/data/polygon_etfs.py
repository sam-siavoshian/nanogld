"""Polygon / Massive ETF basket — drop-in replacement for alpaca_etfs.

Same 9-ETF basket as Source 8 (V1 expansion). Wraps polygon_bars.fetch_bars.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from nanogld.data.polygon_bars import fetch_bars
from nanogld.data.schema import BARS_MANIFEST, validate
from nanogld.data.utils import END_DATE_NAIVE, START_DATE_NAIVE, get_logger, raw_dir

LOG = get_logger("nanogld.data.polygon_etfs")

ETF_BASKET: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000 (small caps)",
    "GDX": "VanEck Gold Miners",
    "SLV": "iShares Silver Trust",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLU": "Utilities",
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
        out = raw_dir() / f"polygon_bars_{sym}_30min.parquet"
        sub.to_parquet(out, compression="zstd", index=False)
        paths.append(str(out))
        LOG.info("wrote %d %s bars -> %s", len(sub), sym, out)
    return df, paths


if __name__ == "__main__":
    df, paths = write_etf_parquets()
    sym_count = df["symbol"].nunique() if not df.empty else 0
    print(f"Polygon ETFs: {len(df)} rows / {sym_count} symbols / {len(paths)} files")
