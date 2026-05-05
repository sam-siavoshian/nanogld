"""Polygon / Massive 30min GLD bars — drop-in replacement for alpaca_bars.

Spec: replaces Source 1 of plan/02-DATA-PIPELINE.md after the Alpaca KYC
detour. Same NEWS_MANIFEST-adjacent BARS_MANIFEST schema (release_ts =
timestamp + 30min, t_visible = release_ts).

Polygon free tier (Stocks Basic):
- 5 req/min (use polygon-api-client iterator; SDK paginates)
- ~2-year rolling history on free; 5y on paid Starter+ (verify per ticker)
- Adjusted prices: Polygon adjusts splits + dividends by default

Hard rules (V1):
- Bar `t` from Polygon is bar START in ms epoch (UTC). t_visible = t + 30min.
- adjusted=True (split + dividend); for forward-only adjustment, use as_of
  parameter when shipping live cycle (doc 08).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import requests

from nanogld.data.schema import BARS_MANIFEST, validate
from nanogld.data.utils import (
    END_DATE_NAIVE,
    START_DATE_NAIVE,
    UTC,
    get_logger,
    raw_dir,
)

LOG = get_logger("nanogld.data.polygon_bars")

DEFAULT_SYMBOL = "GLD"
BAR_DURATION = pd.Timedelta(minutes=30)
POLYGON_BASE = "https://api.polygon.io"
# Free tier = 5 req/min. Sleep 12.5s between calls to stay safely under.
POLITE_SEC_FREE = 13.0


def _api_key() -> str:
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError(
            "POLYGON_API_KEY missing — populate ~/.config/nanogld/.env.paper after "
            "signing up at https://massive.com/dashboard/signup."
        )
    return api_key


def _paginated_aggs(
    *,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_str: str,
    to_str: str,
    polite_sec: float = POLITE_SEC_FREE,
) -> list[dict]:
    """Manually paginate /v2/aggs to respect Polygon free 5 req/min.

    SDK iterator does NOT throttle and trips 429s mid-pull. We manage it
    here with a fixed sleep between page fetches.
    """
    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": _api_key()}
    aggs: list[dict] = []
    pages = 0
    while True:
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code == 429:
            LOG.warning("[polygon] 429 on %s — sleep 30s + retry", ticker)
            time.sleep(30)
            continue
        if resp.status_code != 200:
            LOG.warning("[polygon] %s %d on %s — abort ticker", ticker, resp.status_code, url[:120])
            return aggs
        data = resp.json()
        results = data.get("results") or []
        aggs.extend(results)
        pages += 1
        next_url = data.get("next_url")
        if not next_url:
            break
        # Polygon's next_url already encodes cursor; just append apiKey.
        url = next_url
        params = {"apiKey": _api_key()}
        time.sleep(polite_sec)
    LOG.info("[polygon] %s: %d bars across %d pages", ticker, len(aggs), pages)
    return aggs


def fetch_bars(
    symbols: str | Iterable[str] = DEFAULT_SYMBOL,
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
    multiplier: int = 30,
    timespan: str = "minute",
) -> pd.DataFrame:
    """Pull 30min bars for one or many symbols. Returns long tidy frame."""
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    LOG.info("Polygon /aggs: %s, %s -> %s, %dmin", syms, start, end, multiplier)

    rows: list[dict[str, object]] = []
    for sym in syms:
        aggs = _paginated_aggs(
            ticker=sym,
            multiplier=multiplier,
            timespan=timespan,
            from_str=start.strftime("%Y-%m-%d"),
            to_str=end.strftime("%Y-%m-%d"),
        )
        for agg in aggs:
            ts = pd.Timestamp(agg["t"], unit="ms", tz=UTC)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": float(agg["o"]),
                    "high": float(agg["h"]),
                    "low": float(agg["l"]),
                    "close": float(agg["c"]),
                    "volume": float(agg["v"]),
                    "trade_count": float(agg.get("n") or 0),
                    "vwap": float(agg.get("vw") or 0.0),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype("string")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

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
    out_path = raw_dir() / f"polygon_bars_{DEFAULT_SYMBOL}_30min.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d %s bars -> %s", len(df), DEFAULT_SYMBOL, out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_gld_parquet()
    print(f"GLD Polygon bars: {len(df)} rows -> {p}")
