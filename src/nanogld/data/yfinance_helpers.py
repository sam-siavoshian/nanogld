"""Brent + WTI daily futures via yfinance — Source 5.

Spec hard rules (plan/02-DATA-PIPELINE.md "Source 5"):
- DAILY only. yfinance 30m bars are capped at 60 days.
- yfinance==1.3.0 pinned (April 2026 dividends fix).
- curl_cffi session with chrome impersonation to dodge Yahoo rate limiting.
- BZ=F + CL=F return tz=America/New_York; convert to UTC immediately.
- Settlement times: CL ~14:30 ET (NYMEX), BZ ~15:00 ET (ICE).
- Drop today's partial bar (period='5y' includes the unsettled current day).
- Brent must lag ≥1 bar when joined to US RTH (Brent close lands AFTER GLD close).
"""

from __future__ import annotations

from datetime import datetime
from datetime import time as dt_time

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

from nanogld.data.schema import YFINANCE_DAILY_MANIFEST, validate
from nanogld.data.utils import ET, UTC, get_logger, raw_dir

LOG = get_logger("nanogld.data.yfinance")

# Settlement time of day in ET — bar `t_visible` = settlement
SETTLEMENT_TOD_ET: dict[str, dt_time] = {
    "CL=F": dt_time(14, 30),  # NYMEX WTI 14:30 ET
    "BZ=F": dt_time(15, 0),  # ICE Brent ~15:00 ET in summer / 15:00 ET in winter
}


def _impersonating_session() -> curl_requests.Session:
    return curl_requests.Session(impersonate="chrome")


def _settlement_ts_utc(observation_date: pd.Timestamp, ticker: str) -> pd.Timestamp:
    """Return UTC settlement timestamp for a daily observation."""
    tod = SETTLEMENT_TOD_ET.get(ticker, dt_time(16, 0))
    naive = observation_date.tz_localize(None) if observation_date.tzinfo else observation_date
    et = pd.Timestamp(datetime.combine(naive.date(), tod, tzinfo=ET))
    return et.tz_convert(UTC)


def fetch_daily(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch daily bars for a single ticker. Returns tidy frame."""
    LOG.info("fetching yfinance %s period=%s", ticker, period)
    session = _impersonating_session()
    raw = yf.Ticker(ticker, session=session).history(period=period, interval="1d")
    if raw.empty:
        LOG.warning("yfinance returned empty frame for %s", ticker)
        return pd.DataFrame()

    raw = raw.reset_index()
    date_col = next(c for c in raw.columns if c.lower() in {"date", "datetime"})
    raw = raw.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    raw["ticker"] = pd.Series([ticker] * len(raw), dtype="string")
    raw["date"] = pd.to_datetime(raw["date"], utc=True)

    # Drop today's partial bar — only fully settled days
    today_utc = pd.Timestamp.now(tz=UTC).normalize()
    raw = raw[raw["date"] < today_utc].copy()

    raw["settlement_ts_utc"] = raw["date"].apply(lambda d: _settlement_ts_utc(d, ticker))
    raw["release_ts"] = raw["settlement_ts_utc"]
    raw["t_visible"] = raw["release_ts"]

    cols = [c.name for c in YFINANCE_DAILY_MANIFEST.columns]
    out = raw[[c for c in cols if c in raw.columns]].copy()
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["volume"] = pd.to_numeric(
        out.get("volume", pd.Series([pd.NA] * len(out))), errors="coerce"
    ).astype("float64")
    return out.reset_index(drop=True)


def fetch_brent_wti() -> pd.DataFrame:
    """Pull both BZ=F (Brent) and CL=F (WTI) for the 10y window."""
    frames = [fetch_daily("BZ=F"), fetch_daily("CL=F")]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return df.dropna(subset=["close"]).reset_index(drop=True)


def write_yfinance_parquet() -> tuple[pd.DataFrame, str]:
    df = fetch_brent_wti()
    if df.empty:
        LOG.warning("yfinance produced 0 rows; check network + version pin")
        return df, ""
    validate(df, YFINANCE_DAILY_MANIFEST)

    out_dir = raw_dir()
    by_ticker = df.groupby("ticker")
    paths: list[str] = []
    for ticker, sub in by_ticker:
        suffix = (
            "brent" if ticker == "BZ=F" else "wti" if ticker == "CL=F" else ticker.replace("=", "_")
        )
        p = out_dir / f"{suffix}_daily.parquet"
        sub.to_parquet(p, compression="zstd", index=False)
        paths.append(str(p))
        LOG.info("wrote %d %s rows -> %s", len(sub), ticker, p)
    return df, ", ".join(paths)


if __name__ == "__main__":
    df, paths = write_yfinance_parquet()
    print(f"yfinance: {len(df)} rows -> {paths}")
    print(df.groupby("ticker").size())
