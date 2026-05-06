"""Point-in-time joiner — produces the bar-aligned snapshot consumed by doc 04.

Hard rules from plan/02-DATA-PIPELINE.md:
- Strict `<` joins everywhere (`allow_exact_matches=False`). A source row whose
  t_visible == bar_close is visible at the NEXT bar, not this one.
- Bar visibility = bar END = bar.timestamp + 30min. So the row indexed by
  bar_close_utc T contains ONLY data with t_visible < T.
- News fields are NOT text-aggregated here (doc 03 owns embedding); we emit
  per-bar counts for sanity / downstream feature engineering.

Inputs are produced by the per-source modules. Owner orchestrates via
`python -m nanogld.data build`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from nanogld.data.utils import (
    assert_t_visible_invariant,
    get_logger,
    nyse_rth_index,
    raw_dir,
)

LOG = get_logger("nanogld.data.join")

PRIMARY_SYMBOL = "GLD"
ETF_SYMBOLS = ("SPY", "QQQ", "IWM", "GDX", "SLV", "XLF", "XLE", "XLK", "XLU")


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _enforce_pit(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ensure (release_ts, t_visible) exist + invariant holds."""
    if df.empty:
        return df
    if "release_ts" not in df.columns or "t_visible" not in df.columns:
        raise ValueError(f"[join.{name}] source missing release_ts/t_visible columns")
    assert_t_visible_invariant(df)
    return df.sort_values("t_visible").reset_index(drop=True)


def _bar_close_index(bars_gld: pd.DataFrame) -> pd.DataFrame:
    """Build the bar_close_utc index from GLD bars."""
    g = bars_gld[bars_gld["symbol"] == PRIMARY_SYMBOL].copy()
    if g.empty:
        return pd.DataFrame(columns=["bar_close_utc"])
    g["bar_close_utc"] = g["release_ts"]  # = timestamp + 30min
    out = g[["bar_close_utc", "open", "high", "low", "close", "volume", "vwap"]].rename(
        columns={c: f"gld_{c}" for c in ("open", "high", "low", "close", "volume", "vwap")}
    )
    return out.sort_values("bar_close_utc").reset_index(drop=True)


def _attach_bars_lag(
    base: pd.DataFrame, bars: pd.DataFrame, *, sym: str, prefix: str
) -> pd.DataFrame:
    """Lag-1 bar features from `sym` joined onto each bar_close_utc.

    A bar with t_visible == bar_close is excluded by the strict-< join, which
    is correct: the bar that just closed is not yet "available" for decisions
    keyed at that timestamp.
    """
    if bars.empty or "symbol" not in bars.columns:
        return base
    sub = bars[bars["symbol"] == sym].copy()
    if sub.empty:
        return base
    sub = (
        sub[["t_visible", "open", "high", "low", "close", "volume", "vwap"]]
        .rename(
            columns={c: f"{prefix}_{c}" for c in ("open", "high", "low", "close", "volume", "vwap")}
        )
        .sort_values("t_visible")
    )
    base = base.copy()
    base["bar_close_utc"] = pd.to_datetime(base["bar_close_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    sub["t_visible"] = pd.to_datetime(sub["t_visible"], utc=True).astype("datetime64[ns, UTC]")
    return pd.merge_asof(
        base.sort_values("bar_close_utc"),
        sub,
        left_on="bar_close_utc",
        right_on="t_visible",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns=["t_visible"])


def _attach_daily(
    base: pd.DataFrame,
    df: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Forward-fill the latest visible daily value onto each bar_close_utc."""
    if df.empty:
        base[out_col] = pd.NA
        return base
    sub = df[["t_visible", value_col]].dropna().rename(columns={value_col: out_col})
    sub = sub.sort_values("t_visible")
    # pandas 2.3 strict resolution: coerce both sides to ns UTC.
    base = base.copy()
    base["bar_close_utc"] = pd.to_datetime(base["bar_close_utc"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    sub["t_visible"] = pd.to_datetime(sub["t_visible"], utc=True).astype("datetime64[ns, UTC]")
    return pd.merge_asof(
        base.sort_values("bar_close_utc"),
        sub,
        left_on="bar_close_utc",
        right_on="t_visible",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns=["t_visible"])


def _attach_news_counts(
    base: pd.DataFrame, news: pd.DataFrame, *, source_label: str
) -> pd.DataFrame:
    """Per-bar count of news rows whose t_visible falls in (prev_close, close].

    Pre-V1 articles (t_visible < first bar_close - 30min) are dropped so they
    don't dump into bar 0 via the diff `fillna(counts[0])` path. Same on the
    tail: post-last-bar articles are ignored for snapshot purposes.
    """
    if news.empty:
        base[f"{source_label}_news_count"] = 0
        return base
    base = base.sort_values("bar_close_utc").reset_index(drop=True)
    first_close = base["bar_close_utc"].iloc[0]
    last_close = base["bar_close_utc"].iloc[-1]
    window_start = first_close - pd.Timedelta(minutes=30)
    news_in_window = news[(news["t_visible"] > window_start) & (news["t_visible"] <= last_close)]
    if news_in_window.empty:
        base[f"{source_label}_news_count"] = 0
        return base
    news_idx = pd.DatetimeIndex(news_in_window["t_visible"]).sort_values()
    counts = news_idx.searchsorted(pd.DatetimeIndex(base["bar_close_utc"]), side="right")
    counts_diff = pd.Series(counts).diff().fillna(counts[0]).clip(lower=0).astype("int64")
    base[f"{source_label}_news_count"] = counts_diff.values
    return base


def _attach_calendar_proximity(base: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Add a binary `event_within_60min` flag per bar (V1 hard rule §14:
    binary windows ONLY, no minutes_until_event).
    """
    if cal.empty:
        base["event_within_60min"] = False
        return base
    base = base.sort_values("bar_close_utc").reset_index(drop=True)
    events = pd.DatetimeIndex(cal["event_ts_utc"]).sort_values()
    flag = []
    # Symmetric ±60min window (V1 hard rule §14: binary windows ONLY).
    for c in base["bar_close_utc"]:
        window_lo = c - pd.Timedelta(minutes=60)
        window_hi = c + pd.Timedelta(minutes=60)
        idx = events.searchsorted(window_lo, side="left")
        flag.append(bool(idx < len(events) and events[idx] <= window_hi))
    base["event_within_60min"] = flag
    return base


def join_snapshot(
    sources: Mapping[str, pd.DataFrame],
    *,
    cot_value: str = "mm_long",
) -> pd.DataFrame:
    """Build the primary bar-aligned snapshot.

    `sources` keys:
      bars             — Alpaca GLD bars (primary)
      etf_bars         — Alpaca ETF basket bars (multi-symbol long form)
      alpaca_news      — Benzinga news
      gdelt            — GDELT GKG (optional)
      fred             — long-form ALFRED cube (any subset of FRED_SERIES_V1)
      brent_wti        — yfinance Brent + WTI daily
      gpr              — Caldara-Iacoviello GPR + AI-GPR
      cot              — CFTC COT weekly
      wgc              — WGC central-bank monthly
      calendar         — deterministic FOMC/CPI/NFP/... schedule
    Missing sources are tolerated; they just don't contribute columns.
    """
    bars = _enforce_pit(sources.get("bars", pd.DataFrame()), "bars")
    base = _bar_close_index(bars)
    if base.empty:
        raise RuntimeError("join: no GLD bars in primary source — check bars parquet")
    LOG.info("join base = %d GLD bars", len(base))

    base = _attach_bars_lag(base, bars, sym=PRIMARY_SYMBOL, prefix="gld_lag1")

    etf_bars = _enforce_pit(sources.get("etf_bars", pd.DataFrame()), "etf_bars")
    for sym in ETF_SYMBOLS:
        base = _attach_bars_lag(base, etf_bars, sym=sym, prefix=f"{sym.lower()}_lag1")

    fred = _enforce_pit(sources.get("fred", pd.DataFrame()), "fred")
    if not fred.empty:
        for sid, sub in fred.groupby("series_id"):
            base = _attach_daily(base, sub, value_col="value", out_col=f"fred_{sid.lower()}")

    bw = _enforce_pit(sources.get("brent_wti", pd.DataFrame()), "brent_wti")
    if not bw.empty:
        for ticker in ("BZ=F", "CL=F"):
            sub = bw[bw["ticker"] == ticker]
            label = "brent" if ticker == "BZ=F" else "wti"
            base = _attach_daily(base, sub, value_col="close", out_col=f"{label}_close")

    gpr = _enforce_pit(sources.get("gpr", pd.DataFrame()), "gpr")
    if not gpr.empty:
        # collapse to one canonical GPR series for V1 — owner can broaden in doc 04
        canon = gpr[gpr["series"].str.contains("GPR", case=False, regex=False)]
        if not canon.empty:
            latest_per_date = canon.sort_values("fetch_ts").groupby(["series", "date"]).tail(1)
            base = _attach_daily(base, latest_per_date, value_col="value", out_col="gpr_value")

    cot = _enforce_pit(sources.get("cot", pd.DataFrame()), "cot")
    if not cot.empty:
        base = _attach_daily(base, cot, value_col=cot_value, out_col=f"cot_{cot_value}")

    wgc = _enforce_pit(sources.get("wgc", pd.DataFrame()), "wgc")
    if not wgc.empty and "holdings_tonnes" in wgc.columns:
        wgc_world = wgc.groupby("period")["holdings_tonnes"].sum().reset_index()
        wgc_world["t_visible"] = wgc.groupby("period")["t_visible"].max().values
        base = _attach_daily(
            base,
            wgc_world.rename(columns={"holdings_tonnes": "value"}),
            value_col="value",
            out_col="wgc_world_holdings_t",
        )

    cal = sources.get("calendar", pd.DataFrame())
    base = _attach_calendar_proximity(base, cal if not cal.empty else pd.DataFrame())

    alp_news = _enforce_pit(sources.get("alpaca_news", pd.DataFrame()), "alpaca_news")
    base = _attach_news_counts(base, alp_news, source_label="alpaca")

    gdelt = _enforce_pit(sources.get("gdelt", pd.DataFrame()), "gdelt")
    base = _attach_news_counts(base, gdelt, source_label="gdelt")

    for src_key, label in (
        ("central_bank", "central_bank"),
        ("kitco", "kitco"),
        ("investing", "investing"),
        ("bullionvault", "bullionvault"),
        ("multisource", "multisource"),
        ("fnspid", "fnspid"),
    ):
        df = _enforce_pit(sources.get(src_key, pd.DataFrame()), src_key)
        base = _attach_news_counts(base, df, source_label=label)

    base["bar_close_utc"] = pd.to_datetime(base["bar_close_utc"], utc=True)
    return base.sort_values("bar_close_utc").reset_index(drop=True)


def load_default_sources() -> dict[str, pd.DataFrame]:
    """Best-effort: load every parquet under data/raw/ that the spec produces.

    GLD primary bars: prefer Polygon (replaces Alpaca after KYC switch).
    Alpaca path retained as fallback for owner-specific runs.
    """
    rd = raw_dir()
    polygon_gld = rd / "polygon_bars_GLD_30min.parquet"
    alpaca_gld = rd / "alpaca_bars_GLD_30min.parquet"
    bars_path = polygon_gld if polygon_gld.exists() else alpaca_gld
    polygon_news_path = rd / "polygon_news_GLD.parquet"
    alpaca_news_path = rd / "alpaca_news_GLD.parquet"
    news_path = polygon_news_path if polygon_news_path.exists() else alpaca_news_path
    out: dict[str, pd.DataFrame] = {
        "bars": _load_parquet(bars_path),
        "alpaca_news": _load_parquet(news_path),  # key kept "alpaca_news" for joiner compat
        "gdelt": _load_parquet(rd / "gdelt_gkg_5y.parquet"),
        "brent_wti": _concat_parquets([rd / "brent_daily.parquet", rd / "wti_daily.parquet"]),
        "gpr": _load_parquet(rd / "gpr_combined.parquet"),
        "cot": _load_parquet(rd / "cftc_cot_gold_weekly.parquet"),
        "wgc": _load_parquet(
            rd / "wgc_central_bank_quarterly.parquet"
            if (rd / "wgc_central_bank_quarterly.parquet").exists()
            else rd / "wgc_central_bank_monthly.parquet"
        ),
        "calendar": _load_parquet(rd / "calendar_events_v1.parquet"),
        "central_bank": _load_parquet(rd / "central_bank_news.parquet"),
        "kitco": _load_parquet(rd / "kitco_news.parquet"),
        "investing": _load_parquet(rd / "investing_gold_news.parquet"),
        "bullionvault": _load_parquet(rd / "bullionvault_news.parquet"),
        "multisource": _load_parquet(rd / "multisource_news.parquet"),
        "fnspid": _load_parquet(rd / "fnspid_gold_news.parquet"),
    }
    # ETFs: long-form concat. Prefer Polygon over Alpaca per-symbol.
    etf_paths: list[Path] = []
    for s in ETF_SYMBOLS:
        poly_p = rd / f"polygon_bars_{s}_30min.parquet"
        alp_p = rd / f"alpaca_bars_{s}_30min.parquet"
        etf_paths.append(poly_p if poly_p.exists() else alp_p)
    out["etf_bars"] = _concat_parquets(etf_paths)
    # FRED: long-form concat across all 35 series
    out["fred"] = _concat_parquets(list(rd.glob("fred_*_all_releases.parquet")))
    return out


def _concat_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = [_load_parquet(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def expected_bar_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Approximate row-count gate: ~16,380 bars for 5y RTH 30min."""
    return len(nyse_rth_index(start, end))
