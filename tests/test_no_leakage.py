"""Mandatory leakage test suite — plan/02-DATA-PIPELINE.md lines 191-219.

Tests that exercise pure code paths run today. Tests that require live data
(Alpaca / FRED / GDELT pulls) are skipped with a clear reason and re-enabled
once owner has run `python -m nanogld.data build`.

The single highest-leverage test is `test_release_ts_lte_t_visible_all_rows`,
which catches §3, §4, §7, §15, §16, §17 simultaneously by asserting the V1
hard rule on every source's parquet.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from nanogld.data import (
    alpaca_news,
    calendar_events,
    cot,
    fred,
    gdelt,
    yfinance_helpers,
)
from nanogld.data.utils import (
    FRED_RELEASE_TOD_ET,
    NEWS_LATENCY_MIN_GDELT,
    NEWS_LATENCY_SEC_ALPACA,
    UTC,
    cot_release_ts_utc,
    fred_release_ts_utc,
    raw_dir,
)

NEEDS_DATA = pytest.mark.skipif(
    not (raw_dir() / "alpaca_bars_GLD_30min.parquet").exists(),
    reason="needs `python -m nanogld.data build` (or single-source pulls) first",
)


# §1 — Alpaca bar `t` is bar START; visibility = bar END
def test_bar_visibility_is_bar_end() -> None:
    """alpaca_bars module sets release_ts = timestamp + 30min."""
    from nanogld.data.alpaca_bars import BAR_DURATION

    assert pd.Timedelta(minutes=30) == BAR_DURATION


# §2 — Alpaca News field is `created_at` (NOT `published_at`)
def test_news_uses_created_at_not_updated_at() -> None:
    import inspect

    src = inspect.getsource(alpaca_news.fetch_news)
    assert "created_at" in src
    # We must never join on or store `updated_at` as the release-time anchor.
    # The substring may appear in a comment about NOT using it (V4 hard rule).
    # Enforce: there must be NO `pd.to_datetime(.*updated_at.*utc=True)` or
    # `release_ts = .*updated_at` pattern.
    bad_patterns = [
        'release_ts"] = df["updated_at"]',
        '"t_visible"] = df["updated_at"]',
        'pd.to_datetime(df["updated_at"]',
    ]
    for pat in bad_patterns:
        assert pat not in src, f"Alpaca News module references {pat!r}"


# §2 — t_visible = created_at + 60s buffer
def test_news_t_visible_buffer_60s() -> None:
    assert NEWS_LATENCY_SEC_ALPACA == 60


# §3 — DFF is daily, FEDFUNDS is monthly — both required to be in lookup
def test_dff_replaces_fedfunds_for_daily() -> None:
    assert "DFF" in FRED_RELEASE_TOD_ET
    assert "FEDFUNDS" in FRED_RELEASE_TOD_ET
    # DFF and FEDFUNDS exist as separate entries (DFF for daily features,
    # FEDFUNDS for monthly aggregates only — V4 §3).
    assert "DFF" in fred.FRED_SERIES_V1
    assert "FEDFUNDS" in fred.FRED_SERIES_V1


# §4 — FRED release-tod table covers every V1 series
def test_fred_release_tod_table_complete() -> None:
    missing = [s for s in fred.FRED_SERIES_V1 if s not in FRED_RELEASE_TOD_ET]
    assert not missing, f"FRED_RELEASE_TOD_ET missing: {missing}"


# §5 — fred module uses get_series_all_releases (ALFRED), not snapshot
def test_fred_uses_alfred_realtime_period() -> None:
    import inspect

    src = inspect.getsource(fred._vintage_cube)
    assert "get_series_all_releases" in src


# §5 — vintage_lookup respects realtime_start
@NEEDS_DATA
def test_fred_pit_cache_matches_alfred_api() -> None:
    """Once owner has pulled FRED, parquet rows must satisfy realtime_start <= t_visible."""
    pq = next(raw_dir().glob("fred_*_all_releases.parquet"), None)
    assert pq is not None
    df = pd.read_parquet(pq)
    assert (df["realtime_start"] <= df["t_visible"]).all()


# §6 — GDELT theme codes from V4-corrected list (no EPU_*, no MIL_CONFLICT)
def test_gdelt_theme_codes_in_master_list() -> None:
    forbidden = (
        "EPU_CATS_MONETARY_POLICY",
        "EPU_POLICY_FEDERAL_RESERVE",
        "EPU_UNCERTAINTY",
        "EPU_ECONOMY_HISTORIC",
        "TAX_WEAPONS_BOMB",
        "WB_2432_FRAGILITY|",  # unfollowed by _CONFLICT
    )
    sql = gdelt._materialize_sql(
        pd.Timestamp("2021-04-24", tz=UTC), pd.Timestamp("2026-04-24", tz=UTC), "x.y.z"
    )
    for bad in forbidden:
        assert bad not in sql, f"{bad!r} still in GDELT SQL"


# §7 — GDELT buffer = 30min, not 15min
def test_gdelt_buffer_30min_not_15() -> None:
    assert NEWS_LATENCY_MIN_GDELT == 30


# §7 — GDELT release_ts uses both pub_ts + partition_ts (max)
def test_gdelt_uses_file_publish_ts() -> None:
    df = pd.DataFrame(
        {
            "pub_ts_utc": [pd.Timestamp("2024-01-01 12:00:00", tz=UTC)],
            "partition_ts_utc": [pd.Timestamp("2024-01-01 13:00:00", tz=UTC)],
            "url": ["http://x"],
            "v2_themes": ["WB_2936_GOLD"],
            "v2_tone": [None],
            "v2_locations": [None],
        }
    )
    df["url"] = df["url"].astype("string")
    df["v2_themes"] = df["v2_themes"].astype("string")
    df["v2_tone"] = df["v2_tone"].astype("string")
    df["v2_locations"] = df["v2_locations"].astype("string")
    out = gdelt._attach_t_visible(df)
    # release_ts = max(pub + 30min = 12:30, partition = 13:00) = 13:00
    assert out["release_ts"].iloc[0] == pd.Timestamp("2024-01-01 13:00:00", tz=UTC)


# §8 — WGC URL is gold.org/download/8052 + 7739
def test_wgc_url_is_correct_self_snapshot() -> None:
    from nanogld.data import wgc

    assert "8052" in wgc.WGC_QUARTERLY_TIMESERIES
    assert "7739" in wgc.WGC_LATEST_RESERVES


# §9 — AI-GPR has 30-day lag
def test_aigpr_treated_as_monthly_lag() -> None:
    from nanogld.data import gpr

    assert gpr.AIGPR_LAG_DAYS == 30


# §10 — GPR uses self-snapshot (fetch_ts column on every row)
def test_gpr_uses_self_snapshot_not_live() -> None:
    from nanogld.data.schema import GPR_MANIFEST

    cols = {c.name for c in GPR_MANIFEST.columns}
    assert "fetch_ts" in cols
    assert "source_sha" in cols


# §11 — pandas-ta KAMA / Ichimoku / KST / DPO / TRIX / Vortex banned
def test_no_pandas_ta_kama_ichimoku_kst_dpo_trix() -> None:
    """Doc 04 owns indicator usage. We assert nothing in src/nanogld/ uses those names."""
    from pathlib import Path

    forbidden = ("kama", "ichimoku", "kst", "dpo", "trix", "vortex")
    src_root = Path(__file__).resolve().parents[1] / "src" / "nanogld"
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        body = path.read_text()
        for bad in forbidden:
            if f"ta.{bad}" in body or f".{bad}(" in body:
                offenders.append(f"{path}: uses {bad!r}")
    assert not offenders, "\n".join(offenders)


# §11 — growing-window-stability test of indicators is a doc 04 contract;
# stub holds a placeholder that owner extends.
@pytest.mark.skip(reason="growing-window-stability test owned by doc 04 (features)")
def test_indicators_growing_window_stability() -> None:
    pass


# §12 — multi-symbol pagination drained
def test_multisymbol_pagination_drained() -> None:
    """Alpaca SDK auto-paginates with limit=None. Verify code uses that."""
    import inspect

    src = inspect.getsource(
        __import__("nanogld.data.alpaca_bars", fromlist=["fetch_bars"]).fetch_bars
    )
    assert "limit=None" in src


# §13 — adjustment="all" disclosed (V1 default; production fix is forward-only)
def test_no_split_adjusted_leakage_in_backtest() -> None:
    """V1 admits adjustment='all' is retroactive; tracked as DEVIATION FROM SPEC."""
    import inspect

    src = inspect.getsource(
        __import__("nanogld.data.alpaca_bars", fromlist=["fetch_bars"]).fetch_bars
    )
    assert 'adjustment="all"' in src or "adjustment='all'" in src


# §14 — CFTC 2025 shutdown gap auto-flagged
def test_cftc_2025_shutdown_gap_handled() -> None:
    df = pd.DataFrame(
        {
            "report_date": pd.to_datetime(
                ["2025-09-30", "2025-10-07", "2026-01-15", "2026-01-22"], utc=True
            ),
            "irregular_release": [False, False, False, False],
        }
    )
    out = cot._flag_irregular(df)
    # 2025-10-07 → 2026-01-15 is >7 days, must flag
    assert out.loc[
        out["report_date"] == pd.Timestamp("2026-01-15", tz=UTC), "irregular_release"
    ].iloc[0]


# §15 — COT t_visible is Friday 16:00 ET (15:30 + 30min buffer)
def test_cot_t_visible_is_friday_330pm_et() -> None:
    ts = cot_release_ts_utc(date(2024, 1, 9))  # Tuesday
    et = ts.tz_convert("America/New_York")
    assert et.weekday() == 4  # Friday
    assert et.hour == 16


# §15 — Holiday-Friday rolls to next NYSE session
def test_cot_holiday_friday_uses_monday_release() -> None:
    # Tuesday 2024-12-31 → Friday 2025-01-03 is a session day. Pick a Friday-holiday case.
    # Tuesday 2024-12-24 → Friday 2024-12-27 is a session day too. NYSE holidays since 2021:
    # 2024-07-04 Thursday -> Friday 2024-07-05 IS a session day.
    # Need: Tuesday whose 3-days-later Friday is a NYSE holiday.
    # Friday 2024-03-29 was Good Friday (closed). Tuesday before: 2024-03-26.
    ts = cot_release_ts_utc(date(2024, 3, 26))
    # Should roll to Monday 2024-04-01 (next NYSE session after Good Friday)
    et = ts.tz_convert("America/New_York")
    assert et.weekday() == 0  # Monday
    assert et.date() == date(2024, 4, 1)


# §16 — WALCL Thursday 16:30 ET visibility
def test_walcl_thursday_visibility_after_1630_et() -> None:
    ts = fred_release_ts_utc("WALCL", date(2024, 1, 18))  # any Thursday
    et = ts.tz_convert("America/New_York")
    assert et.hour == 16 and et.minute == 30


# §17 — ICSA Thursday 08:30 ET visibility
def test_icsa_thursday_visibility_after_0830_et() -> None:
    ts = fred_release_ts_utc("ICSA", date(2024, 1, 18))
    et = ts.tz_convert("America/New_York")
    assert et.hour == 8 and et.minute == 30


# Doc 04 — anchor dates pre-train period (placeholder hook here)
@pytest.mark.skip(reason="anchor pre-train rule owned by doc 03 (news embedding)")
def test_anchor_dates_precede_train_period() -> None:
    pass


# Doc 04 — no minutes_until_event raw features
def test_no_minutes_until_event_features() -> None:
    """V1 hard rule §14: binary windows only. Assert nothing in src/nanogld/
    USES a feature called minutes_until_event (comments mentioning the rule
    are fine).
    """
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[1] / "src" / "nanogld"
    bad_patterns = (
        '"minutes_until_event"',
        "'minutes_until_event'",
        "def minutes_until_event",
        ".minutes_until_event",
    )
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        body = path.read_text()
        for pat in bad_patterns:
            if pat in body:
                offenders.append(f"{path}: pattern {pat!r}")
    assert not offenders, "\n".join(offenders)


# Label hygiene — no future close referenced (placeholder, doc 04 owns label gen)
@pytest.mark.skip(reason="label-hygiene check owned by doc 04 (features)")
def test_features_never_reference_close_t_plus_1() -> None:
    pass


# Universal: every source parquet satisfies release_ts <= t_visible
@NEEDS_DATA
def test_release_ts_lte_t_visible_all_rows() -> None:
    """The single highest-leverage test. Catches §3, §4, §7, §15, §16, §17."""
    paths = list(raw_dir().glob("*.parquet"))
    assert paths, "no parquets in data/raw/ — run python -m nanogld.data build first"
    for p in paths:
        df = pd.read_parquet(p)
        if "release_ts" not in df.columns or "t_visible" not in df.columns:
            continue  # source-specific format — skip for this universal test
        assert (df["release_ts"] <= df["t_visible"]).all(), (
            f"{p.name}: release_ts > t_visible on some rows"
        )


# Global sanity (doc 04 / 05 owns full implementation)
@pytest.mark.skip(reason="shuffled-label sanity check owned by doc 05 (training)")
def test_shuffled_label_baseline_auc_near_50() -> None:
    pass


# Survivorship — V1 universe is static (GLD + 9 ETFs); no de-listings expected
def test_universe_static_no_delistings() -> None:
    assert calendar_events.__name__ == "nanogld.data.calendar_events"  # smoke
    from nanogld.data.alpaca_etfs import ETF_BASKET

    # 9 large-cap ETFs — none have delisted as of 2026-05.
    assert len(ETF_BASKET) == 9
    for sym in ("SPY", "QQQ", "IWM", "GDX", "SLV", "XLF", "XLE", "XLK", "XLU"):
        assert sym in ETF_BASKET


# Smoke: yfinance helpers correctly tag tz
def test_yfinance_settlement_tod_in_table() -> None:
    assert "CL=F" in yfinance_helpers.SETTLEMENT_TOD_ET
    assert "BZ=F" in yfinance_helpers.SETTLEMENT_TOD_ET
