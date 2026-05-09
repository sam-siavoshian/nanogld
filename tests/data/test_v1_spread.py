"""Unit tests for V1 GLD spread feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanogld.features import spread


def _synthetic_bars(n: int = 8) -> pd.DataFrame:
    bar_close = pd.date_range("2024-06-03 13:30:00+00:00", periods=n, freq="30min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n)) * 0.1
    return pd.DataFrame(
        {
            "bar_close_utc": bar_close,
            "gld_high": close + 0.05,
            "gld_low": close - 0.05,
            "gld_close": close,
        }
    )


@pytest.mark.smoke
def test_spread_proxy_when_no_quotes() -> None:
    df = _synthetic_bars()
    out = spread.add_spread_feature(df, quotes_df=None)
    assert "gld_spread_bps_t" in out.columns
    assert (out["gld_spread_bps_t"] >= 0).all()
    assert out["gld_spread_bps_t"].notna().all()


@pytest.mark.smoke
def test_spread_uses_quotes_when_available() -> None:
    df = _synthetic_bars(n=8)
    quotes = pd.DataFrame(
        {
            "bar_close_utc": pd.date_range("2024-06-03 13:00:00+00:00", periods=20, freq="5min"),
            "gld_spread_bps": np.full(20, 0.7, dtype=np.float32),
        }
    )
    out = spread.add_spread_feature(df, quotes_df=quotes)
    assert out["gld_spread_bps_t"].mean() < 5.0, "real quotes should give tight spread"


@pytest.mark.smoke
def test_spread_clip_non_negative() -> None:
    df = _synthetic_bars()
    df["gld_high"] = df["gld_close"]
    df["gld_low"] = df["gld_close"]
    out = spread.add_spread_feature(df, quotes_df=None)
    assert (out["gld_spread_bps_t"] >= 0).all()
