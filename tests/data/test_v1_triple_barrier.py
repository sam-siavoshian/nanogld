"""Unit tests for triple_barrier label function (V1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanogld.features import triple_barrier as tb


@pytest.mark.smoke
def test_basic_up_down_flat() -> None:
    nlr = np.array([0.005, -0.005, 0.0001, 0.0, 0.0010, -0.0010])
    up = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    dn = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    spread_bps = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    labels = tb.triple_barrier_label(nlr, up, dn, spread_bps)
    assert labels.tolist() == [1, -1, 0, 0, 1, -1]


@pytest.mark.smoke
def test_spread_neutral_overrides_barrier() -> None:
    """If |return| < spread/1e4, label is FLAT even when barrier touched."""
    nlr = np.array([0.0001])
    up = np.array([0.00005])
    dn = np.array([0.00005])
    spread_bps = np.array([2.0])

    labels = tb.triple_barrier_label(nlr, up, dn, spread_bps)
    assert labels[0] == 0


@pytest.mark.smoke
def test_nan_inputs_map_to_flat() -> None:
    nlr = np.array([np.nan, 0.005, np.nan])
    up = np.array([0.001, np.nan, 0.001])
    dn = np.array([0.001, 0.001, 0.001])
    spread_bps = np.array([2.0, 2.0, np.nan])

    labels = tb.triple_barrier_label(nlr, up, dn, spread_bps)
    assert labels.tolist() == [0, 0, 0]


@pytest.mark.smoke
def test_to_ce_class_mapping() -> None:
    arr = np.array([-1, 0, 1, -1, 1], dtype=np.int8)
    ce = tb.to_ce_class(arr)
    assert ce.tolist() == [0, 1, 2, 0, 2]


@pytest.mark.smoke
def test_class_distribution_sums_to_one() -> None:
    labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    dist = tb.class_distribution(labels)
    assert abs(dist["DOWN"] + dist["FLAT"] + dist["UP"] - 1.0) < 1e-9
    assert dist["DOWN"] == pytest.approx(2 / 9)
    assert dist["UP"] == pytest.approx(4 / 9)


@pytest.mark.smoke
def test_add_columns_round_trip() -> None:
    df = pd.DataFrame(
        {
            "next_log_return": [0.005, -0.005, 0.0001, 0.0010],
            "barrier_up": [0.001, 0.001, 0.001, 0.001],
            "barrier_down": [0.001, 0.001, 0.001, 0.001],
            "gld_spread_bps_t": [2.0, 2.0, 2.0, 2.0],
        }
    )
    out = tb.add_triple_barrier_columns(df)
    assert "label_triple_barrier" in out.columns
    assert "label_ce" in out.columns
    assert out["label_triple_barrier"].tolist() == [1, -1, 0, 1]
    assert out["label_ce"].tolist() == [2, 0, 1, 2]


@pytest.mark.smoke
def test_missing_column_raises() -> None:
    df = pd.DataFrame({"next_log_return": [0.005], "barrier_up": [0.001]})
    with pytest.raises(KeyError):
        tb.add_triple_barrier_columns(df)
