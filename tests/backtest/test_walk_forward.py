"""Regression locks for ``walk_forward.py`` harness + ``report.py`` renderer.

Tests the high-level shape contract: feed synthetic per-fold contexts +
trivial strategy functions, verify the result bundle and the rendered
markdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanogld.backtest.report import MODEL_NAME, render_report
from nanogld.backtest.walk_forward import (
    evaluate_strategy_positions,
    run_fold,
    walk_forward,
)


def _toy_ctx(seed: int = 0, t: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nlr = rng.normal(0.0, 0.001, size=t)
    return {
        "next_log_returns": nlr,
        "is_news_present": rng.random(t) > 0.5,
        "close": 100.0 * np.exp(np.cumsum(nlr)),
    }


def _const_pos(value: float) -> Any:
    def _fn(ctx: dict[str, Any]) -> np.ndarray:
        return np.full_like(ctx["next_log_returns"], value)

    return _fn


def test_evaluate_strategy_positions_shape_contract() -> None:
    ctx = _toy_ctx()
    res = evaluate_strategy_positions(
        name="const",
        positions=np.full_like(ctx["next_log_returns"], 0.5),
        next_log_returns=ctx["next_log_returns"],
        is_news_present=ctx["is_news_present"],
    )
    assert res.name == "const"
    assert res.positions.shape == ctx["next_log_returns"].shape
    assert {0.5, 1.0, 1.5} <= set(res.cost_stress.by_multiplier.keys())
    assert {"present", "absent", "both"} == set(res.per_bucket.keys())
    assert isinstance(res.dsr_p_value, float)
    assert isinstance(res.dsr_value, float)


def test_evaluate_strategy_positions_rejects_shape_mismatch() -> None:
    ctx = _toy_ctx()
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_strategy_positions(
            name="bad",
            positions=np.zeros(50),
            next_log_returns=ctx["next_log_returns"],
            is_news_present=ctx["is_news_present"],
        )


def test_run_fold_runs_all_strategies() -> None:
    ctx = _toy_ctx()
    strategies = {
        MODEL_NAME: _const_pos(0.3),
        "buy_hold": _const_pos(1.0),
        "short": _const_pos(-1.0),
    }
    fold = run_fold(0, ctx, strategies)
    assert set(fold.strategies.keys()) == set(strategies.keys())
    assert fold.test_n_bars == 200
    assert 0.0 <= fold.test_news_present_frac <= 1.0


def test_run_fold_requires_minimum_keys() -> None:
    with pytest.raises(ValueError, match="next_log_returns"):
        run_fold(0, {"is_news_present": np.zeros(10, dtype=bool)}, {})


def test_walk_forward_multi_fold() -> None:
    ctxs = [_toy_ctx(seed=i) for i in range(4)]
    strategies = {
        MODEL_NAME: _const_pos(0.3),
        "buy_hold": _const_pos(1.0),
        "ma_cross": _const_pos(-0.5),
    }
    wf = walk_forward(ctxs, strategies)
    assert len(wf.folds) == 4
    assert wf.n_strategies == 3
    assert wf.cost_multipliers == (0.5, 1.0, 1.5)
    assert set(wf.all_strategy_names()) == {MODEL_NAME, "buy_hold", "ma_cross"}


def test_walk_forward_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="fold_contexts"):
        walk_forward([], {MODEL_NAME: _const_pos(0.0)})
    with pytest.raises(ValueError, match="strategies"):
        walk_forward([_toy_ctx()], {})


def test_render_report_writes_markdown_and_json(tmp_path: Path) -> None:
    ctxs = [_toy_ctx(seed=i) for i in range(2)]
    strategies = {
        MODEL_NAME: _const_pos(0.3),
        "buy_hold": _const_pos(1.0),
    }
    wf = walk_forward(ctxs, strategies)
    md_path = render_report(wf, tmp_path, save_figs=False)
    assert md_path.exists()
    assert md_path.suffix == ".md"
    content = md_path.read_text()
    assert "Promotion Gates" in content
    assert "Cost-Stress Mean Sharpe" in content
    assert "Per-Bucket Mean Sharpe" in content
    assert "Per-Fold Breakdown" in content
    assert "Honest Limitations" in content
    # The 8 gates must all appear.
    for gate_id in (
        "gate_1_sharpe_gt_1_at_1x",
        "gate_2_sharpe_gt_0_5_at_1_5x",
        "gate_3_beats_baseline_3of4",
        "gate_4_conformal_coverage",
        "gate_5_sizer_stage2_beats_stage1",
        "gate_6_drawdown_breaker_2_regimes",
        "gate_7_dsr_gt_1",
        "gate_8_per_bucket_positive",
    ):
        assert gate_id in content
    # JSON sidecar exists and parses.
    json_siblings = list(tmp_path.glob("v1_backtest_*.json"))
    assert len(json_siblings) == 1
    payload = json.loads(json_siblings[0].read_text())
    assert "gates" in payload
    assert "manifest" in payload


def test_render_report_pending_gates_show_pending() -> None:
    """Gates 4-6 require cross-cuts (calibration coverage / sizer A/B /
    regime tagging) that walk_forward cannot compute alone. The report
    must surface them as ``pending`` rather than silently passing them."""
    ctxs = [_toy_ctx()]
    strategies = {MODEL_NAME: _const_pos(0.5)}
    wf = walk_forward(ctxs, strategies)
    md_path = render_report(wf, Path("/tmp/_nanogld_pending_smoke"), save_figs=False)
    content = md_path.read_text()
    assert "pending_verification" in content
