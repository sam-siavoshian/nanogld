"""V1 baselines for backtest comparison."""

from nanogld.backtest.baselines.buy_hold import buy_hold_positions
from nanogld.backtest.baselines.donchian import donchian_positions
from nanogld.backtest.baselines.gao_2014 import gao_2014_positions
from nanogld.backtest.baselines.ma_cross import ema, ma_cross_positions

__all__ = [
    "buy_hold_positions",
    "donchian_positions",
    "ema",
    "gao_2014_positions",
    "ma_cross_positions",
]
