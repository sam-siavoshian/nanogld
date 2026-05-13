"""V1 baselines for backtest comparison.

Two flavors of contract:

- **Price-only baselines** (``buy_hold_positions``, ``ma_cross_positions``,
  ``donchian_positions``, ``gao_2014_positions``): take raw price /
  feature arrays directly. Used both standalone and via the CLI's
  default-strategy wrappers.
- **Context baselines** (``xgboost_positions``, ``dlinear_positions``,
  ``tsmixer_positions``, ``timemixer_positions``,
  ``xlstm_time_positions``, ``vlstm_positions``,
  ``forecast_to_fill_positions``): take the walk-forward fold context
  dict. They train on ``ctx["train_*"]`` slices and emit positions for
  the test slice. Fall back to zeros if the train arrays are absent
  (dry-run / harness smoke path).
"""

from nanogld.backtest.baselines.buy_hold import buy_hold_positions
from nanogld.backtest.baselines.dlinear import dlinear_positions
from nanogld.backtest.baselines.donchian import donchian_positions
from nanogld.backtest.baselines.forecast_to_fill import forecast_to_fill_positions
from nanogld.backtest.baselines.gao_2014 import gao_2014_positions
from nanogld.backtest.baselines.ma_cross import ema, ma_cross_positions
from nanogld.backtest.baselines.timemixer import timemixer_positions
from nanogld.backtest.baselines.tsmixer import tsmixer_positions
from nanogld.backtest.baselines.vlstm import vlstm_positions
from nanogld.backtest.baselines.xgboost_baseline import xgboost_positions
from nanogld.backtest.baselines.xlstm_time import xlstm_time_positions

__all__ = [
    "buy_hold_positions",
    "dlinear_positions",
    "donchian_positions",
    "ema",
    "forecast_to_fill_positions",
    "gao_2014_positions",
    "ma_cross_positions",
    "timemixer_positions",
    "tsmixer_positions",
    "vlstm_positions",
    "xgboost_positions",
    "xlstm_time_positions",
]
