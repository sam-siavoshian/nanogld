"""Multi-dim sentiment features — placeholder (doc 04 §V1 update + §5).

Per arXiv:2603.11408 (March 2026), gold price sentiment is best captured by
three dimensions, not a single polarity scalar:
  polarity     bullish / bearish (-1..1)
  intensity    magnitude of conviction (0..1)
  uncertainty  hedging language (0..1)

These are extracted via per-article LLM prompts on top of the V4 news
aggregator (doc 03). doc 03 owns the embedding pipeline + LAFTR adversary;
this module will join its output into the bar-level feature stream once
the embedding cache lands.

V1 stub: returns an empty frame with a `t_visible` column so build.py can
import the module unconditionally and skip gracefully when no embedding
cache exists yet.
"""

from __future__ import annotations

import pandas as pd

from nanogld.data.utils import get_logger
from nanogld.features.utils import FeatureSpec

LOG = get_logger("nanogld.features.sentiment")

# Reserved column names for downstream consumers — populated when doc 03
# embedding cache lands.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("sentiment_polarity_alpaca", source="news"),
    FeatureSpec("sentiment_intensity_alpaca", source="news"),
    FeatureSpec("sentiment_uncertainty_alpaca", source="news"),
)


def build_sentiment_features() -> pd.DataFrame:
    """Stub — returns empty frame with t_visible column.

    Wire-in happens when doc 03 publishes the news embedding cache + the
    multi-dim sentiment extractor. Until then build.py treats this group
    as "missing source" (fail-soft).
    """
    LOG.info("sentiment features stub — depends on doc 03 (news embeddings)")
    return pd.DataFrame({"t_visible": pd.Series(dtype="datetime64[ns, UTC]")})
