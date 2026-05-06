"""nanoGLD feature engineering.

Phase F (2026-05-04, ahead of doc 04 spec): builds a PIT-clean daily macro
feature panel from the keyless raw sources (Brent/WTI/COT/GPR/calendar).
The panel is keyed by `date_utc` and every row carries `t_visible` so that,
once Alpaca GLD 30min bars land, the joiner forward-fills these features
into each bar via strict-< asof.

V1 dataset expansion (2026-05-05): added 8 modules implementing doc 04
§§1-3, 6-9, 11. Daily-frequency modules (macro, treasury, macro_bundle,
wgc) plug directly into the daily panel. Bar-frequency modules (price,
risk, equity) consume GLD 30min bars and are wired in by the joiner via
strict-< asof on t_visible. sentiment.py is a stub placeholder until
doc 03 publishes the news embedding cache.

Spec: plan/04-FEATURE-ENGINEERING.md.
"""

from nanogld.features import (
    build,
    calendar_features,
    equity,
    geopolitical,
    macro,
    macro_bundle,
    oil,
    positioning,
    price,
    risk,
    sentiment,
    treasury,
    utils,
    wgc,
)

__all__ = [
    "build",
    "calendar_features",
    "equity",
    "geopolitical",
    "macro",
    "macro_bundle",
    "oil",
    "positioning",
    "price",
    "risk",
    "sentiment",
    "treasury",
    "utils",
    "wgc",
]
