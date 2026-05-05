"""nanoGLD feature engineering.

Phase F (2026-05-04, ahead of doc 04 spec): builds a PIT-clean daily macro
feature panel from the keyless raw sources (Brent/WTI/COT/GPR/calendar).
The panel is keyed by `date_utc` and every row carries `t_visible` so that,
once Alpaca GLD 30min bars land, the joiner forward-fills these features
into each bar via strict-< asof.

Spec (eventual home): plan/04-FEATURE-ENGINEERING.md.
"""

from nanogld.features import (
    build,
    calendar_features,
    geopolitical,
    oil,
    positioning,
    utils,
)

__all__ = ["build", "calendar_features", "geopolitical", "oil", "positioning", "utils"]
