"""Group the 681 numeric features into semantic categories.

Categories (V1 invariant: every feature must belong to exactly one):
    - price        : OHLC + log-return + h5 family
    - volatility   : ATR, RV, realized vol, GARCH residuals
    - macro        : DFF, DGS10, VIX, FRED + ALFRED releases
    - calendar     : FOMC weeks, NFP-week, CPI-day, options-expiry
    - regime       : HMM posterior + tercile one-hots
    - news         : daily aggregated sentiment / volume (legacy lane)
    - flow         : cross-market spreads (SPY, GLDvSLV, etc)
    - rates        : yield-curve diffs, real-yield, breakevens
    - other        : anything that doesn't match the above

Falls back to substring match on standardized lowercase names. The
result is a dict {feature_name: category}. The function is pure: no
file I/O, no mutable global state.

Spec: plan/V1-SPEC.md §11.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CATEGORIES: tuple[str, ...] = (
    "price",
    "volatility",
    "macro",
    "calendar",
    "regime",
    "news",
    "flow",
    "rates",
    "other",
)


_PRICE_TOKENS = ("close", "open", "high", "low", "log_return", "h5", "ret_", "tr_")
_VOL_TOKENS = ("atr", "rv_", "realized_vol", "garch", "vol_", "stdev", "_vol")
_MACRO_TOKENS = ("dff", "dgs", "vix", "tnx", "fred", "alfred", "cpi_", "ppi_", "nfp")
_CALENDAR_TOKENS = ("fomc", "is_fomc", "is_nfp", "is_cpi", "opex", "is_opex", "session")
_REGIME_TOKENS = ("regime", "tercile", "hmm", "era_")
_NEWS_TOKENS = ("news_", "sentiment", "headline", "article")
_FLOW_TOKENS = ("spy_", "qqq_", "iwm_", "gldvslv", "_spread", "ratio")
_RATES_TOKENS = ("yield", "real_", "breakeven", "_curve", "_2y", "_10y", "_30y")


def _classify_one(name: str) -> str:
    """Map a single feature name to its category.

    Order matters: more-specific token sets are checked first so that a
    name like `gld_atr_14` is classified as "volatility" before falling
    through to the broader "price" check (which contains `tr_`).
    """
    lower = name.lower()
    if any(t in lower for t in _REGIME_TOKENS):
        return "regime"
    if any(t in lower for t in _CALENDAR_TOKENS):
        return "calendar"
    if any(t in lower for t in _NEWS_TOKENS):
        return "news"
    if any(t in lower for t in _RATES_TOKENS):
        return "rates"
    if any(t in lower for t in _FLOW_TOKENS):
        return "flow"
    if any(t in lower for t in _VOL_TOKENS):
        return "volatility"
    if any(t in lower for t in _MACRO_TOKENS):
        return "macro"
    if any(t in lower for t in _PRICE_TOKENS):
        return "price"
    return "other"


def classify_features(names: Iterable[str]) -> dict[str, str]:
    """Return {name: category} for an iterable of feature names."""
    return {name: _classify_one(name) for name in names}


@dataclass(frozen=True)
class GroupRollup:
    """Aggregate importance per category, sorted desc by absolute mean."""

    category: str
    n_features: int
    mean_abs_importance: float
    sum_abs_importance: float
    top_feature: str
    top_value: float


def rollup_by_group(
    names: list[str],
    importance: list[float],
) -> list[GroupRollup]:
    """Aggregate per-feature importance into per-category rollups.

    Args:
        names: feature names parallel to `importance`.
        importance: scalar per-feature importance score (any sign).

    Returns:
        List of GroupRollup sorted by sum_abs_importance descending.
    """
    if len(names) != len(importance):
        raise ValueError(f"names {len(names)} != importance {len(importance)}")
    cat_of = classify_features(names)
    buckets: dict[str, list[tuple[str, float]]] = {c: [] for c in CATEGORIES}
    for nm, val in zip(names, importance, strict=False):
        buckets[cat_of[nm]].append((nm, float(val)))

    rollups: list[GroupRollup] = []
    for cat, items in buckets.items():
        if not items:
            continue
        abs_vals = [abs(v) for _, v in items]
        sum_abs = sum(abs_vals)
        mean_abs = sum_abs / len(abs_vals)
        top_nm, top_val = max(items, key=lambda kv: abs(kv[1]))
        rollups.append(
            GroupRollup(
                category=cat,
                n_features=len(items),
                mean_abs_importance=mean_abs,
                sum_abs_importance=sum_abs,
                top_feature=top_nm,
                top_value=top_val,
            )
        )
    rollups.sort(key=lambda r: r.sum_abs_importance, reverse=True)
    return rollups
