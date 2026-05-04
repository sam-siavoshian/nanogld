"""nanoGLD data pipeline.

Pulls 5y of 30min GLD bars + ETF basket + Alpaca News (Benzinga) + GDELT GKG +
FRED ALFRED + yfinance Brent/WTI + GPR + CFTC COT + WGC + calendar events +
multi-source news scrapes. Joins with strict point-in-time discipline. Writes
immutable hashed parquet snapshots consumed by `nanogld.features` (doc 04).

Spec: plan/02-DATA-PIPELINE.md.
"""

from nanogld.data import schema, snapshot, utils

__all__ = ["schema", "snapshot", "utils"]
