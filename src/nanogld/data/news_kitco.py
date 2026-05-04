"""Kitco News scraper — Source 13.

Free site, no API. Date-slugged URLs make 10y backfill feasible.

Spec: plan/02-DATA-PIPELINE.md "Source 13".
- Bias tier: industry_bullish (LAFTR head learns the prior in doc 03).
- Throttle 1 req/2s minimum.
- robots.txt allows crawl as of V4 audit; respect ToS.
- RSS feeds for live cycle: kitco.com/news/category/{markets,mining,commodities}/rss

V1 implementation: RSS-only for live cycle. Historical archive crawl is a
multi-day soak that owner triggers explicitly via `python -m nanogld.data.news_kitco
backfill --start 2021-04-24 --end 2026-04-24`. The crawler is intentionally
conservative — small batches, long pauses, journaled state.
"""

from __future__ import annotations

import time
from datetime import datetime
from urllib.parse import urljoin

import feedparser
import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import UTC, get_logger, raw_dir

LOG = get_logger("nanogld.data.news_kitco")

KITCO_RSS = {
    "markets": "https://www.kitco.com/news/category/markets/rss",
    "mining": "https://www.kitco.com/news/category/mining/rss",
    "commodities": "https://www.kitco.com/news/category/commodities/rss",
}
KITCO_HOME = "https://www.kitco.com"
BIAS_TIER = "industry_bullish"
THROTTLE_SEC = 2


def fetch_rss(window_min: int = 1440) -> pd.DataFrame:
    """Pull recent items from Kitco RSS feeds. Live-cycle slice (default 24h).

    For deep historical backfill, see backfill_archive() (owner runs explicitly).
    """
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(minutes=window_min)
    rows: list[dict[str, object]] = []
    for category, url in KITCO_RSS.items():
        LOG.info("fetching Kitco RSS: %s", url)
        try:
            feed = feedparser.parse(url)
        except Exception as e:  # noqa: BLE001
            LOG.warning("Kitco RSS parse failed (%s): %s", category, e)
            continue
        for entry in feed.entries:
            try:
                pub = pd.to_datetime(getattr(entry, "published", None), utc=True, errors="coerce")
            except Exception:  # noqa: BLE001
                pub = pd.NaT
            if pd.isna(pub) or pub < cutoff:
                continue
            rows.append(
                {
                    "article_id": str(getattr(entry, "id", entry.link)),
                    "source": f"kitco_{category}",
                    "created_at": pub,
                    "title": str(getattr(entry, "title", "")),
                    "body": str(getattr(entry, "summary", "") or ""),
                    "url": urljoin(KITCO_HOME, getattr(entry, "link", "")),
                    "symbols": pd.NA,
                }
            )
        time.sleep(THROTTLE_SEC)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["bias_tier"] = pd.Series([BIAS_TIER] * len(df), dtype="string")
    for c in ("article_id", "source", "title", "body", "url"):
        df[c] = df[c].astype("string")
    df["symbols"] = df["symbols"].astype("string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"]
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def backfill_archive(start: datetime, end: datetime) -> pd.DataFrame:
    """Owner-triggered historical scrape. Skeleton — full implementation defers
    to a long-running soak job (multi-day) with journaled state to avoid
    hammering the site after a crash.
    """
    LOG.warning(
        "backfill_archive(%s, %s) skeleton — implement as journaled multi-day soak before V1 train",
        start,
        end,
    )
    return pd.DataFrame()


def write_kitco_rss_parquet(window_min: int = 1440) -> tuple[pd.DataFrame, str]:
    df = fetch_rss(window_min=window_min)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "kitco_news_recent.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Kitco rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_kitco_rss_parquet()
    print(f"Kitco RSS recent: {len(df)} rows -> {p}")
