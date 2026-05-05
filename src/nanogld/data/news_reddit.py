"""Reddit retail sentiment — HF mirror via DuckDB (no torrent).

Replaces the prior torrent-only path. The HF dataset `open-index/arctic`
mirrors the Pushshift Parquet dumps with predicate pushdown, so we can
query the slice we want without downloading 1.1 TB.

Spec: plan/02-DATA-PIPELINE.md "Source 17". V4-corrected.

Subreddits: Gold / wallstreetbets / investing / Goldandsilverstackers / Commodities.
Keyword filter: gold|GLD|silver|SLV|gdx|comex|xau|bullion|miner.

bias_tier = retail_social. Often counter-indicator at extremes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_reddit")

ARCTIC_HF = "hf://datasets/open-index/arctic"
SUBREDDITS = ("Gold", "wallstreetbets", "investing", "Goldandsilverstackers", "Commodities")
KEYWORD_REGEX = r"\b(gold|GLD|silver|SLV|gdx|comex|xau|bullion|miner)\b"
BIAS_TIER = "retail_social"


def query_arctic(
    *,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
    subreddits: tuple[str, ...] = SUBREDDITS,
    keyword_regex: str = KEYWORD_REGEX,
    posts_only: bool = True,
) -> pd.DataFrame:
    """Use DuckDB predicate pushdown over the HF arctic Parquet mirror.

    Returns NEWS_MANIFEST-conformant rows.
    """
    import duckdb  # noqa: PLC0415

    sub_list = ",".join(f"'{s}'" for s in subreddits)
    table_root = "RS" if posts_only else "RC"  # RS = submissions, RC = comments
    start_unix = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_unix = int(datetime.combine(end, datetime.min.time(), tzinfo=UTC).timestamp())

    sql = f"""
    SELECT
        id,
        subreddit,
        created_utc,
        title,
        selftext,
        url,
        permalink
    FROM read_parquet('{ARCTIC_HF}/reddit/{table_root}/**/*.parquet', union_by_name=true)
    WHERE subreddit IN ({sub_list})
      AND created_utc BETWEEN {start_unix} AND {end_unix}
      AND (
        regexp_matches(coalesce(title, ''), '{keyword_regex}', 'i')
        OR regexp_matches(coalesce(selftext, ''), '{keyword_regex}', 'i')
      )
    """
    LOG.info("DuckDB arctic query: %s", sql.strip().split("WHERE")[0])
    try:
        df = duckdb.sql(sql).df()
    except Exception as e:  # noqa: BLE001
        LOG.warning("arctic DuckDB query failed: %s", e)
        return pd.DataFrame()

    if df.empty:
        LOG.info("arctic returned 0 rows for filter")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["article_id"] = (df["id"].astype(str) + "_" + df["subreddit"].astype(str)).astype("string")
    out["source"] = ("reddit_" + df["subreddit"].astype(str).str.lower()).astype("string")
    out["created_at"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    out["title"] = df["title"].fillna("").astype("string")
    out["body"] = df["selftext"].fillna("").astype("string")
    out["url"] = df["url"].fillna("").astype("string")
    out["symbols"] = pd.array([pd.NA] * len(df), dtype="string")
    out["bias_tier"] = pd.array([BIAS_TIER] * len(df), dtype="string")
    out = out.dropna(subset=["created_at"])
    out["release_ts"] = out["created_at"]
    out["t_visible"] = out["created_at"]
    return out[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_reddit_parquet(
    *,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
) -> tuple[pd.DataFrame, str]:
    df = query_arctic(start=start, end=end)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "reddit_gold_filtered.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Reddit rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_reddit_parquet()
    print(f"Reddit (arctic): {len(df)} rows -> {p}")
