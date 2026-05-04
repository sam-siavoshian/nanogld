"""Alpaca News API (Benzinga only) — Source 2.

Spec hard rules (plan/02-DATA-PIPELINE.md "Source 2"):
- Single source: Benzinga firehose. NOT Reuters (Reuters paywalled 2024).
- Field is `created_at`, NOT `published_at` (does not exist).
- NEVER join on `updated_at` (drifts forward on edits — guaranteed leak).
- t_visible = created_at + 60s (wire-clock skew safety, NEWS_LATENCY_SEC_ALPACA).
- NewsClient REQUIRES api keys despite stale PyPI docs claiming otherwise.
- 50 articles per page max — drain all next_page_tokens.
- Historical depth back to 2015. Symbol-filtered queries work for GLD.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import (
    END_DATE_NAIVE,
    NEWS_LATENCY_SEC_ALPACA,
    START_DATE_NAIVE,
    get_logger,
    raw_dir,
)

LOG = get_logger("nanogld.data.alpaca_news")

DEFAULT_SYMBOL = "GLD"
BIAS_TIER = "mainstream_neutral"  # Benzinga = mainstream wire (see doc 03 SOURCE_REGISTRY)


def _client() -> NewsClient:
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_API_SECRET")
    if not key or not sec or "FILL_ME" in str(key):
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_API_SECRET missing — populate ~/.config/nanogld/.env.paper"
        )
    # NewsClient REQUIRES keys (V4 verified — stale PyPI docs claim otherwise)
    return NewsClient(key, sec)


def fetch_news(
    symbol: str = DEFAULT_SYMBOL,
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
) -> pd.DataFrame:
    """Pull 5y of Benzinga news for `symbol`. SDK auto-paginates via next_page_token."""
    LOG.info("fetching Alpaca News for %s, %s -> %s", symbol, start, end)
    req = NewsRequest(
        symbols=symbol,
        start=start,
        end=end,
        include_content=True,
        exclude_contentless=True,
        limit=50,  # max per page (V4)
    )
    raw = _client().get_news(req)
    df = raw.df.reset_index()
    if df.empty:
        LOG.warning("Alpaca News returned 0 rows for %s", symbol)
        return pd.DataFrame()

    # Alpaca News fields: id, headline, summary, content, author,
    # created_at, updated_at, url, symbols, source.
    if "id" in df.columns:
        df["article_id"] = df["id"].astype("string")
    elif "news_id" in df.columns:
        df["article_id"] = df["news_id"].astype("string")
    else:
        df["article_id"] = df.index.astype(str)
        df["article_id"] = df["article_id"].astype("string")

    df["source"] = pd.Series(["alpaca_benzinga"] * len(df), dtype="string")
    df["bias_tier"] = pd.Series([BIAS_TIER] * len(df), dtype="string")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # title + body — Alpaca News has 'headline' + 'content'/'summary'
    df["title"] = df.get("headline", df.get("title", "")).astype("string")
    body = df.get("content", df.get("summary", pd.Series([None] * len(df))))
    df["body"] = body.astype("string")
    df["url"] = df.get("url", pd.Series([None] * len(df))).astype("string")
    df["symbols"] = (
        df.get("symbols", pd.Series([None] * len(df)))
        .apply(lambda v: "|".join(v) if isinstance(v, list) else (v if v else None))
        .astype("string")
    )

    # release_ts = created_at; t_visible = created_at + 60s buffer
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"] + pd.Timedelta(seconds=NEWS_LATENCY_SEC_ALPACA)

    out = (
        df[[c.name for c in NEWS_MANIFEST.columns]].sort_values("created_at").reset_index(drop=True)
    )
    return out


def write_news_parquet(
    symbol: str = DEFAULT_SYMBOL,
    *,
    start: datetime = START_DATE_NAIVE,
    end: datetime = END_DATE_NAIVE,
) -> tuple[pd.DataFrame, str]:
    df = fetch_news(symbol, start=start, end=end)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / f"alpaca_news_{symbol}.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Alpaca News rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_news_parquet()
    print(f"Alpaca News: {len(df)} rows -> {p}")
