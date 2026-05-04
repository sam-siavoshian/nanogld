"""Investing.com Gold scraper — Source 14.

Aggregator with the largest archive among user-listed sources. Mostly neutral
wire syndication.

Spec: plan/02-DATA-PIPELINE.md "Source 14".
- Bias tier: aggregator_neutral.
- Cloudflare anti-bot — use curl_cffi browser impersonation.
- ToS: research-use grey zone. Throttle ≥ 1 req/3s.

V1 ships a fetch-one-page primitive + a paged scrape skeleton owner runs
explicitly. Live RSS / live cycle uses Alpaca News + Kitco for first-pass.
"""

from __future__ import annotations

import time

import pandas as pd
from curl_cffi import requests as curl_requests

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_investing")

GOLD_NEWS_INDEX = "https://www.investing.com/commodities/gold-news"
BIAS_TIER = "aggregator_neutral"
THROTTLE_SEC = 3


def _session() -> curl_requests.Session:
    return curl_requests.Session(impersonate="chrome")


def fetch_gold_index_html() -> str | None:
    """One-shot fetch of the gold-news index page. Cloudflare-aware via curl_cffi."""
    LOG.info("fetching Investing.com gold-news index")
    try:
        resp = _session().get(GOLD_NEWS_INDEX, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        LOG.warning("Investing.com fetch failed: %s", e)
        return None


def parse_index_articles(html: str) -> list[dict[str, object]]:
    """Best-effort extract of (title, url, pub_ts) tuples from the index page."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, object]] = []
    for art in soup.select("article, div.textDiv, li.js-article-item"):
        a = art.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        url = a["href"]
        if not url.startswith("http"):
            url = "https://www.investing.com" + url
        time_el = art.find("time")
        pub = (
            pd.to_datetime(time_el["datetime"], utc=True, errors="coerce")
            if time_el and time_el.get("datetime")
            else pd.NaT
        )
        rows.append({"title": title, "url": url, "pub_ts": pub})
    return rows


def backfill_pages(max_pages: int = 5) -> pd.DataFrame:
    """Page through the gold-news index. Throttled. Best-effort layout heuristic."""
    sess = _session()
    all_rows: list[dict[str, object]] = []
    for page in range(1, max_pages + 1):
        url = GOLD_NEWS_INDEX if page == 1 else f"{GOLD_NEWS_INDEX}/{page}"
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            LOG.warning("page %d failed: %s", page, e)
            break
        rows = parse_index_articles(resp.text)
        all_rows.extend(rows)
        LOG.info("page %d -> %d items", page, len(rows))
        time.sleep(THROTTLE_SEC)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["article_id"] = df["url"].astype("string")
    df["source"] = pd.Series(["investing_com"] * len(df), dtype="string")
    df["created_at"] = pd.to_datetime(df["pub_ts"], utc=True)
    df = df.dropna(subset=["created_at"])
    df["bias_tier"] = pd.Series([BIAS_TIER] * len(df), dtype="string")
    df["title"] = df["title"].astype("string")
    df["body"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["url"] = df["url"].astype("string")
    df["symbols"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"]
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_investing_parquet(max_pages: int = 5) -> tuple[pd.DataFrame, str]:
    df = backfill_pages(max_pages=max_pages)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "investing_gold_news.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Investing.com rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_investing_parquet()
    print(f"Investing.com gold news: {len(df)} rows -> {p}")
