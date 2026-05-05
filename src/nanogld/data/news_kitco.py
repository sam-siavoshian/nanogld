"""Kitco News — Source 13. Wayback CDX backfill (RSS endpoint serves HTML, broken).

Spec: plan/02-DATA-PIPELINE.md "Source 13".

Strategy as of 2026-05:
- Live RSS (`kitco.com/news/category/<cat>/rss`) currently returns the HTML
  page, NOT RSS XML — verified failing on direct curl. RSS path retained
  as a stub in case Kitco fixes it; default backfill route is Wayback CDX.
- Wayback CDX scrape of kitco.com/news/article/* for the V1 window
  (2021-04-24 → 2026-04-24).
- bias_tier = industry_bullish (LAFTR head learns the per-source prior in doc 03).
- robots.txt allowed Internet Archive at original capture time, so this is
  legal scraping of an existing public archive.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

import pandas as pd
from bs4 import BeautifulSoup

from nanogld.data import wayback_helpers
from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_kitco")

BIAS_TIER = "industry_bullish"
SOURCE_NAME = "kitco"
ARTICLE_URL_GLOB = "kitco.com/news/*"

# Kitco URL slug pattern: /news/article/2024-12-15/<slug>
SLUG_DATE_RE = re.compile(r"/news/article/(\d{4}-\d{2}-\d{2})/")


def _parse_kitco_html(html: bytes) -> dict[str, str | None]:
    """Best-effort extract of (title, body, pub_ts) from a Kitco article page."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:  # noqa: BLE001
        LOG.warning("Kitco HTML parse failed: %s", e)
        return {"title": None, "body": None, "pub_ts": None}

    # Title — h1, og:title, or article-title class
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    # Body — try article tag, then large paragraph blocks
    body = None
    art = soup.find("article")
    if art:
        body = art.get_text(" ", strip=True)
    if not body:
        paras = soup.find_all("p")
        if paras:
            body = " ".join(p.get_text(" ", strip=True) for p in paras[:30])

    # pub_ts — <time datetime="..."> or article:published_time meta tag
    pub_ts = None
    t = soup.find("time")
    if t and t.get("datetime"):
        pub_ts = t["datetime"]
    if not pub_ts:
        meta = soup.find("meta", property="article:published_time")
        if meta and meta.get("content"):
            pub_ts = meta["content"]

    return {"title": title, "body": body, "pub_ts": pub_ts}


def _extract_pub_from_url(url: str) -> str | None:
    """Fallback: pull YYYY-MM-DD from the article slug."""
    m = SLUG_DATE_RE.search(url)
    return m.group(1) if m else None


def _capture_ts_to_iso(ts_yyyymmddhhmmss: str) -> pd.Timestamp:
    """Wayback timestamp → UTC pd.Timestamp."""
    dt = datetime.strptime(ts_yyyymmddhhmmss, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return pd.Timestamp(dt)


def backfill_wayback(
    *,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
    cdx_limit: int = 10000,
    polite_sec: float = 2.0,
) -> pd.DataFrame:
    """Use Wayback CDX to enumerate Kitco news article URLs and fetch each.

    Returns NEWS_MANIFEST-conformant DataFrame (may be partial if Wayback
    halts mid-soak — cache resumes on re-run).
    """
    captures = wayback_helpers.cdx_search(ARTICLE_URL_GLOB, start=start, end=end, limit=cdx_limit)
    LOG.info("Kitco: %d unique URL captures from CDX", len(captures))

    rows: list[dict[str, object]] = []
    for capture_ts, original_url in captures:
        # Only keep article-detail URLs, not category landing pages
        if "/news/article/" not in original_url:
            continue
        body_bytes = wayback_helpers.fetch_capture(
            capture_ts, original_url, source=SOURCE_NAME, polite_sec=polite_sec
        )
        if body_bytes is None:
            continue
        parsed = _parse_kitco_html(body_bytes)
        if not parsed["title"]:
            continue

        # pub_ts: HTML metadata > URL slug > capture timestamp (worst case)
        pub = parsed["pub_ts"]
        if not pub:
            slug_date = _extract_pub_from_url(original_url)
            if slug_date:
                pub = f"{slug_date}T00:00:00Z"
        created_at = pd.to_datetime(pub, utc=True, errors="coerce")
        if pd.isna(created_at):
            created_at = _capture_ts_to_iso(capture_ts)

        rows.append(
            {
                "article_id": original_url,
                "source": SOURCE_NAME,
                "created_at": created_at,
                "title": parsed["title"],
                "body": parsed["body"],
                "url": original_url,
                "symbols": pd.NA,
                "bias_tier": BIAS_TIER,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for c in ("article_id", "source", "title", "body", "url", "bias_tier"):
        df[c] = df[c].astype("string")
    df["symbols"] = df["symbols"].astype("string")
    df = df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    # 60s wire-clock skew buffer (matches Alpaca News convention)
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"] + pd.Timedelta(seconds=60)
    return df[[c.name for c in NEWS_MANIFEST.columns]]


def fetch_recent_rss(window_min: int = 1440) -> pd.DataFrame:
    """STUB — Kitco's /rss endpoints currently serve HTML not XML (verified
    2026-05). Returns empty until they fix the feed. Live cycle should rely
    on the Wayback backfill plus daily cron-driven fetches of the article
    index page.
    """
    LOG.info("Kitco RSS skipped — endpoint serves HTML not XML as of 2026-05")
    return pd.DataFrame()


def write_kitco_parquet(
    *,
    use_wayback: bool = True,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
) -> tuple[pd.DataFrame, str]:
    """Default: full Wayback backfill. Owner can disable for a quick smoke run."""
    df = backfill_wayback(start=start, end=end) if use_wayback else fetch_recent_rss()
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "kitco_news.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Kitco rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_kitco_parquet()
    print(f"Kitco: {len(df)} rows -> {p}")
