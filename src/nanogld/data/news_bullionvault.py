"""BullionVault dealer commentary — Source 15. Wayback CDX backfill.

Live scraper selectors don't match the modern layout. Wayback covers it.

Spec: plan/02-DATA-PIPELINE.md "Source 15".
- bias_tier = dealer_bullish (LAFTR head down-weights at inference per doc 03).
- Smaller corpus than Kitco (handful of authors).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
from bs4 import BeautifulSoup

from nanogld.data import wayback_helpers
from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_bullionvault")

BIAS_TIER = "dealer_bullish"
SOURCE_NAME = "bullionvault"
ARTICLE_URL_GLOB = "bullionvault.com/gold-news/*"


def _parse_html(html: bytes) -> dict[str, str | None]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:  # noqa: BLE001
        LOG.warning("BullionVault parse failed: %s", e)
        return {"title": None, "body": None, "pub_ts": None}

    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    body = None
    art = soup.find("article")
    if art:
        body = art.get_text(" ", strip=True)
    if not body:
        paras = soup.find_all("p")
        if paras:
            body = " ".join(p.get_text(" ", strip=True) for p in paras[:30])

    pub_ts = None
    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        pub_ts = meta["content"]
    if not pub_ts:
        t = soup.find("time")
        if t and t.get("datetime"):
            pub_ts = t["datetime"]

    return {"title": title, "body": body, "pub_ts": pub_ts}


def _capture_ts_to_iso(ts: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=UTC))


def backfill_wayback(
    *,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
    cdx_limit: int = 5000,
    polite_sec: float = 2.0,
) -> pd.DataFrame:
    captures = wayback_helpers.cdx_search(ARTICLE_URL_GLOB, start=start, end=end, limit=cdx_limit)
    LOG.info("BullionVault: %d captures from CDX", len(captures))

    rows: list[dict[str, object]] = []
    for capture_ts, url in captures:
        if "/gold-news/" not in url:
            continue
        body = wayback_helpers.fetch_capture(
            capture_ts, url, source=SOURCE_NAME, polite_sec=polite_sec
        )
        if body is None:
            continue
        parsed = _parse_html(body)
        if not parsed["title"]:
            continue
        created_at = pd.to_datetime(parsed["pub_ts"], utc=True, errors="coerce")
        if pd.isna(created_at):
            created_at = _capture_ts_to_iso(capture_ts)
        rows.append(
            {
                "article_id": url,
                "source": SOURCE_NAME,
                "created_at": created_at,
                "title": parsed["title"],
                "body": parsed["body"],
                "url": url,
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
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"] + pd.Timedelta(seconds=60)
    return df[[c.name for c in NEWS_MANIFEST.columns]]


def write_bullionvault_parquet(
    *,
    start: date = date(2021, 4, 24),
    end: date = date(2026, 4, 24),
) -> tuple[pd.DataFrame, str]:
    df = backfill_wayback(start=start, end=end)
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "bullionvault_news.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d BullionVault rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_bullionvault_parquet()
    print(f"BullionVault: {len(df)} rows -> {p}")
