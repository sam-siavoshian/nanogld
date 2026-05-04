"""BullionVault author-pages scraper — Source 15.

Bullish dealer marketing. Use as bias-extreme feature, NOT raw signal.
LAFTR will heavily down-weight in inference.

Spec: plan/02-DATA-PIPELINE.md "Source 15".
- Bias tier: dealer_bullish.
- No news API exists.
- robots FAQ: bullionvault.com/help/FAQs/FAQs_bots.html — review before scraping.

V1 ships a fetch-author-page primitive owner runs explicitly. Light V1 scope
(~500MB worth of articles).
"""

from __future__ import annotations

import time
from urllib.parse import urljoin

import pandas as pd
import requests

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_bullionvault")

BV_HOME = "https://www.bullionvault.com"
BV_AUTHOR_PAGES = (
    "https://www.bullionvault.com/gold-news/users/adrian-ash",
    "https://www.bullionvault.com/gold-news/users/gold-report",
)
BIAS_TIER = "dealer_bullish"
THROTTLE_SEC = 3


def fetch_author_page(url: str) -> pd.DataFrame:
    """Single author archive page. Returns articles with (title, url, pub_ts)."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    LOG.info("fetching BullionVault: %s", url)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "nanoGLD/0.1"})
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        LOG.warning("BullionVault fetch failed (%s): %s", url, e)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows: list[dict[str, object]] = []
    for art in soup.select("article, div.story, li.article"):
        a = art.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        href = urljoin(BV_HOME, a["href"])
        time_el = art.find("time")
        pub = (
            pd.to_datetime(time_el.get("datetime"), utc=True, errors="coerce")
            if time_el and time_el.get("datetime")
            else pd.NaT
        )
        rows.append({"article_id": href, "title": title, "url": href, "pub_ts": pub})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_all_authors() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url in BV_AUTHOR_PAGES:
        frames.append(fetch_author_page(url))
        time.sleep(THROTTLE_SEC)
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        return df

    df["source"] = pd.Series(["bullionvault"] * len(df), dtype="string")
    df["created_at"] = pd.to_datetime(df["pub_ts"], utc=True)
    df = df.dropna(subset=["created_at"])
    df["bias_tier"] = pd.Series([BIAS_TIER] * len(df), dtype="string")
    df["title"] = df["title"].astype("string")
    df["body"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["url"] = df["url"].astype("string")
    df["article_id"] = df["article_id"].astype("string")
    df["symbols"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"]
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_bullionvault_parquet() -> tuple[pd.DataFrame, str]:
    df = fetch_all_authors()
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
