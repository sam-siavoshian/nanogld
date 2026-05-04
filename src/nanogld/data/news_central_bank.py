"""Central-bank speeches + government press releases — Source 16.

Highest-impact news class for gold (Fed/ECB/BIS rate-setting language drives
the real-rate path that drives gold). Public domain (US 17 USC §105 + ECB
free research).

Spec: plan/02-DATA-PIPELINE.md "Source 16".
- Bias tiers: central_bank_official (Fed/ECB/BIS) | government_official (Treasury/CFTC).
- HF datasets:
  - samchain/bis_central_bank_speeches (1997-2023+, 90+ banks)
  - istat-ai/ECB-FED-speeches (1996-2025, 30 MB)
- FOMC statements: github.com/fomc/statements (cleaner than scraping Fed site).
- Treasury press releases: home.treasury.gov/news/press-releases (paginated).
- CFTC speeches: cftc.gov/PressRoom/SpeechesTestimony.

V1 ships HF dataset loaders; live-fetch scrapers are skeletons owner runs
periodically (1 req/3s throttle).
"""

from __future__ import annotations

import os
import time
from urllib.parse import urljoin

import pandas as pd
import requests

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_central_bank")

HF_BIS = "samchain/bis_central_bank_speeches"
HF_ECB_FED = "istat-ai/ECB-FED-speeches"

TREASURY_INDEX = "https://home.treasury.gov/news/press-releases"
CFTC_SPEECHES = "https://www.cftc.gov/PressRoom/SpeechesTestimony"
THROTTLE_SEC = 3


def _ensure_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok if tok and "FILL_ME" not in str(tok) else None


def _hf_to_news(records: list[dict], source: str, bias_tier: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["article_id"] = df.get("id", df.index.astype(str)).astype("string")
    df["source"] = pd.Series([source] * len(df), dtype="string")
    df["created_at"] = pd.to_datetime(
        df.get("date", df.get("published_at", df.get("speech_date"))), utc=True, errors="coerce"
    )
    df = df.dropna(subset=["created_at"])
    df["title"] = df.get("title", df.get("speech_title", "")).astype("string")
    df["body"] = df.get("body", df.get("text", df.get("content", ""))).astype("string")
    df["url"] = df.get("url", pd.Series([pd.NA] * len(df))).astype("string")
    df["symbols"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["bias_tier"] = pd.Series([bias_tier] * len(df), dtype="string")
    df["release_ts"] = df["created_at"]
    df["t_visible"] = df["created_at"]
    return df[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def fetch_bis_speeches() -> pd.DataFrame:
    """HF samchain/bis_central_bank_speeches (1997-2023, 90+ banks)."""
    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("loading BIS speeches HF dataset")
    ds = load_dataset(HF_BIS, split="train", token=_ensure_token())
    return _hf_to_news(list(ds), "bis_speeches", "central_bank_official")


def fetch_ecb_fed_speeches() -> pd.DataFrame:
    """HF istat-ai/ECB-FED-speeches (1996-2025, 30 MB)."""
    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("loading ECB+FED speeches HF dataset")
    ds = load_dataset(HF_ECB_FED, split="train", token=_ensure_token())
    return _hf_to_news(list(ds), "ecb_fed_speeches", "central_bank_official")


def _scrape_index_page(url: str, source: str, bias_tier: str) -> pd.DataFrame:
    """Treasury / CFTC index page best-effort scrape."""
    from bs4 import BeautifulSoup  # noqa: PLC0415

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "nanoGLD/0.1"})
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        LOG.warning("%s fetch failed: %s", source, e)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows: list[dict[str, object]] = []
    for art in soup.select("article, div.views-row, li.views-row, div.press-release"):
        a = art.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a["href"]
        if href and not href.startswith("http"):
            href = urljoin(url, href)
        time_el = art.find("time")
        pub = (
            pd.to_datetime(time_el.get("datetime"), utc=True, errors="coerce")
            if time_el and time_el.get("datetime")
            else pd.NaT
        )
        rows.append({"id": href, "title": title, "url": href, "date": pub})
    return _hf_to_news(rows, source, bias_tier)


def fetch_treasury_press_releases() -> pd.DataFrame:
    """home.treasury.gov press releases (single-page snapshot for V1)."""
    return _scrape_index_page(TREASURY_INDEX, "treasury_press", "government_official")


def fetch_cftc_speeches() -> pd.DataFrame:
    """cftc.gov SpeechesTestimony (single-page snapshot for V1)."""
    return _scrape_index_page(CFTC_SPEECHES, "cftc_speeches", "government_official")


def write_central_bank_parquet() -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    for fn, label in [
        (fetch_bis_speeches, "BIS"),
        (fetch_ecb_fed_speeches, "ECB+FED"),
        (fetch_treasury_press_releases, "Treasury"),
        (fetch_cftc_speeches, "CFTC"),
    ]:
        try:
            df = fn()
            LOG.info("%s -> %d rows", label, len(df))
            if not df.empty:
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            LOG.warning("%s fetch raised: %s", label, e)
        time.sleep(THROTTLE_SEC)

    if not frames:
        return pd.DataFrame(), ""
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["created_at", "title"])
    df = df.drop_duplicates(subset=["article_id", "source"])
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "central_bank_news.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d central-bank rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_central_bank_parquet()
    print(f"central-bank news: {len(df)} rows -> {p}")
