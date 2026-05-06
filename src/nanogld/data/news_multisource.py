"""Multi-source bulk historical news — Source 12B.

HF dataset `Brianferrell787/financial-news-multisource` — 57.1M rows
1990-2025, includes Reuters / Bloomberg / Benzinga / NASDAQ subsets.

License: NON-COMMERCIAL (Other / research-only). Gated behind
NANOGLD_NONCOMMERCIAL=1 env flag (default off). Owner explicitly opts in
for personal V1 research training.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import NEWS_LATENCY_SEC_ALPACA, get_logger, raw_dir

LOG = get_logger("nanogld.data.news_multisource")

DATASET = "Brianferrell787/financial-news-multisource"
GOLD_RELEVANT_TICKERS = frozenset(
    ["GLD", "GDX", "SLV", "IAU", "NEM", "GOLD", "FNV", "AEM", "TLT", "IEF", "UUP"]
)
BIAS_BY_SOURCE = {
    "reuters": "mainstream_neutral",
    "bloomberg": "mainstream_neutral",
    "benzinga": "mainstream_neutral",
    "nasdaq": "mainstream_neutral",
    "marketwatch": "mainstream_neutral",
    "wsj": "mainstream_neutral",
    "ft": "mainstream_neutral",
    "cnbc": "mainstream_neutral",
    "seeking_alpha": "retail_pundit",
    "motley_fool": "retail_pundit",
    "yahoo": "aggregator_neutral",
}
DEFAULT_BIAS = "aggregator_neutral"


def _gate_open() -> bool:
    if os.environ.get("NANOGLD_NONCOMMERCIAL") != "1":
        LOG.warning(
            "financial-news-multisource skipped — license is non-commercial. "
            "Set NANOGLD_NONCOMMERCIAL=1 to enable for personal / research training."
        )
        return False
    return True


def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok if tok and "FILL_ME" not in str(tok) else None


def fetch_filtered() -> pd.DataFrame:
    if not _gate_open():
        return pd.DataFrame()

    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("streaming %s (gold-relevant filter inline)", DATASET)
    # Stream mode: never materializes the 28+ GB arrow cache. Iterate + filter
    # to ~50-200 MB in RAM. Fail-safe for low-disk machines (Errno 28).
    ds_iter = load_dataset(DATASET, split="train", token=_hf_token(), streaming=True)

    # Real schema (Nia 2026-05): top-level fields = {date, text, extra_fields}.
    # `extra_fields` is a JSON STRING (not dict) — must json.loads() per row.
    # Tickers live at extras["stocks"] (plural array). Sources/url/publisher also nested.
    rows: list[dict] = []
    seen = 0
    for ex in ds_iter:
        seen += 1
        try:
            extras_raw = ex.get("extra_fields")
            extras = json.loads(extras_raw) if isinstance(extras_raw, str) else (extras_raw or {})
        except (json.JSONDecodeError, TypeError):
            extras = {}
        stocks = extras.get("stocks") or []
        if not isinstance(stocks, list):
            stocks = []
        relevant = any(str(s).upper().strip() in GOLD_RELEVANT_TICKERS for s in stocks)
        if relevant:
            rows.append(
                {
                    "date": ex.get("date"),
                    "text": ex.get("text"),
                    "stocks": stocks,
                    "source": str(extras.get("source") or "").lower(),
                    "publisher": str(extras.get("publisher") or "").lower(),
                    "url": str(extras.get("url") or ""),
                    "author": str(extras.get("author") or ""),
                    "category": str(extras.get("category") or ""),
                }
            )
        if seen % 1_000_000 == 0:
            LOG.info("multisource scan: %d examined, %d kept", seen, len(rows))

    LOG.info("multisource scan complete: %d examined, %d gold-relevant", seen, len(rows))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(df)

    created_at = pd.to_datetime(df["date"], utc=True, errors="coerce")
    src_raw = df["source"].fillna("").astype(str).str.lower()
    pub_raw = df["publisher"].fillna("").astype(str).str.lower()
    src_combined = (src_raw + "|" + pub_raw).str.strip("|")
    bias = src_combined.map(
        lambda s: next((v for k, v in BIAS_BY_SOURCE.items() if k in s), DEFAULT_BIAS)
    )

    # article_id: prefer URL when present, else "{source}_{idx}"
    url_series = df["url"].fillna("").astype(str)
    fallback_ids = pd.Series([f"multisource_{i}" for i in range(n)], dtype="string", index=df.index)
    article_id = url_series.where(url_series.str.len() > 0, other=fallback_ids)
    symbols_str = df["stocks"].apply(lambda lst: "|".join(str(x) for x in (lst or [])))

    out = pd.DataFrame(
        {
            "article_id": article_id.astype("string"),
            "source": (
                "multisource_" + src_raw.where(src_raw.str.len() > 0, other="unknown")
            ).astype("string"),
            "created_at": created_at,
            "title": df["text"].astype("string"),
            "body": pd.array([pd.NA] * n, dtype="string"),
            "url": url_series.astype("string"),
            "symbols": symbols_str.astype("string"),
            "bias_tier": pd.array(bias.tolist(), dtype="string"),
        }
    )
    out = out.dropna(subset=["created_at"])
    # Multi-ticker tagging in source dataset duplicates the same URL across stock
    # tags. Keep the first occurrence so (source, article_id) is unique downstream.
    out = out.drop_duplicates(subset=["article_id"], keep="first").reset_index(drop=True)
    # Date-precise rows get a 1-day buffer; minute-precise rows keep 60s.
    out["release_ts"] = out["created_at"]
    out["t_visible"] = out["created_at"] + pd.Timedelta(seconds=NEWS_LATENCY_SEC_ALPACA)
    return out[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_multisource_parquet() -> tuple[pd.DataFrame, str]:
    df = fetch_filtered()
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "multisource_news.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d multisource rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_multisource_parquet()
    print(f"multisource: {len(df)} rows -> {p}")
