"""Multi-source bulk historical news — Source 12B.

HF dataset `Brianferrell787/financial-news-multisource` — 57.1M rows
1990-2025, includes Reuters / Bloomberg / Benzinga / NASDAQ subsets.

License: NON-COMMERCIAL (Other / research-only). Gated behind
NANOGLD_NONCOMMERCIAL=1 env flag (default off). Owner explicitly opts in
for personal V1 research training.
"""

from __future__ import annotations

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

    LOG.info("loading %s (filter num_proc=4)", DATASET)
    ds = load_dataset(DATASET, split="train", token=_hf_token())

    def _rel(ex: dict) -> bool:
        sym = str(ex.get("symbol") or "").upper().strip()
        if sym in GOLD_RELEVANT_TICKERS:
            return True
        # fallback: check tickers list / extra_fields
        ts = ex.get("tickers") or []
        return any(str(t).upper().strip() in GOLD_RELEVANT_TICKERS for t in ts)

    filtered = ds.filter(_rel, num_proc=4)
    if len(filtered) == 0:
        return pd.DataFrame()

    df = filtered.to_pandas().reset_index(drop=True)
    n = len(df)

    def col_or_na(*names: str) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([pd.NA] * n, index=df.index)

    pub_raw = col_or_na("date", "published_at", "time_published", "datetime")
    created_at = pd.to_datetime(pub_raw, utc=True, errors="coerce")

    src_raw = col_or_na("source", "publisher").fillna("unknown").astype(str).str.lower()
    bias = src_raw.map(
        lambda s: next((v for k, v in BIAS_BY_SOURCE.items() if k in s), DEFAULT_BIAS)
    )

    out = pd.DataFrame(
        {
            "article_id": col_or_na("id", "article_id", "url").astype("string"),
            "source": ("multisource_" + src_raw).astype("string"),
            "created_at": created_at,
            "title": col_or_na("title", "headline").astype("string"),
            "body": col_or_na("body", "text", "summary", "content").astype("string"),
            "url": col_or_na("url").astype("string"),
            "symbols": col_or_na("symbol", "ticker").astype("string"),
            "bias_tier": pd.array(bias.tolist(), dtype="string"),
        }
    )
    out = out.dropna(subset=["created_at"])
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
