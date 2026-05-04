"""Kaggle gold-labeled sentiment — Source 18.

Direct gold-labeled headlines for training a binary "is this article gold-relevant?"
classifier (used by doc 03 to filter the bigger corpora). NOT a feature input.

Spec: plan/02-DATA-PIPELINE.md "Source 18".
- ankurzing/sentiment-analysis-in-commodity-market-gold (CC0/CC BY)
- Mirror: huggingface.co/datasets/SaguaroCapital/sentiment-analysis-in-commodity-market-gold

V1 prefers the HF mirror (no Kaggle CLI auth needed). Kaggle CLI is the fallback
if HF mirror is taken down.
"""

from __future__ import annotations

import os

import pandas as pd

from nanogld.data.schema import NEWS_MANIFEST, validate
from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.news_kaggle")

HF_MIRROR = "SaguaroCapital/sentiment-analysis-in-commodity-market-gold"
BIAS_TIER = "labeled_corpus"  # not a real news source — labeled training corpus


def _ensure_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok if tok and "FILL_ME" not in str(tok) else None


def fetch_kaggle_gold_labeled() -> pd.DataFrame:
    """Load the HF mirror of the Kaggle gold-sentiment dataset."""
    from datasets import load_dataset  # noqa: PLC0415

    LOG.info("loading Kaggle gold-labeled HF mirror")
    try:
        ds = load_dataset(HF_MIRROR, split="train", token=_ensure_token())
    except Exception as e:  # noqa: BLE001
        LOG.warning("Kaggle gold-labeled HF load failed: %s", e)
        return pd.DataFrame()

    rows = list(ds)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(df)

    def col_or_na(*names: str) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([pd.NA] * n, index=df.index)

    date_col = next(
        (
            c
            for c in df.columns
            if c.lower().startswith(("date", "datetime", "timestamp", "pub_date"))
        ),
        None,
    )
    created_at = (
        pd.to_datetime(df[date_col], utc=True, errors="coerce")
        if date_col
        else pd.Series([pd.NaT] * n, index=df.index)
    )

    article_id = col_or_na("id", "article_id", "uid")
    if article_id.isna().all():
        article_id = pd.Series(df.index.astype(str), index=df.index)

    out = pd.DataFrame(
        {
            "article_id": article_id.astype("string"),
            "source": pd.array(["kaggle_gold_labeled"] * n, dtype="string"),
            "created_at": created_at,
            "title": col_or_na("title", "headline", "News").astype("string"),
            "body": col_or_na("body", "text", "content", "News").astype("string"),
            "url": col_or_na("url", "URL").astype("string"),
            "symbols": pd.array([pd.NA] * n, dtype="string"),
            "bias_tier": pd.array([BIAS_TIER] * n, dtype="string"),
        }
    )
    out = out.dropna(subset=["created_at"])
    out["release_ts"] = out["created_at"]
    out["t_visible"] = out["created_at"]
    return out[[c.name for c in NEWS_MANIFEST.columns]].reset_index(drop=True)


def write_kaggle_parquet() -> tuple[pd.DataFrame, str]:
    df = fetch_kaggle_gold_labeled()
    if df.empty:
        return df, ""
    validate(df, NEWS_MANIFEST)
    out_path = raw_dir() / "kaggle_gold_labeled.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d Kaggle-labeled rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    df, p = write_kaggle_parquet()
    print(f"Kaggle gold-labeled: {len(df)} rows -> {p}")
