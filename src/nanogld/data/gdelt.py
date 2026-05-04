"""GDELT 2.0 GKG via BigQuery — Source 3.

V4-corrected theme codes from plan/02-DATA-PIPELINE.md "Source 3":
- 6 codes REFUTED (EPU_*, TAX_WEAPONS_BOMB, WB_2432_FRAGILITY).
- Replacements: TAX_WEAPONS, WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE.

Hard rules:
- maximum_bytes_billed = 1.1 TB cap on every query (catastrophe mitigation).
- Dry-run BEFORE every real query — print scan estimate, abort if >1 TB.
- Stream results via Arrow + create_bqstorage_client (10-50x faster).
- Filter on `_PARTITIONTIME` for partition pruning (DATE column does NOT prune).
- t_visible = max(DATE + 30min, _PARTITIONTIME) — V4 §7 + §9.

Materialize 5y of gold-relevant GKG ONCE to a local table
(`<project>.gold_news.gkg_5y`, ~5-10 GB, fits free 10 GB BigQuery storage tier).
Subsequent queries hit the local table (~5-10 GB scan instead of 931 GB).
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from nanogld.data.schema import GDELT_MANIFEST, validate
from nanogld.data.utils import (
    END_DATE_UTC,
    NEWS_LATENCY_MIN_GDELT,
    START_DATE_UTC,
    get_logger,
    raw_dir,
)

LOG = get_logger("nanogld.data.gdelt")

DEFAULT_PROJECT = os.environ.get("NANOGLD_GCP_PROJECT", "nanogld-data")
DEFAULT_DATASET = "gold_news"
DEFAULT_TABLE = "gkg_5y"
MAX_BYTES_BILLED = 1_100_000_000_000  # 1.1 TB hard cap

# V4-corrected GKG theme regex (plan/02-DATA-PIPELINE.md "Source 3" + V4 §6)
GOLD_THEMES = "WB_2936_GOLD|ECON_GOLDPRICE|WB_2937_SILVER|SLFID_MINERAL_RESOURCES"
MONETARY_THEMES = (
    "ECON_INTEREST_RATES|ECON_INFLATION|ECON_CENTRALBANK"
    "|WB_1235_CENTRAL_BANKS|WB_444_MONETARY_POLICY"
)
CONFLICT_THEMES = (
    "ARMEDCONFLICT|WB_2433_CONFLICT_AND_VIOLENCE"
    "|WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE"
    "|TERROR|SANCTIONS|TAX_WEAPONS|MARITIME_INCIDENT"
)
STRESS_THEMES = "ECON_BANKRUPTCY|ECON_TRADE_DISPUTE|ECON_DEBT"


def _materialize_sql(start_utc: datetime, end_utc: datetime, target_fqn: str) -> str:
    """SQL that filters gdelt-bq.gdeltv2.gkg_partitioned to gold-relevant rows.

    target_fqn = "<project>.<dataset>.<table>" (back-tick wrapped at call site).
    """
    return f"""
CREATE OR REPLACE TABLE `{target_fqn}` AS
SELECT
  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING)) AS pub_ts_utc,
  g._PARTITIONTIME                                          AS partition_ts_utc,
  g.DocumentIdentifier                                       AS url,
  g.V2Themes                                                 AS v2_themes,
  g.V2Tone                                                   AS v2_tone,
  g.V2Locations                                              AS v2_locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned` AS g
WHERE g._PARTITIONTIME BETWEEN
        TIMESTAMP('{start_utc.strftime("%Y-%m-%d")}')
    AND TIMESTAMP('{end_utc.strftime("%Y-%m-%d")}')
  AND g.TranslationInfo = ''
  AND (
    REGEXP_CONTAINS(g.V2Themes, r'{GOLD_THEMES}')
    OR REGEXP_CONTAINS(g.V2Themes, r'{MONETARY_THEMES}')
    OR REGEXP_CONTAINS(g.V2Themes, r'{CONFLICT_THEMES}')
    OR REGEXP_CONTAINS(g.V2Themes, r'{STRESS_THEMES}')
  )
""".strip()


def _client():
    """Lazy import — bigquery is heavy and only needed if owner has authed gcloud."""
    from google.cloud import bigquery  # noqa: PLC0415

    return bigquery.Client(project=DEFAULT_PROJECT)


def _job_config(*, dry_run: bool = False):
    from google.cloud import bigquery  # noqa: PLC0415

    return bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_BILLED,
        use_query_cache=True,
        dry_run=dry_run,
    )


def dry_run(sql: str) -> int:
    """Return scan-byte estimate for `sql`. Aborts if estimate > 1 TB."""
    client = _client()
    job = client.query(sql, job_config=_job_config(dry_run=True))
    bytes_scan = int(job.total_bytes_processed or 0)
    LOG.info("dry-run: %.2f GB will be scanned", bytes_scan / 1e9)
    if bytes_scan > MAX_BYTES_BILLED:
        raise RuntimeError(
            f"dry-run estimate {bytes_scan / 1e9:.1f} GB > MAX_BYTES_BILLED "
            f"{MAX_BYTES_BILLED / 1e9:.0f} GB — aborting"
        )
    return bytes_scan


def materialize(
    *,
    project: str = DEFAULT_PROJECT,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    start_utc: datetime = START_DATE_UTC,
    end_utc: datetime = END_DATE_UTC,
) -> str:
    """One-time 5y materialize. ~931 GB scan — under free 1 TB tier ONCE.

    Subsequent queries hit the local table at ~5-10 GB each.
    Returns the fully-qualified table name.
    """
    target = f"{project}.{dataset}.{table}"
    sql = _materialize_sql(start_utc, end_utc, target)
    LOG.info("preparing GDELT materialize: %s", target)

    bytes_scan = dry_run(sql)
    LOG.warning(
        "REAL run will scan %.1f GB. This consumes ~%.0f%% of monthly free 1 TB tier.",
        bytes_scan / 1e9,
        100 * bytes_scan / 1e12,
    )

    client = _client()
    job = client.query(sql, job_config=_job_config(dry_run=False))
    job.result()  # block until done
    LOG.info("materialized %s (job_id=%s)", target, job.job_id)
    return target


def query_local(
    *,
    project: str = DEFAULT_PROJECT,
    dataset: str = DEFAULT_DATASET,
    table: str = DEFAULT_TABLE,
    start_utc: datetime = START_DATE_UTC,
    end_utc: datetime = END_DATE_UTC,
) -> pd.DataFrame:
    """Read a slice of the materialized local table into a tidy DataFrame.

    Streams via Arrow with create_bqstorage_client for 10-50x throughput.
    """
    sql = f"""
SELECT pub_ts_utc, partition_ts_utc, url, v2_themes, v2_tone, v2_locations
FROM `{project}.{dataset}.{table}`
WHERE pub_ts_utc BETWEEN
        TIMESTAMP('{start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}')
    AND TIMESTAMP('{end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}')
""".strip()

    dry_run(sql)
    client = _client()
    df = (
        client.query(sql, job_config=_job_config(dry_run=False))
        .result()
        .to_dataframe(create_bqstorage_client=True)
    )
    return _attach_t_visible(df)


def _attach_t_visible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["pub_ts_utc"] = pd.to_datetime(df["pub_ts_utc"], utc=True)
    df["partition_ts_utc"] = pd.to_datetime(df["partition_ts_utc"], utc=True)
    for c in ("url", "v2_themes", "v2_tone", "v2_locations"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    # release_ts = max(pub_ts + 30min, partition_ts)  (V4 §7 + §9)
    pub_plus_buffer = df["pub_ts_utc"] + pd.Timedelta(minutes=NEWS_LATENCY_MIN_GDELT)
    df["release_ts"] = pub_plus_buffer.where(
        pub_plus_buffer >= df["partition_ts_utc"], df["partition_ts_utc"]
    )
    df["t_visible"] = df["release_ts"]
    return df[[c.name for c in GDELT_MANIFEST.columns]]


def write_gdelt_parquet() -> tuple[pd.DataFrame, str]:
    df = query_local()
    if df.empty:
        LOG.warning("GDELT query returned 0 rows — has the table been materialized?")
        return df, ""
    validate(df, GDELT_MANIFEST)
    out_path = raw_dir() / "gdelt_gkg_5y.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d GDELT rows -> %s", len(df), out_path)
    return df, str(out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "materialize":
        materialize()
    else:
        df, p = write_gdelt_parquet()
        print(f"GDELT: {len(df)} rows -> {p or '(no rows; run materialize first)'}")
