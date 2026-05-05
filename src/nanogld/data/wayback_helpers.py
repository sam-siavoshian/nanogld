"""Wayback Machine CDX + capture helpers.

Used by news_kitco / news_investing / news_bullionvault to backfill 5y of
articles when the live source's RSS / scraper path is broken or rate-limited.

CDX API: http://web.archive.org/cdx/search/cdx
Fetch:   https://web.archive.org/web/<timestamp>/<url>

Hard rules:
- Polite: 2s sleep between fetches by default; CDX limits to ~60/min hard.
- Exponential backoff on 429/503 up to 60s, then halt with clear "resume tomorrow" log.
- Every captured byte stream cached under data/raw/wayback_cache/<source>/<ts>_<sha>.html
  so re-runs skip the network. Idempotent: cache hits are silent.
- Returns ARE NOT auto-parsed — caller is responsible for HTML extraction
  per source (Kitco / Investing / BullionVault each have different layouts).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from datetime import date
from pathlib import Path

import requests

from nanogld.data.utils import get_logger, raw_dir

LOG = get_logger("nanogld.data.wayback")

CDX_URL = "http://web.archive.org/cdx/search/cdx"
WAYBACK_FETCH_TEMPLATE = "https://web.archive.org/web/{ts}/{url}"

DEFAULT_POLITE_SEC = 2.0
DEFAULT_MAX_RETRY_BACKOFF = 60.0
DEFAULT_HARD_HALT_AFTER = 5  # consecutive 429/503 → halt


def _cache_dir(source: str) -> Path:
    p = raw_dir() / "wayback_cache" / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def cdx_search(
    url_pattern: str,
    *,
    start: date,
    end: date,
    limit: int = 10000,
    collapse: str = "urlkey",
) -> list[tuple[str, str]]:
    """Query Wayback CDX for captures matching url_pattern in [start, end].

    Args:
        url_pattern: globbed URL pattern, e.g. 'kitco.com/news/article/*'.
        start, end: date range, inclusive.
        limit: max captures returned (CDX server-side cap is high; we set 10K default).
        collapse: dedupe key. 'urlkey' collapses identical URLs across timestamps.

    Returns:
        List of (capture_ts_yyyymmddhhmmss, original_url) tuples.
    """
    params = {
        "url": url_pattern,
        "from": start.strftime("%Y%m%d"),
        "to": end.strftime("%Y%m%d"),
        "output": "json",
        "fl": "timestamp,original",
        "collapse": collapse,
        "limit": str(limit),
    }
    qs = urllib.parse.urlencode(params)
    full_url = f"{CDX_URL}?{qs}"
    LOG.info("CDX query: %s", full_url)

    resp = _polite_get(full_url, source="cdx")
    if resp is None:
        return []
    try:
        data = json.loads(resp)
    except json.JSONDecodeError as e:
        LOG.warning("CDX JSON parse failed: %s", e)
        return []
    if not data or len(data) < 2:
        return []
    # First row is header (['timestamp', 'original'])
    captures = [(row[0], row[1]) for row in data[1:] if len(row) >= 2]
    LOG.info("CDX returned %d captures for %s", len(captures), url_pattern)
    return captures


def fetch_capture(
    capture_ts: str,
    original_url: str,
    *,
    source: str,
    polite_sec: float = DEFAULT_POLITE_SEC,
) -> bytes | None:
    """Fetch one Wayback capture, with raw-byte caching keyed by ts+sha.

    Returns the captured HTML bytes, or None on terminal failure.
    Cache key: data/raw/wayback_cache/<source>/<ts>_<url-sha>.html
    """
    url_sha = hashlib.sha256(original_url.encode()).hexdigest()[:16]
    cache_path = _cache_dir(source) / f"{capture_ts}_{url_sha}.html"
    if cache_path.exists():
        return cache_path.read_bytes()

    snapshot_url = WAYBACK_FETCH_TEMPLATE.format(ts=capture_ts, url=original_url)
    body = _polite_get(snapshot_url, source=source, polite_sec=polite_sec)
    if body is None:
        return None
    cache_path.write_bytes(body.encode() if isinstance(body, str) else body)
    return cache_path.read_bytes()


def _polite_get(
    url: str,
    *,
    source: str,
    polite_sec: float = DEFAULT_POLITE_SEC,
    max_backoff: float = DEFAULT_MAX_RETRY_BACKOFF,
    hard_halt_after: int = DEFAULT_HARD_HALT_AFTER,
    timeout: int = 300,
) -> bytes | None:
    """GET with polite delay + exponential backoff on 429/503.

    timeout default 300s — CDX queries with broad globbed URLs over 5y can
    legitimately take 60-180s server-side; the prior 30s default was the
    main reason backfills returned 0 rows.

    Halts after `hard_halt_after` consecutive throttle responses with a
    "resume tomorrow" log. Returns None on terminal failure.
    """
    backoff = polite_sec
    consecutive_throttle = 0
    while True:
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "nanoGLD/0.1 (+https://github.com/sam-siavoshian/nanogld)"},
            )
        except requests.RequestException as e:
            LOG.warning("[%s] request failed: %s", source, e)
            return None

        if resp.status_code == 200:
            time.sleep(polite_sec)
            return resp.content

        if resp.status_code in (429, 503):
            consecutive_throttle += 1
            if consecutive_throttle >= hard_halt_after:
                LOG.warning(
                    "[%s] %d consecutive %d responses — halting (resume tomorrow)",
                    source,
                    consecutive_throttle,
                    resp.status_code,
                )
                return None
            backoff = min(backoff * 2, max_backoff)
            LOG.info(
                "[%s] %d backoff %.1fs (consecutive=%d)",
                source,
                resp.status_code,
                backoff,
                consecutive_throttle,
            )
            time.sleep(backoff)
            continue

        # 404 / 5xx other → log + skip (not a halt)
        LOG.info("[%s] %d on %s — skip", source, resp.status_code, url)
        time.sleep(polite_sec)
        return None


def cache_summary(source: str) -> dict[str, int]:
    """Report cache stats for a source. Useful for resuming soak jobs."""
    d = _cache_dir(source)
    files = list(d.glob("*.html"))
    return {
        "source": source,
        "captures_cached": len(files),
        "bytes_total": sum(f.stat().st_size for f in files),
    }
