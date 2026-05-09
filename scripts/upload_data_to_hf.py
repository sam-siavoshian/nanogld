"""Upload V1 training data to a private HuggingFace dataset.

One-time push of unified.pt + sidecar.pt + HMM joblib + splits.yaml
to a private HF Hub dataset. RunPod pulls from HF (200 MB/s reliably)
instead of using flaky Tailscale across the open internet.

Pre-req:
    huggingface-cli login   # supplies HF_TOKEN

Usage:
    python scripts/upload_data_to_hf.py --repo-id <user>/nanogld-v1-data

Spec: plan/V1-SPEC.md §13.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nanogld.data.utils import data_root, get_logger

LOG = get_logger("nanogld.scripts.upload_data_to_hf")


def upload_artifacts(
    repo_id: str,
    files: list[Path],
    private: bool = True,
) -> None:
    """Push files to a private HF dataset repo."""
    from huggingface_hub import HfApi, create_repo  # noqa: PLC0415

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    LOG.info("repo ready: %s (private=%s)", repo_id, private)

    for f in files:
        if not f.exists():
            LOG.warning("skipping missing file: %s", f)
            continue
        LOG.info("uploading %s (%d bytes)", f, f.stat().st_size)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    LOG.info("upload complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="<user>/nanogld-v1-data")
    parser.add_argument(
        "--unified",
        type=Path,
        default=data_root() / "processed" / "training_v1_unified.pt",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=data_root() / "processed" / "training_v1_sidecar.pt",
    )
    parser.add_argument(
        "--hmm",
        type=Path,
        default=data_root() / "processed" / "v1_hmm.joblib",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=data_root() / "splits" / "v1_walk_forward.yaml",
    )
    parser.add_argument("--public", action="store_true", help="upload to public dataset")
    args = parser.parse_args()

    files = [args.unified, args.sidecar, args.hmm, args.splits]
    upload_artifacts(repo_id=args.repo_id, files=files, private=not args.public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
