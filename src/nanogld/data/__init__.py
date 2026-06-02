"""nanogld data subpackage."""

from nanogld.data.dataset import NanoGLDDataset
from nanogld.data.integrity import MANIFEST_NAME, verify_artifacts, write_manifest
from nanogld.data.utils import ET, get_logger, raw_dir, repo_root
from nanogld.data.walk_forward_splits import FoldBoundary, compute_fold_boundaries

__all__ = [
    "ET",
    "FoldBoundary",
    "MANIFEST_NAME",
    "NanoGLDDataset",
    "compute_fold_boundaries",
    "get_logger",
    "raw_dir",
    "repo_root",
    "verify_artifacts",
    "write_manifest",
]
