"""CI smoke tests. Keep trivial — exists so CI is green from day 1."""

import sys


def test_python_version_in_range() -> None:
    assert sys.version_info[:2] in {(3, 11), (3, 12)}, (
        f"pyproject.toml requires Python 3.11 or 3.12, got {sys.version_info[:3]}"
    )


def test_critical_imports() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import torch  # noqa: F401


def test_nanogld_package_imports() -> None:
    import nanogld

    assert nanogld.__version__ == "0.1.0"
