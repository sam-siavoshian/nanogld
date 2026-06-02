"""Tests for ``nanogld._atomic``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from nanogld._atomic import (
    atomic_dir_commit,
    atomic_dir_writer,
    atomic_save_torch,
    atomic_write_bytes,
    atomic_write_json,
)


# ---- single-file helpers ----


def test_atomic_save_torch_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "state.pt"
    payload = {"w": torch.ones(4), "b": torch.zeros(4)}
    atomic_save_torch(payload, p)
    loaded = torch.load(p, weights_only=False)
    assert torch.equal(loaded["w"], torch.ones(4))
    assert torch.equal(loaded["b"], torch.zeros(4))
    assert not (tmp_path / "state.pt.tmp").exists()


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "meta.json"
    payload = {"fold": 0, "git_sha": "abc123", "stage": "ssl"}
    atomic_write_json(p, payload)
    loaded = json.loads(p.read_text())
    assert loaded == payload
    assert not (tmp_path / "meta.json.tmp").exists()


def test_atomic_write_bytes_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    data = b"\x00\x01\x02\x03"
    atomic_write_bytes(p, data)
    assert p.read_bytes() == data


def test_atomic_save_torch_missing_parent_raises(tmp_path: Path) -> None:
    p = tmp_path / "missing_dir" / "state.pt"
    with pytest.raises(FileNotFoundError):
        atomic_save_torch({"a": 1}, p)


def test_atomic_write_json_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}


def test_atomic_save_torch_keeps_old_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "state.pt"
    atomic_save_torch({"v": torch.tensor([1.0])}, p)
    sentinel_size = p.stat().st_size

    def _explode(*_a: object, **_kw: object) -> None:
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(torch, "save", _explode)
    with pytest.raises(RuntimeError, match="mid-write"):
        atomic_save_torch({"v": torch.tensor([99.0])}, p)
    assert p.exists()
    assert p.stat().st_size == sentinel_size
    assert not (tmp_path / "state.pt.tmp").exists()


# ---- dir-level helpers ----


def test_atomic_dir_writer_success(tmp_path: Path) -> None:
    out = tmp_path / "fold_0"
    with atomic_dir_writer(out) as tmp:
        atomic_write_json(tmp / "meta.json", {"fold": 0})
        atomic_save_torch({"w": torch.tensor([1.0])}, tmp / "state.pt")
        (tmp / "extra.txt").write_text("hello")
    assert out.exists()
    assert (out / "meta.json").exists()
    assert (out / "state.pt").exists()
    assert (out / "extra.txt").read_text() == "hello"
    assert not (tmp_path / "fold_0.tmp").exists()
    assert not (tmp_path / "fold_0.bak").exists()


def test_atomic_dir_writer_rollback_preserves_old(tmp_path: Path) -> None:
    out = tmp_path / "fold_1"
    out.mkdir()
    (out / "sentinel").write_text("preserved")
    (out / "old_state.pt").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="mid-write"):
        with atomic_dir_writer(out) as tmp:
            atomic_write_json(tmp / "new.json", {"v": 1})
            raise RuntimeError("mid-write crash")

    assert (out / "sentinel").read_text() == "preserved"
    assert (out / "old_state.pt").read_bytes() == b"old"
    assert not (out / "new.json").exists()
    assert not (tmp_path / "fold_1.tmp").exists()
    assert not (tmp_path / "fold_1.bak").exists()


def test_atomic_dir_writer_new_dir(tmp_path: Path) -> None:
    out = tmp_path / "brand_new"
    assert not out.exists()
    with atomic_dir_writer(out) as tmp:
        atomic_write_json(tmp / "x.json", {"k": "v"})
    assert out.exists()
    assert (out / "x.json").exists()


def test_atomic_dir_commit_explicit(tmp_path: Path) -> None:
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    (tmp / "a.txt").write_text("aaa")
    final = tmp_path / "committed"
    atomic_dir_commit(tmp, final)
    assert final.exists()
    assert (final / "a.txt").read_text() == "aaa"
    assert not tmp.exists()


def test_atomic_dir_commit_replaces_existing(tmp_path: Path) -> None:
    final = tmp_path / "target"
    final.mkdir()
    (final / "old.txt").write_text("OLD")
    tmp = tmp_path / "fresh"
    tmp.mkdir()
    (tmp / "new.txt").write_text("NEW")
    atomic_dir_commit(tmp, final)
    assert (final / "new.txt").read_text() == "NEW"
    assert not (final / "old.txt").exists()
    assert not (tmp_path / "target.bak").exists()


def test_atomic_dir_commit_missing_tmp_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        atomic_dir_commit(tmp_path / "nope", tmp_path / "dst")


def test_atomic_dir_commit_tmp_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        atomic_dir_commit(f, tmp_path / "dst")
