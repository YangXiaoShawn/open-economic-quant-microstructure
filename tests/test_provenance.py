from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import microstructure
import microstructure.provenance as provenance_module
from microstructure.provenance import (
    ImportOriginError,
    assert_project_module_origins,
    git_source_tree_sha256,
    git_state,
    read_json,
    sha256_file,
    strict_git_state,
    write_json,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_sha256_and_atomic_json_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_json(target, {"b": 2, "a": [1, 3]})

    assert read_json(target) == {"a": [1, 3], "b": 2}
    assert sha256_file(target) == sha256_file(target)
    assert not list(tmp_path.glob("*.tmp"))


def test_git_state_handles_unborn_clean_and_dirty_repositories(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")

    unborn = git_state(repository)
    unborn_source = git_source_tree_sha256(repository)
    assert unborn.commit == "UNBORN"
    assert unborn.dirty is True

    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Provenance Test",
            "-c",
            "user.email=provenance@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    clean = git_state(repository)
    clean_source = git_source_tree_sha256(repository)
    assert len(clean.commit) == 40
    assert clean.dirty is False
    assert clean_source == unborn_source

    tracked.write_text("modified\n", encoding="utf-8")
    dirty = git_state(repository)
    dirty_source = git_source_tree_sha256(repository)
    assert dirty.commit == clean.commit
    assert dirty.dirty is True
    assert dirty_source != clean_source

    (repository / "untracked.txt").write_text("new source\n", encoding="utf-8")
    assert git_source_tree_sha256(repository) != dirty_source


def test_git_state_fails_closed_when_status_cannot_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def failed_status(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="a" * 40 + "\n")
        return subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal\n")

    monkeypatch.setattr(subprocess, "run", failed_status)

    with pytest.raises(RuntimeError, match="strict Git working-tree identity"):
        strict_git_state(tmp_path)


def test_loaded_module_origin_rejects_foreign_file_and_mixed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert_project_module_origins(PROJECT_ROOT, provenance_module)

    foreign = tmp_path / "checkout" / "src" / "microstructure" / "provenance.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# foreign checkout\n", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(provenance_module, "__file__", str(foreign))
        with pytest.raises(ImportOriginError, match="foreign source root"):
            assert_project_module_origins(PROJECT_ROOT, provenance_module)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            microstructure,
            "__path__",
            [str(PROJECT_ROOT / "src" / "microstructure"), str(foreign.parent)],
        )
        with pytest.raises(ImportOriginError, match="mixed namespace"):
            assert_project_module_origins(PROJECT_ROOT, provenance_module)
