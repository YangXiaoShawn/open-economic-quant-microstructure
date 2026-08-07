from __future__ import annotations

import subprocess
from pathlib import Path

from microstructure.provenance import (
    git_source_tree_sha256,
    git_state,
    read_json,
    sha256_file,
    write_json,
)


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
