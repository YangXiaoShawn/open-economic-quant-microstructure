from __future__ import annotations

from pathlib import Path

from microstructure.provenance import git_state, read_json, sha256_file, write_json


def test_sha256_and_atomic_json_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    write_json(target, {"b": 2, "a": [1, 3]})

    assert read_json(target) == {"a": [1, 3], "b": 2}
    assert sha256_file(target) == sha256_file(target)
    assert not list(tmp_path.glob("*.tmp"))


def test_git_state_handles_unborn_repository() -> None:
    state = git_state(Path(__file__).parents[1])
    assert state.commit == "UNBORN" or len(state.commit) == 40
    assert state.dirty is True
