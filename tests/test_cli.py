from __future__ import annotations

from pathlib import Path

from microstructure.cli import main

PROJECT_ROOT = Path(__file__).parents[1]


def test_cli_help_and_version_are_available(capsys: object) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0


def test_validate_command_runs_offline(capsys: object) -> None:
    exit_code = main(["validate", "--config", str(PROJECT_ROOT / "configs" / "smoke.toml")])

    assert exit_code == 0
