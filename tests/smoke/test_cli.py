"""P0 CLI smoke tests."""

from typer.testing import CliRunner

from dithyramba import __version__
from dithyramba.cli import app

runner = CliRunner()


def test_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Local, source-grounded interpretive memory" in result.stdout


def test_version_is_machine_readable() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_about_is_calibrated() -> None:
    result = runner.invoke(app, ["about"])

    assert result.exit_code == 0
    assert f"Dithyramba {__version__}" in result.stdout
    assert "v1 release candidate" in result.stdout
    assert "does not establish research truth" in result.stdout
