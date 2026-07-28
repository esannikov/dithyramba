"""Tests for the aggregate check runner."""

from __future__ import annotations

from typing import Any

from dithyramba.tooling.checks import Check, run_checks


class _Completed:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_checks_run_in_order(monkeypatch: Any, capsys: Any) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, check: bool) -> _Completed:
        assert check is False
        observed.append(command)
        return _Completed(0)

    monkeypatch.setattr("dithyramba.tooling.checks.subprocess.run", fake_run)
    checks = (Check("first", ("one",)), Check("second", ("two",)))

    assert run_checks(checks) == 0
    assert observed == [("one",), ("two",)]
    assert "all required checks passed" in capsys.readouterr().out


def test_checks_stop_on_first_failure(monkeypatch: Any, capsys: Any) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, check: bool) -> _Completed:
        assert check is False
        observed.append(command)
        return _Completed(7 if command == ("bad",) else 0)

    monkeypatch.setattr("dithyramba.tooling.checks.subprocess.run", fake_run)
    checks = (Check("ok", ("ok",)), Check("bad", ("bad",)), Check("late", ("late",)))

    assert run_checks(checks) == 7
    assert observed == [("ok",), ("bad",)]
    assert "failed with exit code 7" in capsys.readouterr().err
