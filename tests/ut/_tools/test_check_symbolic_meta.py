# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "tools" / "check_symbolic_meta.py"
SPEC = importlib.util.spec_from_file_location("check_symbolic_meta", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_symbolic_meta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_symbolic_meta)


def test_reports_high_risk_concrete_shape(tmp_path: Path):
    target = tmp_path / "torch_binding_meta.cpp"
    target.write_text("auto rows = x.size(0);\n", encoding="utf-8")

    violations = check_symbolic_meta.check_file(target)

    assert len(violations) == 1
    assert "uses `.size(`" in violations[0]


def test_honors_exemption_with_reason(tmp_path: Path):
    target = tmp_path / "torch_binding_meta.cpp"
    target.write_text(
        "// symbolic-meta-ok: schema argument, not a tensor shape\nauto length = active_expert_range.size();\n",
        encoding="utf-8",
    )

    assert check_symbolic_meta.check_file(target) == []


def test_missing_target_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    target = tmp_path / "missing.cpp"
    monkeypatch.setattr(check_symbolic_meta.sys, "argv", ["check_symbolic_meta.py", str(target)])

    assert check_symbolic_meta.main() == 1
    output = capsys.readouterr().out
    assert "cannot read target file" in output
    assert str(target) in output


def test_unreadable_target_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "unreadable.cpp"
    target.write_text("safe content\n", encoding="utf-8")
    original_read_text = Path.read_text

    def raise_permission_error(self: Path, *args, **kwargs):
        if self == target:
            raise PermissionError("permission denied for test")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    violations = check_symbolic_meta.check_file(target)

    assert len(violations) == 1
    assert "cannot read target file" in violations[0]
    assert "permission denied for test" in violations[0]


def test_invalid_utf8_fails_closed(tmp_path: Path):
    target = tmp_path / "invalid.cpp"
    target.write_bytes(b"\xff")

    violations = check_symbolic_meta.check_file(target)

    assert len(violations) == 1
    assert "cannot read target file" in violations[0]
    assert "utf-8" in violations[0]
