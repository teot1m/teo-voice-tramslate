"""Verify completed userscript audio loading/cancellation without network calls."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_completed_userscript_audio_lifecycle():
    if not shutil.which("node"):
        pytest.skip("Node.js is needed for userscript lifecycle checks")
    result = subprocess.run(
        ["node", str(ROOT / "tests/userscript_completed_audio.cjs"), str(ROOT / "browser/uvt.user.js")],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip())["passed"] >= 10
