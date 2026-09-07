"""Exercise the userscript network adapter in real Chrome with simulated GM APIs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_userscript_main_player import _playwright_module

ROOT = Path(__file__).resolve().parents[1]


def test_userscript_extension_transport_in_real_chrome():
    if not shutil.which("node"):
        pytest.skip("Node.js is needed for userscript network fixtures")
    module = _playwright_module()
    if not module:
        pytest.skip("Set UVT_PLAYWRIGHT_MODULE to an installed Playwright package")
    result = subprocess.run(
        ["node", str(ROOT / "tests/userscript_transport.cjs"), str(ROOT / "browser/uvt.user.js")],
        env={**os.environ, "UVT_PLAYWRIGHT_MODULE": module},
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["passed"] >= 12
    assert report["browserErrors"] == []
