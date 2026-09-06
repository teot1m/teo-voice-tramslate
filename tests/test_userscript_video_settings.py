"""Exercise video-local settings in a real browser without models or a server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_video_settings_in_real_dom():
    if not shutil.which("node"):
        pytest.skip("Node.js needed for userscript browser fixtures")
    module = os.environ.get("UVT_PLAYWRIGHT_MODULE")
    if not module:
        candidates = list((Path.home() / ".npm" / "_npx").glob("*/node_modules/playwright/package.json"))
        module = str(candidates[0].parent) if candidates else None
    if not module:
        pytest.skip("Set UVT_PLAYWRIGHT_MODULE to an installed Playwright package")
    result = subprocess.run(
        ["node", str(ROOT / "tests/userscript_video_settings.cjs"), str(ROOT / "browser/uvt.user.js")],
        env={**os.environ, "UVT_PLAYWRIGHT_MODULE": module},
        text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["passed"] >= 5
    assert report["browserErrors"] == []
