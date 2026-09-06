"""Check all-player visibility and idle controls in an installed Chromium browser."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _playwright_module() -> str | None:
    configured = os.environ.get("UVT_PLAYWRIGHT_MODULE")
    if configured:
        return configured
    # Reuse a tool runtime if present; tests never install npm dependencies.
    candidates = list((Path.home() / ".npm" / "_npx").glob("*/node_modules/playwright/package.json"))
    return str(candidates[0].parent) if candidates else None


def test_visible_players_and_idle_controls_in_real_dom():
    if not shutil.which("node"):
        pytest.skip("Node.js is needed for the userscript browser fixtures")
    module = _playwright_module()
    if not module:
        pytest.skip("Set UVT_PLAYWRIGHT_MODULE to an installed Playwright package")
    env = {**os.environ, "UVT_PLAYWRIGHT_MODULE": module}
    result = subprocess.run(["node", str(ROOT / "tests/userscript_main_player.cjs"),
        str(ROOT / "browser/uvt.user.js")], env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["passed"] >= 20
    assert report["browserErrors"] == []
