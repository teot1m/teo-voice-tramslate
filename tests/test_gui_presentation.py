"""Небольшие pure-ish проверки copy для GUI (GUI extra может отсутствовать)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from uvt.config import load_config  # noqa: E402
from uvt.gui.main_window import _is_virtual_device, _privacy_summary  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_privacy_copy_separates_local_free_and_cloud():
    local = load_config(str(ROOT / "profiles" / "local.yaml"))
    free = load_config(str(ROOT / "profiles" / "free.yaml"))
    cloud = load_config(str(ROOT / "profiles" / "cloud-fast.yaml"))

    assert _privacy_summary(local)[0] == "PRIVATE LOCAL"
    assert _privacy_summary(free)[0] == "HYBRID"
    assert "Microsoft Edge TTS" in _privacy_summary(free)[1]
    assert _privacy_summary(cloud)[0] == "CLOUD"


def test_virtual_source_marker_covers_common_names():
    assert _is_virtual_device("BlackHole 2ch")
    assert _is_virtual_device("Monitor of Built-in Audio")
    assert not _is_virtual_device("MacBook Pro Microphone")
