"""Профили и CLI не должны снова скрывать маршрут данных или fake mode."""
from __future__ import annotations

from pathlib import Path

import pytest

from uvt.cli import _build_parser, _overrides, main
from uvt.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected_tts"),
    [
        ("local", "piper"),
        ("free", "edge"),
        ("free-quality", "edge"),
        ("cloud-fast", "openai"),
        ("cloud-quality", "openai"),
        ("cloud", "openai"),
    ],
)
def test_curated_profile_routes_load(name, expected_tts):
    cfg = load_config(str(ROOT / "profiles" / f"{name}.yaml"))
    assert cfg.mode == "voiceover"
    assert cfg.tts.engine == expected_tts
    if name.startswith("cloud"):
        assert cfg.stt.fallback is not None
        assert cfg.stt.fallback.engine == "faster-whisper"
        assert cfg.translation.fallback is not None
        assert cfg.translation.fallback.base_url.startswith("http://127.0.0.1")
        assert cfg.translation.fallback.api_key_env != "OPENAI_API_KEY"
        assert cfg.translation.fallback.batch_size == 4
        assert cfg.translation.fallback.concurrency == 1


def test_local_profile_is_configured_for_real_local_tts():
    cfg = load_config(str(ROOT / "profiles" / "local.yaml"))
    assert cfg.stt.engine == "faster-whisper"
    assert cfg.translation.base_url.startswith("http://localhost")
    assert cfg.tts.engine == "piper"
    assert cfg.tts.model_path


def test_free_profile_does_not_silently_claim_local_tts():
    cfg = load_config(str(ROOT / "profiles" / "free.yaml"))
    assert cfg.tts.engine == "edge"


def test_free_quality_is_an_opt_in_serial_local_llm_profile():
    cfg = load_config(str(ROOT / "profiles" / "free-quality.yaml"))
    assert cfg.stt.engine == "mlx-whisper"
    assert cfg.stt.model == "large-v3-turbo"
    assert cfg.translation.model == "qwen2.5:3b"
    assert cfg.translation.batch_size == 4
    assert cfg.translation.concurrency == 1
    assert cfg.tts.engine == "edge"


def test_legacy_mode_is_normalized_to_honest_voiceover():
    args = _build_parser().parse_args(["run", "--mode", "replace"])
    assert _overrides(args)["mode"] == "voiceover"


def test_profiles_command_explains_edge_network_route(capsys):
    assert main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "cloud-fast" in out
    assert "Microsoft" in out
    assert "бесплатен по цене" in out
