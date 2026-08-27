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
        ("local-fast", "piper"),
        ("local-balanced", "piper"),
        ("local-quality", "piper"),
        ("free", "edge"),
        ("free-vps", "edge"),
        ("free-quality", "piper"),
        ("cloud-fast", "openai"),
        ("cloud-eleven", "elevenlabs"),
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


def test_free_vps_profile_is_explicitly_linux_cpu_safe():
    cfg = load_config(str(ROOT / "profiles" / "free-vps.yaml"))
    assert cfg.stt.engine == "faster-whisper"
    assert cfg.stt.device == "cpu"
    assert cfg.stt.compute_type == "int8"
    assert cfg.translation.model == "qwen2.5:3b"


def test_free_quality_is_memory_bounded_private_mac_profile():
    cfg = load_config(str(ROOT / "profiles" / "free-quality.yaml"))
    assert cfg.stt.engine == "mlx-whisper"
    assert cfg.stt.model == "large-v3-turbo"
    assert cfg.translation.engine == "nllb-ct2"
    assert cfg.translation.model == "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
    assert cfg.translation.compute_type == "int8"
    assert cfg.translation.batch_size == 32
    assert cfg.translation.concurrency == 1
    assert cfg.tts.engine == "piper"
    assert cfg.tts.voice_models["uk:male"].endswith("mykyta-high.onnx")


def test_legacy_mode_is_normalized_to_honest_voiceover():
    args = _build_parser().parse_args(["run", "--mode", "replace"])
    assert _overrides(args)["mode"] == "voiceover"


def test_personal_server_command_has_three_distinct_default_ports(monkeypatch):
    for name in (
        "UVT_FREE_PORT",
        "UVT_CLOUD_PORT",
        "UVT_ELEVEN_PORT",
        "UVT_FREE_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("uvt.cli.sys.platform", "linux")
    args = _build_parser().parse_args(["serve-personal"])
    assert (args.free_port, args.gpt_port, args.eleven_port) == (8765, 8766, 8767)
    assert args.free_profile == "free-vps"


def test_personal_server_accepts_apple_silicon_free_profile(monkeypatch):
    monkeypatch.delenv("UVT_FREE_PROFILE", raising=False)
    monkeypatch.setattr("uvt.cli.sys.platform", "darwin")
    monkeypatch.setattr("uvt.cli.platform.machine", lambda: "arm64")
    args = _build_parser().parse_args(["serve-personal"])
    assert args.free_profile == "local-balanced"


def test_setup_mac_local_defaults_to_balanced_and_supports_offline_check():
    args = _build_parser().parse_args(["setup-mac-local", "--check"])
    assert args.preset == "balanced"
    assert args.check is True


@pytest.mark.parametrize(
    ("preset", "expected_launch"),
    [
        ("balanced", "uvt serve-personal"),
        ("all", "uvt serve-personal"),
        ("fast", "uvt serve-personal --free-profile local-fast"),
        ("quality", "uvt serve-personal --free-profile local-quality"),
    ],
)
def test_setup_mac_local_prints_matching_server_profile(
    monkeypatch, capsys, preset, expected_launch
):
    monkeypatch.setattr(
        "uvt.setup_local.setup_mac_local",
        lambda selected, dry_run=False: {
            "ready": True,
            "preset": selected,
            "dry_run": dry_run,
            "models": {},
            "piper": {},
        },
    )
    monkeypatch.setattr(
        "uvt.setup_local.format_setup_report", lambda _result: "ready"
    )

    assert main(["setup-mac-local", "--preset", preset, "--check"]) == 0
    assert f"Запуск: {expected_launch}" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["cloud-fast", "cloud-eleven"])
def test_personal_cloud_profiles_use_parallel_batch_settings(name):
    cfg = load_config(str(ROOT / "profiles" / f"{name}.yaml"))
    assert cfg.stt.concurrency == 8
    assert cfg.translation.batch_size == 24
    assert cfg.translation.concurrency == 6


def test_profiles_command_explains_edge_network_route(capsys):
    assert main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "cloud-fast" in out
    assert "Microsoft" in out
    assert "бесплатен по цене" in out
