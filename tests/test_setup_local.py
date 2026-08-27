"""Pinned local setup never resolves moving model revisions during jobs."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from uvt import setup_local


def test_profile_engine_pairs_map_to_setup_presets():
    def cfg(stt: str, translation: str):
        return SimpleNamespace(
            stt=SimpleNamespace(engine=stt),
            translation=SimpleNamespace(engine=translation),
        )

    assert setup_local.preset_for_profile(cfg("parakeet-mlx", "nllb-ct2")) == "fast"
    assert (
        setup_local.preset_for_profile(
            cfg("parakeet-mlx", "translategemma-mlx")
        )
        == "balanced"
    )
    assert (
        setup_local.preset_for_profile(
            cfg("mlx-whisper", "translategemma-mlx")
        )
        == "quality"
    )
    assert setup_local.preset_for_profile(cfg("dummy", "dummy")) is None


def _fake_hub(tmp_path):
    snapshots: list[dict] = []
    piper_files: list[dict] = []

    def snapshot_download(**kwargs):
        snapshots.append(kwargs)
        spec = next(
            item
            for item in setup_local.LOCAL_MODEL_MANIFEST.values()
            if item.repo_id == kwargs["repo_id"]
        )
        path = tmp_path / "hub" / "snapshots" / kwargs["revision"]
        path.mkdir(parents=True, exist_ok=True)
        for filename in spec.required_files:
            target = path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"test-model")
        return str(path)

    def hf_hub_download(**kwargs):
        piper_files.append(kwargs)
        path = (
            tmp_path
            / "piper-hub"
            / "snapshots"
            / kwargs["revision"]
            / kwargs["filename"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith(".json"):
            path.write_text(
                json.dumps({"audio": {"sample_rate": 22050}}),
                encoding="utf-8",
            )
        else:
            path.write_bytes(b"test-onnx")
        return str(path)

    return snapshots, piper_files, snapshot_download, hf_hub_download


def test_local_manifest_has_four_pinned_presets():
    assert tuple(setup_local.LOCAL_SETUP_PRESETS) == (
        "fast",
        "balanced",
        "quality",
        "all",
    )
    assert setup_local.LOCAL_SETUP_PRESETS["fast"].model_keys == (
        "parakeet",
        "nllb",
    )
    assert setup_local.LOCAL_SETUP_PRESETS["balanced"].model_keys == (
        "parakeet",
        "translategemma",
    )
    assert setup_local.LOCAL_SETUP_PRESETS["quality"].model_keys == (
        "whisper",
        "translategemma",
    )
    assert set(setup_local.LOCAL_SETUP_PRESETS["all"].model_keys) == set(
        setup_local.LOCAL_MODEL_MANIFEST
    )
    for spec in setup_local.LOCAL_MODEL_MANIFEST.values():
        assert len(spec.revision) == 40
        assert all(
            character in "0123456789abcdef" for character in spec.revision
        )
        assert spec.required_files


def test_setup_balanced_downloads_only_exact_pinned_artifacts(
    monkeypatch, tmp_path
):
    snapshots, piper_files, snapshot_download, hf_hub_download = _fake_hub(
        tmp_path
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=snapshot_download,
            hf_hub_download=hf_hub_download,
        ),
    )
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace())
    voice_dir = tmp_path / "voices"

    result = setup_local.setup_mac_local("balanced", voice_dir=voice_dir)

    expected = [
        setup_local.LOCAL_MODEL_MANIFEST["parakeet"],
        setup_local.LOCAL_MODEL_MANIFEST["translategemma"],
    ]
    assert snapshots == [
        {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "local_files_only": False,
        }
        for spec in expected
    ]
    assert len(piper_files) == len(setup_local.PIPER_VOICES) * 2
    assert all(
        call["repo_id"] == setup_local.PIPER_REPO for call in piper_files
    )
    assert all(
        call["revision"] == setup_local.PIPER_REVISION for call in piper_files
    )
    assert result["preset"] == "balanced"
    assert result["ready"] is True
    assert result["stt_path"].is_dir()
    assert result["translation_path"].is_dir()
    provenance = json.loads(
        (voice_dir / ".uvt-piper-manifest.json").read_text(encoding="utf-8")
    )
    assert provenance["revision"] == setup_local.PIPER_REVISION
    assert set(provenance["voices"]) == set(setup_local.PIPER_VOICES)


def test_dry_preflight_uses_local_cache_only_and_never_downloads(
    monkeypatch, tmp_path
):
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    result = setup_local.setup_mac_local(
        "fast", dry_run=True, voice_dir=tmp_path / "missing-voices"
    )

    assert result["dry_run"] is True
    assert result["ready"] is False
    assert len(calls) == 2
    assert all(call["local_files_only"] is True for call in calls)
    assert "Ничего не скачано" in setup_local.format_setup_report(result)
