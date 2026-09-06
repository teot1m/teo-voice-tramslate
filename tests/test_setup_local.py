"""Pinned local setup never resolves moving model revisions during jobs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
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
        path = Path(kwargs["local_dir"]) if kwargs.get("local_dir") else tmp_path / spec.key / "snapshots" / spec.revision
        path.mkdir(parents=True, exist_ok=True)
        for filename in spec.required_files:
            target = path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"test-model")
            if spec.local_dir:
                metadata = path / ".cache/huggingface/download" / (filename + ".metadata")
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(spec.revision + "\netag\n0\n", encoding="utf-8")
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


def test_local_manifest_has_five_pinned_presets():
    assert tuple(setup_local.LOCAL_SETUP_PRESETS) == (
        "fast",
        "balanced",
        "quality",
        "natural",
        "hymt",
        "moss",
        "nemotron",
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
    # natural — маршрут «живого» дубляжа: контекстный перевод chat-моделью и
    # клонирующая озвучка со своими весами вместо готового голоса
    assert setup_local.LOCAL_SETUP_PRESETS["natural"].model_keys == (
        "parakeet",
        "qwen3-chat",
        "f5-ru",
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


NEW_LOCAL_MODELS = ("hymt2", "nemotron", "moss-tts", "moss-codec")


def _checkout(monkeypatch, tmp_path):
    # Redirect only the module's checkout-root calculation; tests must never
    # touch the real weights downloaded by the user.
    monkeypatch.setattr(setup_local,"__file__",str(tmp_path/"src"/"uvt"/"setup_local.py"))


def _local_files(tmp_path,key):
    spec = setup_local.LOCAL_MODEL_MANIFEST[key]
    directory = tmp_path/spec.local_dir
    for filename in spec.required_files:
        target = directory/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b"model-data")
        metadata = directory/".cache/huggingface/download"/(filename+".metadata")
        metadata.parent.mkdir(parents=True,exist_ok=True)
        metadata.write_text(spec.revision+"\netag\n0\n",encoding="utf-8")
    return spec,directory


def test_new_preset_engine_pairs_and_voice_dependency():
    def cfg(stt,translation,tts="piper"):
        return SimpleNamespace(stt=SimpleNamespace(engine=stt),translation=SimpleNamespace(engine=translation),tts=SimpleNamespace(engine=tts))

    assert setup_local.preset_for_profile(cfg("parakeet-mlx","hymt-mlx")) == "hymt"
    assert setup_local.preset_for_profile(cfg("parakeet-mlx","hymt-mlx","moss-onnx")) == "moss"
    assert setup_local.preset_for_profile(cfg("nemotron-mlx","hymt-mlx")) == "nemotron"
    assert setup_local.LOCAL_SETUP_PRESETS["moss"].requires_piper is False
    assert setup_local.LOCAL_SETUP_PRESETS["moss"].model_keys == ("parakeet","hymt2","moss-tts","moss-codec")


@pytest.mark.parametrize("key",NEW_LOCAL_MODELS)
def test_checkout_local_only_preflight_never_calls_hub(monkeypatch,tmp_path,key):
    _checkout(monkeypatch,tmp_path)
    spec,directory = _local_files(tmp_path,key)
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("checkout-local preflight must not call Hugging Face")

    status=setup_local._model_status(spec,forbidden,local_only=True)
    assert status["ready"] is True
    assert Path(status["path"]) == directory
    assert calls == []


@pytest.mark.parametrize("key",NEW_LOCAL_MODELS)
@pytest.mark.parametrize("damage",["wrong-revision","missing-metadata","missing-file","empty-file"])
def test_checkout_preflight_rejects_unverified_or_incomplete_weights(monkeypatch,tmp_path,key,damage):
    _checkout(monkeypatch,tmp_path)
    spec,directory=_local_files(tmp_path,key)
    name=spec.required_files[0]
    metadata=directory/".cache/huggingface/download"/(name+".metadata")
    if damage == "wrong-revision":
        # The revision must be on line one, not merely present in metadata.
        metadata.write_text("0"*40+"\n"+spec.revision+"\n",encoding="utf-8")
    elif damage == "missing-metadata":
        metadata.unlink()
    elif damage == "missing-file":
        (directory/name).unlink()
    else:
        (directory/name).write_bytes(b"")
    calls=[]

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("no network or Hub fallback during preflight")

    status=setup_local._model_status(spec,forbidden,local_only=True)
    assert status["ready"] is False
    assert status["error"]
    assert calls == []


@pytest.mark.parametrize("key",NEW_LOCAL_MODELS)
def test_setup_uses_exact_revision_allowlist_and_checkout_directory(monkeypatch,tmp_path,key):
    _checkout(monkeypatch,tmp_path)
    snapshots,_,download,_=_fake_hub(tmp_path)
    spec=setup_local.LOCAL_MODEL_MANIFEST[key]
    status=setup_local._model_status(spec,download,local_only=False)
    assert status["ready"] is True
    assert snapshots == [{
        "repo_id":spec.repo_id,"revision":spec.revision,"local_files_only":False,
        "allow_patterns":list(spec.allow_patterns),"local_dir":str(tmp_path/spec.local_dir),
    }]


def test_moss_setup_neither_imports_nor_installs_piper(monkeypatch,tmp_path):
    import sys
    _checkout(monkeypatch,tmp_path)
    snapshots,piper_files,download,piper_download=_fake_hub(tmp_path)
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(snapshot_download=download,hf_hub_download=piper_download))
    monkeypatch.setitem(sys.modules,"piper",None)
    monkeypatch.setattr(setup_local,"_install_piper_voices",lambda *_a,**_k: pytest.fail("MOSS must not install unused Piper voices"))
    report=setup_local.setup_mac_local("moss",voice_dir=tmp_path/"unused-voices")
    assert report["ready"] is True
    assert len(snapshots)==4
    assert piper_files == []
    assert not (tmp_path/"unused-voices").exists()


def test_cache_snapshot_with_empty_required_file_is_not_ready(monkeypatch,tmp_path):
    spec=setup_local.LOCAL_MODEL_MANIFEST["parakeet"]
    directory=tmp_path/"snapshots"/spec.revision
    directory.mkdir(parents=True)
    for filename in spec.required_files:
        target=directory/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b"data")
    (directory/spec.required_files[0]).write_bytes(b"")
    status=setup_local._model_status(spec,lambda **kwargs: str(directory),local_only=True)
    assert status["ready"] is False
    assert "пустые" in status["error"]


def test_moss_preflight_validates_local_files_without_piper_or_new_hub_requests(monkeypatch,tmp_path):
    import sys
    _checkout(monkeypatch,tmp_path)
    for key in ("hymt2","moss-tts","moss-codec"):
        _local_files(tmp_path,key)
    snapshots,_,download,_=_fake_hub(tmp_path)
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules,"piper",None)
    monkeypatch.setattr(setup_local,"_piper_status",lambda *_a,**_k: pytest.fail("MOSS does not depend on Piper files"))
    report=setup_local.preflight_mac_local("moss",voice_dir=tmp_path/"unused-voices")
    assert report["ready"] is True
    assert report["voice_dir"] is None
    assert report["piper"] == {"ready":True,"required":False}
    # Only the existing Hub-cache Parakeet model uses HF's cache resolver;
    # it is explicitly local-only, so even this call cannot use the network.
    parakeet=setup_local.LOCAL_MODEL_MANIFEST["parakeet"]
    assert snapshots == [{"repo_id":parakeet.repo_id,"revision":parakeet.revision,"local_files_only":True}]
    assert "Piper не требуется" in setup_local.format_setup_report(report)
    assert not (tmp_path/"unused-voices").exists()
