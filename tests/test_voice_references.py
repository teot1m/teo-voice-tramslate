"""Private F5 reference library and bounded authenticated neutral-audio uploads."""
from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from uvt.config import AppConfig
from uvt import voice_references as refs


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("UVT_VOICE_LIBRARY_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("UVT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("UVT_API_TOKEN", raising=False)


def neutral_wav(seconds=6):
    rate = 8000
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    samples = .05 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def saved_reference(*, number="a", text="Neutral test sample."):
    root = refs.library_dir()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    voice_id = "ref_" + number * 32
    path = root / (voice_id + ".wav")
    path.write_bytes(neutral_wav())
    metadata = root / (voice_id + ".json")
    data = {"id": voice_id, "text": text, "label": "My sample", "gender": "female", "language": "ru", "languages": ["ru"]}
    metadata.write_text(json.dumps(data), encoding="utf-8")
    return voice_id, path, metadata


def test_library_resolves_owned_sample_and_reports_catalog():
    voice_id, path, _ = saved_reference()
    assert refs.resolve_reference(voice_id) == (path, "Neutral test sample.")
    [voice] = refs.catalog()
    assert voice["id"] == voice_id
    assert voice["gender"] == "female"
    assert voice["languages"] == ["ru"]
    assert voice["engine"] == "f5"
    assert voice["installed"] is True
    assert "path" not in voice
    assert "text" not in voice


@pytest.mark.parametrize("raw", ["{not json", "null", "[]", "42", '"sample"'])
def test_library_rejects_corrupt_metadata_and_catalog_skips_it(raw):
    voice_id, _, metadata = saved_reference()
    metadata.write_text(raw)
    with pytest.raises(ValueError):
        refs.resolve_reference(voice_id)
    assert refs.catalog() == []


@pytest.mark.parametrize("text", ["", "   ", None, 1])
def test_library_rejects_missing_transcript(text):
    voice_id, _, _ = saved_reference(text=text)
    with pytest.raises(ValueError):
        refs.resolve_reference(voice_id)
    assert refs.catalog() == []


@pytest.mark.parametrize("voice_id", ["../private", "ref_" + "a" * 32 + "/../x", "/tmp/sample", "not-a-reference", "ref_"])
def test_reference_identifier_cannot_address_arbitrary_files(voice_id):
    with pytest.raises(ValueError):
        refs.resolve_reference(voice_id)


def test_library_skips_removed_audio_and_oversized_metadata():
    voice_id, path, metadata = saved_reference()
    path.unlink()
    with pytest.raises(ValueError):
        refs.resolve_reference(voice_id)
    assert refs.catalog() == []
    path.write_bytes(neutral_wav())
    metadata.write_bytes(b"x" * 16001)
    with pytest.raises(ValueError):
        refs.resolve_reference(voice_id)
    assert refs.catalog() == []


@pytest.mark.parametrize("link_part", ["audio", "metadata"])
def test_library_rejects_symlink_outside_library(tmp_path, link_part):
    voice_id, path, metadata = saved_reference()
    original = path if link_part == "audio" else metadata
    outside = tmp_path / ("outside.wav" if link_part == "audio" else "outside.json")
    original.rename(outside)
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="вне библиотеки"):
        refs.resolve_reference(voice_id)
    assert refs.catalog() == []


@asynccontextmanager
async def reference_client(tmp_path, token=""):
    from uvt.server import DubServer
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.stt.engine = cfg.translation.engine = cfg.tts.engine = "dummy"
    server = DubServer(cfg)
    server.api_token = token
    server.audio_dir = tmp_path / "audio"
    server.audio_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        yield server, client
    finally:
        await client.close()


def upload_form(data=None, **overrides):
    form = FormData()
    form.add_field("audio", neutral_wav() if data is None else data, filename="../../chosen.wav", content_type="audio/wav")
    fields = {"text": "A neutral reference sample for the test.", "label": "Chosen voice", "gender": "female", "language": "ru"}
    fields.update(overrides)
    for key, value in fields.items():
        if value is not None:
            form.add_field(key, value)
    return form


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg")
@pytest.mark.asyncio
async def test_upload_real_neutral_wav_normalizes_audio_and_keeps_private_files(tmp_path):
    async with reference_client(tmp_path) as (_, client):
        response = await client.post("/voices/reference", data=upload_form())
        assert response.status == 201, await response.text()
        voice = (await response.json())["voice"]
        path, text = refs.resolve_reference(voice["id"])
        assert text == "A neutral reference sample for the test."
        info = sf.info(path)
        assert info.samplerate == 24000
        assert info.channels == 1
        assert info.duration == pytest.approx(6, abs=.02)
        assert path.parent == refs.library_dir()
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.with_suffix(".json").stat().st_mode & 0o777 == 0o600
        assert refs.library_dir().stat().st_mode & 0o777 == 0o700
        assert not list(refs.library_dir().glob("upload-*"))
        assert not (tmp_path / "chosen.wav").exists()
        assert voice["id"] in [item["id"] for item in refs.catalog()]


@pytest.mark.parametrize("headers,token", [({"Origin": "https://untrusted.example"}, ""), ({}, "private-token"), ({"X-UVT-Token": "wrong"}, "private-token")])
@pytest.mark.asyncio
async def test_reference_upload_requires_dashboard_authority(tmp_path, headers, token):
    async with reference_client(tmp_path, token) as (_, client):
        response = await client.post("/voices/reference", data=upload_form(), headers=headers)
        assert response.status == (401 if token else 403)
        assert refs.catalog() == []


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg")
@pytest.mark.asyncio
async def test_reference_upload_accepts_configured_api_token(tmp_path):
    async with reference_client(tmp_path, "private-token") as (_, client):
        response = await client.post("/voices/reference", data=upload_form(), headers={"X-UVT-Token": "private-token"})
        assert response.status == 201, await response.text()


@pytest.mark.parametrize("text", [None, "", "   ", "x" * 2001])
@pytest.mark.asyncio
async def test_upload_requires_transcript_and_does_not_save_partial_files(tmp_path, text):
    async with reference_client(tmp_path) as (_, client):
        response = await client.post("/voices/reference", data=upload_form(text=text))
        assert response.status == 400, await response.text()
        assert refs.catalog() == []
        assert not list(refs.library_dir().iterdir())


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg")
@pytest.mark.parametrize("seconds", [.5, 31])
@pytest.mark.asyncio
async def test_upload_rejects_reference_outside_duration_limit(tmp_path, seconds):
    async with reference_client(tmp_path) as (_, client):
        response = await client.post("/voices/reference", data=upload_form(neutral_wav(seconds)))
        assert response.status == 422, await response.text()
        assert refs.catalog() == []
        assert not list(refs.library_dir().iterdir())


@pytest.mark.asyncio
async def test_upload_rejects_file_over_20_mib_and_cleans_partial_upload(tmp_path):
    async with reference_client(tmp_path) as (_, client):
        response = await client.post("/voices/reference", data=upload_form(b"x" * (20 * 1024**2 + 1)))
        assert response.status == 413, await response.text()
        assert refs.catalog() == []
        assert not list(refs.library_dir().iterdir())
