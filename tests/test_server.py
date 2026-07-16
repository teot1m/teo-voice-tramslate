"""Сервер браузерной кнопки: задача из локального файла → готовая дорожка."""
import asyncio
import shutil

import numpy as np
import pytest
import soundfile as sf

from uvt.config import AppConfig

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.plugin_dirs = []
    cfg.target_lang = "ru"
    cfg.vad.engine = "energy"
    cfg.latency.preset = "ultra"
    cfg.stt.engine = "dummy"
    cfg.translation.engine = "dummy"
    cfg.tts.engine = "dummy"
    return cfg


async def test_dub_job_from_file(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros(rate // 2, dtype=np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, np.concatenate([silence, tone, silence, silence]), rate)

    from uvt.server import DubServer

    server = DubServer(_cfg())
    server.audio_dir = tmp_path  # не пишем в общий кэш из тестов

    client = TestClient(TestServer(server.app()))
    await client.start_server()
    try:
        resp = await client.post("/dub", json={"file": str(src), "target_lang": "de"})
        assert resp.status == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        job_id = (await resp.json())["id"]

        info = None
        for _ in range(200):
            info = await (await client.get(f"/job/{job_id}")).json()
            if info["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)
        assert info is not None and info["status"] == "done", info

        # target_lang из запроса дошёл до перевода
        assert info["entries"][0]["translated"] == "HELLO 1 [de]"
        assert info["progress"] == 1.0

        audio = await client.get(info["audio_url"])
        assert audio.status == 200
        body = await audio.read()
        assert len(body) > 1000  # настоящий m4a, не пустышка

        # неизвестная задача → 404, но с CORS-заголовком
        missing = await client.get("/job/nope")
        assert missing.status == 404
        assert missing.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        await client.close()
