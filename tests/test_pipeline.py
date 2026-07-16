"""Смоук всего конвейера на dummy-движках: захват → VAD → STT → перевод →
TTS → вывод(null) → оверлей → история → экспорт."""
import asyncio

from uvt.app import Pipeline
from uvt.config import AppConfig
from uvt.events import SubtitleEvent


def _test_config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.mode = "voiceover"
    cfg.target_lang = "ru"
    cfg.plugin_dirs = []
    cfg.capture.backend = "dummy"
    cfg.capture.bursts = 2  # extra-ключ dummy-движка
    cfg.vad.engine = "energy"
    cfg.latency.preset = "ultra"
    cfg.stt.engine = "dummy"
    cfg.translation.engine = "dummy"
    cfg.tts.engine = "dummy"
    cfg.output.backend = "null"
    cfg.history.enabled = True
    cfg.history.dir = str(tmp_path)
    cfg.history.formats = ["srt", "json"]
    return cfg


async def test_full_pipeline_smoke(tmp_path):
    subtitles: list[SubtitleEvent] = []
    pipeline = Pipeline(_test_config(tmp_path), subtitle_sink=subtitles.append)
    history = pipeline.get_service("history")

    await pipeline.start()

    async def wait_for_entries():
        while len(history.entries) < 2:
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(wait_for_entries(), timeout=20)
    finally:
        await pipeline.stop()

    # Перевод дошёл до конца конвейера
    assert history.entries[0].original == "hello 1"
    assert history.entries[0].translated == "HELLO 1 [ru]"
    assert history.entries[1].translated == "HELLO 2 [ru]"
    assert len(subtitles) >= 2
    assert subtitles[0].translated == "HELLO 1 [ru]"

    # Метрики прошли все стадии (speech_end → … → play)
    assert "total" in pipeline.metrics.snapshot()

    # История выгружена на диск при остановке
    session_dirs = list(tmp_path.iterdir())
    assert len(session_dirs) == 1
    srt = (session_dirs[0] / "session.srt").read_text(encoding="utf-8")
    assert "-->" in srt and "HELLO 1 [ru]" in srt
    assert (session_dirs[0] / "session.json").is_file()


async def test_subtitles_mode_has_no_tts(tmp_path):
    cfg = _test_config(tmp_path)
    cfg.mode = "subtitles"
    cfg.history.enabled = False
    pipeline = Pipeline(cfg)
    names = [service.name for service in pipeline.services]
    assert "tts" not in names and "output" not in names
    assert "overlay" in names
