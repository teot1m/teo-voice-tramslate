"""Дубляж файлов: аудио и мукс видео (нужен ffmpeg в PATH)."""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from uvt.config import AppConfig
from uvt.dub import dub_file, render_dub_track
from uvt.interfaces import TranslationEngine
from uvt.registry import register

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")


@register("translation", "test-flaky-batch")
class FlakyBatchTranslator(TranslationEngine):
    """Пакетный режим всегда падает — проверяем деградацию до пореплечного."""

    async def translate(self, text, source_lang, target_lang, context):
        return f"{text.upper()} [{target_lang}]"

    async def translate_batch(self, texts, source_lang, target_lang):
        raise RuntimeError("batch endpoint down")


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


def _speech_wav(path, rate: int = 44100) -> None:
    """Две «реплики»: тишина 0.5с → тон 1с → тишина 1с → тон 1с → тишина 0.5с."""
    t = np.arange(rate) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = lambda s: np.zeros(int(rate * s), dtype=np.float32)  # noqa: E731
    sf.write(path, np.concatenate([silence(0.5), tone, silence(1.0), tone, silence(0.5)]), rate)


async def test_dub_audio_file(tmp_path):
    src = tmp_path / "in.wav"
    _speech_wav(src)
    out = await dub_file(_cfg(), src, tmp_path / "out.wav")

    assert out.is_file()
    mixed, rate = sf.read(out)
    assert rate == 48000
    assert len(mixed) >= 48000 * 3.5  # дорожка не короче оригинала

    srt = (tmp_path / "out.srt").read_text(encoding="utf-8")
    assert "-->" in srt
    assert "HELLO 1 [ru]" in srt and "HELLO 2 [ru]" in srt
    assert (tmp_path / "out.json").is_file()


async def test_dub_batch_failure_degrades_to_single(tmp_path):
    """Пачка перевода упала → реплики переводятся по одной, дубляж выходит."""
    src = tmp_path / "in.wav"
    _speech_wav(src)
    cfg = _cfg()
    cfg.translation.engine = "test-flaky-batch"
    out = await dub_file(cfg, src, tmp_path / "out.wav")
    srt = (tmp_path / "out.srt").read_text(encoding="utf-8")
    assert "HELLO 1 [ru]" in srt and "HELLO 2 [ru]" in srt


async def test_voice_only_track_for_browser(tmp_path):
    """Для браузера дорожка содержит только голос перевода: моно, тишина в начале."""
    src = tmp_path / "in.wav"
    _speech_wav(src)
    track, entries = await render_dub_track(_cfg(), src, mix_original=False)
    assert track.ndim == 1
    assert len(entries) == 2
    assert np.max(np.abs(track[: 48000 // 10])) < 1e-6  # первые 100 мс — тишина
    assert np.max(np.abs(track)) > 0.05  # а голос в дорожке есть
    # Browser больше не должен угадывать окно ducking по числу символов.
    assert all(entry.tts_start is not None and entry.tts_end is not None for entry in entries)
    assert all(entry.tts_end > entry.tts_start for entry in entries)


async def test_dub_same_language_fails_clearly(tmp_path):
    """Речь уже на целевом языке → понятная ошибка, а не пустой файл."""
    src = tmp_path / "in.wav"
    _speech_wav(src)
    cfg = _cfg()
    cfg.target_lang = "en"  # dummy-STT возвращает language="en"
    with pytest.raises(RuntimeError, match="языке перевода|целевом языке"):
        await dub_file(cfg, src, tmp_path / "out.wav")


async def test_dub_video_keeps_video_and_original(tmp_path):
    wav = tmp_path / "voice.wav"
    _speech_wav(wav)
    src = tmp_path / "in.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=4:size=128x72:rate=10",
            "-i", str(wav),
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
            str(src),
        ],
        check=True,
    )

    out = await dub_file(_cfg(), src, tmp_path / "out.mkv")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    kinds = probe.stdout.split()
    assert "video" in kinds
    assert kinds.count("audio") >= 2  # переведённая дорожка + оригинальная
