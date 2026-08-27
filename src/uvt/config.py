"""Конфигурация приложения: pydantic-модели, профили YAML, пресеты задержки.

Все секции движков допускают дополнительные ключи (extra="allow") — так
пользовательские плагины получают свои параметры без изменения ядра (ТЗ §15).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class LatencyPreset:
    """Пресеты задержки (ТЗ §11): пауза-отсечка, потолок сегмента, пред-буфер."""

    min_silence_ms: int
    max_segment_s: float
    pre_roll_ms: int


PRESETS: dict[str, LatencyPreset] = {
    "ultra": LatencyPreset(min_silence_ms=300, max_segment_s=5.0, pre_roll_ms=200),
    "balanced": LatencyPreset(min_silence_ms=500, max_segment_s=8.0, pre_roll_ms=300),
    "accuracy": LatencyPreset(min_silence_ms=800, max_segment_s=15.0, pre_roll_ms=400),
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class CaptureConfig(_Section):
    backend: str = "sounddevice"  # sounddevice | wasapi-loopback | dummy | плагин
    device: int | str | None = None  # None → устройство по умолчанию
    sample_rate: int | None = None  # None → нативная частота устройства
    channels: int | None = None


class VADConfig(_Section):
    engine: str = "silero"  # silero | webrtc | energy | плагин
    threshold: float = 0.5
    min_speech_ms: int = 250
    # None → значение берётся из пресета задержки (latency.preset)
    min_silence_ms: int | None = None
    pre_roll_ms: int | None = None
    max_segment_s: float | None = None
    model_path: str | None = None  # свой путь к silero_vad.onnx


class STTFallbackConfig(_Section):
    """Локальный резерв для облачного STT.

    Используется только после ошибки удалённого openai-compatible движка.
    По умолчанию резерв выключен: облачные профили включают его явно, чтобы
    чужой пользовательский endpoint не менял маршрут без его выбора.
    """

    engine: str = "faster-whisper"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1


class STTConfig(_Section):
    engine: str = "faster-whisper"  # faster-whisper | openai-compatible | dummy
    model: str = "small"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"
    beam_size: int = 1
    # Batch fallback через VAD: облачные короткие сегменты можно распознавать
    # параллельно; локальный Whisper по умолчанию остаётся последовательным.
    concurrency: int | None = Field(default=None, ge=1)
    # Для openai-compatible (OpenAI, Groq и любые совместимые endpoint)
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    fallback: STTFallbackConfig | None = None


class TranslationFallbackConfig(_Section):
    """Локальный резерв для облачного LLM-перевода (Ollama/LM Studio)."""

    engine: str = "openai-compatible"
    base_url: str = "http://127.0.0.1:11434/v1"
    # Не наследуем OPENAI_API_KEY: локальному серверу он не нужен и не должен
    # случайно попасть в заголовок Authorization.
    api_key_env: str = "UVT_OLLAMA_API_KEY"
    model: str = "qwen2.5:3b"
    temperature: float = 0.3
    timeout_s: float = 90.0
    prompt_template: str | None = None
    context_pairs: int = 3
    glossary: list[str] = Field(default_factory=list)
    # Маленькие локальные LLM надёжнее держат короткие пачки и один запрос
    # одновременно. Это также ограничивает повтор cloud-пачки после failover.
    batch_size: int = Field(default=4, ge=1)
    concurrency: int = Field(default=1, ge=1)


class TranslationConfig(_Section):
    # openai-compatible | nllb-ct2 | translategemma-mlx | none | dummy
    engine: str = "openai-compatible"
    base_url: str = "http://localhost:11434/v1"  # по умолчанию — локальный Ollama
    model: str = "qwen2.5:7b-instruct"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.3
    timeout_s: float = 60.0  # для пакетного перевода растёт с размером пачки
    # None → встроенный шаблон; путь к файлу или сам текст шаблона (ТЗ §5)
    prompt_template: str | None = None
    context_pairs: int = 3  # сколько прошлых реплик отдавать LLM как контекст
    glossary: list[str] = Field(default_factory=list)
    # Для batch-дубляжа: размер одной пачки и число одновременных запросов.
    # Локальный Ollama по умолчанию работает последовательно, облако — параллельно.
    batch_size: int | None = Field(default=None, ge=1)
    concurrency: int | None = Field(default=None, ge=1)
    fallback: TranslationFallbackConfig | None = None


class TTSConfig(_Section):
    engine: str = "edge"  # edge | openai | elevenlabs | kokoro | piper | none | dummy
    voice: str = "auto"  # auto → голос по целевому языку
    rate: str = "+0%"  # темп речи для edge-tts
    speed: float = 1.0  # темп для kokoro
    # Минимальная пауза между стартами сетевых TTS-запросов. Для локальных
    # движков остаётся 0; ElevenLabs может задать её для бесплатного тарифа.
    request_interval_s: float = Field(default=0.0, ge=0.0)
    model_path: str | None = None
    voices_path: str | None = None
    # ``auto`` выбирается SpeakerService для каждой реплики; явное male/female
    # сохраняет прежнее ручное поведение и побеждает авто-определение.
    voice_gender: str = "auto"
    # Необязательный идентификатор одной из моделей из ``voice_models``.
    # Это именно allowlisted stem файла, а не путь от HTTP-клиента.
    voice_id: str | None = None


class SpeakerConfig(_Section):
    """Лёгкая локальная привязка реплик к спикерам для live TTS.

    Это не биометрическая идентификация и не определение личности: состояние
    живёт только в памяти сессии и использует тембр/F0, чтобы голос перевода
    не сваливался в один мужской default.
    """

    enabled: bool = True
    max_speakers: int = 8
    match_threshold: float = 0.42
    # Явные правки для конкретных session IDs, например
    # {"speaker-1": "female", "speaker-2": "male"}.
    voice_map: dict[str, str] = Field(default_factory=dict)
    # Когда F0 неуверен, роли назначаются стабильно по очереди, а не
    # наследуют прежний male voice.
    fallback_voice_roles: list[str] = Field(default_factory=lambda: ["female", "male"])


class OutputConfig(_Section):
    backend: str = "sounddevice"  # sounddevice | null
    device: int | str | None = None  # наушники, VB-Cable, BlackHole → OBS/Discord
    sample_rate: int = 24000
    max_backlog_s: float = 8.0  # допустимая просрочка target, после которой пропускаем
    # От конца исходной реплики до запланированного старта перевода. Небольшой
    # запас даёт STT/MT/TTS шанс закончить, но не превращает live в batch.
    target_delay_s: float = 0.75
    # Обработчики тяжёлых стадий берут последний сегмент из очереди; старые
    # выбрасываются до следующего сетевого/модельного вызова.
    latest_wins: bool = True


class OverlayConfig(_Section):
    enabled: bool = True
    show: str = "both"  # original | translation | both (ТЗ §8)
    font_size: int = 20
    opacity: float = 0.85
    color: str = "#FFFFFF"
    position: str = "bottom"  # top | bottom


class HistoryConfig(_Section):
    enabled: bool = True
    dir: str = "~/UVT/history"
    formats: list[str] = Field(default_factory=lambda: ["srt", "txt", "json"])


class LatencyConfig(_Section):
    preset: str = "balanced"  # ultra | balanced | accuracy


class AppConfig(_Section):
    # Режимы (ТЗ §9): subtitles — только субтитры; voiceover — озвучка поверх;
    # replace / dual — пока работают как voiceover (см. ROADMAP в README)
    mode: str = "voiceover"
    source_lang: str = "auto"
    target_lang: str = "ru"
    plugin_dirs: list[str] = Field(default_factory=lambda: ["plugins"])

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_profile(profile: str) -> Path:
    candidates = [
        Path(profile),
        Path("profiles") / f"{profile}.yaml",
        Path.home() / ".config" / "uvt" / "profiles" / f"{profile}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"профиль '{profile}' не найден (искал: {searched})")


def load_config(profile: str | None = None, overrides: dict | None = None) -> AppConfig:
    """Собирает конфиг: значения по умолчанию ← профиль YAML ← переопределения CLI."""
    data: dict = {}
    if profile:
        path = _resolve_profile(profile)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"профиль {path} должен быть YAML-словарём")
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)
