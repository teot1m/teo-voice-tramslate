"""Persisted, allowlisted settings for the personal web dashboard.

The dashboard may choose models and voices, but it never receives or stores
provider API keys.  Settings are deliberately limited to values that can be
applied to an already configured route; engine types, base URLs and key names
remain owned by the YAML profile.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from uvt.config import AppConfig

SETTINGS_VERSION = 1

OPENAI_STT_MODELS = (
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "whisper-1",
)
OPENAI_TRANSLATION_MODELS = (
    "gpt-4o-mini",
    "gpt-4o",
)
OPENAI_TTS_MODELS = (
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
)
OPENAI_LEGACY_VOICES = (
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
)
OPENAI_MINI_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
OPENAI_VOICE_MODELS = {
    "tts-1": OPENAI_LEGACY_VOICES,
    "tts-1-hd": OPENAI_LEGACY_VOICES,
    "gpt-4o-mini-tts": OPENAI_MINI_TTS_VOICES,
}
OPENAI_VOICES = tuple(dict.fromkeys((*OPENAI_LEGACY_VOICES, *OPENAI_MINI_TTS_VOICES)))
ELEVENLABS_TTS_MODELS = (
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
)

LANGUAGES = (
    ("auto", "Авто"),
    ("ru", "Русский"),
    ("uk", "Украинский"),
    ("en", "Английский"),
    ("de", "Немецкий"),
    ("fr", "Французский"),
    ("es", "Испанский"),
    ("it", "Итальянский"),
    ("pt", "Португальский"),
    ("pl", "Польский"),
    ("bg", "Болгарский"),
    ("cs", "Чешский"),
    ("da", "Датский"),
    ("el", "Греческий"),
    ("et", "Эстонский"),
    ("fi", "Финский"),
    ("hr", "Хорватский"),
    ("hu", "Венгерский"),
    ("lt", "Литовский"),
    ("lv", "Латышский"),
    ("mt", "Мальтийский"),
    ("nl", "Нидерландский"),
    ("ro", "Румынский"),
    ("sk", "Словацкий"),
    ("sl", "Словенский"),
    ("sv", "Шведский"),
    ("ja", "Японский"),
    ("zh", "Китайский"),
    ("ko", "Корейский"),
    ("tr", "Турецкий"),
    ("ar", "Арабский"),
    ("hi", "Хинди"),
)

_LANGUAGE_IDS = {item[0] for item in LANGUAGES}
PARAKEET_LANGUAGE_IDS = {
    "auto",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "uk",
}
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_VOICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROUTE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class SettingsConflictError(ValueError):
    """The dashboard tried to overwrite a newer settings revision."""


def default_settings_path() -> Path:
    explicit = os.environ.get("UVT_SETTINGS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    root = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    ).expanduser()
    return root / "uvt" / "server-settings.json"


def route_key(label: str) -> str:
    known = {
        "free": "free",
        "local": "free",
        "gpt": "cloud",
        "cloud": "cloud",
        "elevenlabs": "eleven",
        "eleven": "eleven",
    }
    normalized = str(label or "uvt").strip().lower()
    if normalized in known:
        return known[normalized]
    slug = re.sub(r"[^a-z0-9._-]+", "-", normalized).strip("-") or "uvt"
    return slug[:64]


class ServerSettingsStore:
    """Small process-shared JSON store with atomic, permission-tight writes."""

    def __init__(self, path: Path | None = None, *, persistent: bool = False) -> None:
        self.path = path
        self.persistent = bool(persistent and path is not None)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"version": SETTINGS_VERSION, "routes": {}}
        self.load_error: str | None = None
        if self.persistent:
            self._load()

    @classmethod
    def memory(cls) -> "ServerSettingsStore":
        return cls()

    @classmethod
    def default(cls) -> "ServerSettingsStore":
        return cls(default_settings_path(), persistent=True)

    def _load(self) -> None:
        assert self.path is not None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            # A hand-edited or interrupted settings file must not make all
            # three personal routes unavailable.  Keep the shipped defaults
            # and surface the exact problem through GET /settings instead.
            self.load_error = f"не удалось прочитать {self.path}: {exc}"
            return
        if not isinstance(raw, dict) or raw.get("version") != SETTINGS_VERSION:
            self.load_error = f"неподдерживаемый формат: {self.path}"
            return
        routes = raw.get("routes")
        if not isinstance(routes, dict):
            self.load_error = f"повреждён раздел routes в {self.path}"
            return
        self._data = {"version": SETTINGS_VERSION, "routes": routes}

    def _write(self, data: dict[str, Any] | None = None) -> None:
        if not self.persistent:
            return
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        payload = json.dumps(
            data if data is not None else self._data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        try:
            temp.write_text(payload + "\n", encoding="utf-8")
            os.chmod(temp, 0o600)
            temp.replace(self.path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def get_entry(self, key: str) -> dict[str, Any]:
        if not _ROUTE_RE.fullmatch(key):
            return {"revision": 0, "saved": False, "settings": None}
        with self._lock:
            value = dict(self._data.get("routes", {})).get(key)
            if not isinstance(value, dict):
                return {"revision": 0, "saved": False, "settings": None}
            # Accept the short-lived development format where the route value
            # was the settings object itself.
            if "settings" not in value:
                return {
                    "revision": 1,
                    "saved": True,
                    "settings": copy.deepcopy(value),
                }
            settings = value.get("settings")
            try:
                revision = max(0, int(value.get("revision") or 0))
            except (TypeError, ValueError):
                self.load_error = "в файле настроек повреждена revision"
                return {"revision": 0, "saved": False, "settings": None}
            return {
                "revision": revision,
                "saved": isinstance(settings, dict),
                "settings": copy.deepcopy(settings) if isinstance(settings, dict) else None,
            }

    def get(self, key: str) -> dict[str, Any] | None:
        return self.get_entry(key)["settings"]

    def set(
        self,
        key: str,
        value: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if not _ROUTE_RE.fullmatch(key):
            raise ValueError("некорректный идентификатор маршрута")
        with self._lock:
            current = self.get_entry(key)
            if expected_revision is not None and expected_revision != current["revision"]:
                raise SettingsConflictError(
                    "настройки уже изменились в другой вкладке; обновите форму"
                )
            staged = copy.deepcopy(self._data)
            routes = staged.setdefault("routes", {})
            routes[key] = {
                "revision": current["revision"] + 1,
                "settings": copy.deepcopy(value),
            }
            self._write(staged)
            self._data = staged
            self.load_error = None
            return self.get_entry(key)

    def reset(
        self, key: str, *, expected_revision: int | None = None
    ) -> dict[str, Any]:
        if not _ROUTE_RE.fullmatch(key):
            raise ValueError("некорректный идентификатор маршрута")
        with self._lock:
            current = self.get_entry(key)
            if expected_revision is not None and expected_revision != current["revision"]:
                raise SettingsConflictError(
                    "настройки уже изменились в другой вкладке; обновите форму"
                )
            staged = copy.deepcopy(self._data)
            routes = staged.setdefault("routes", {})
            routes[key] = {"revision": current["revision"] + 1, "settings": None}
            self._write(staged)
            self._data = staged
            self.load_error = None
            return self.get_entry(key)


def settings_kind(cfg: AppConfig, *, selectable_profiles: bool = False) -> str:
    if selectable_profiles or str(cfg.tts.engine) == "piper":
        return "local"
    if str(cfg.tts.engine) == "elevenlabs":
        return "elevenlabs"
    if str(cfg.tts.engine) == "openai":
        return "openai"
    return "generic"


def _clean_language(value: object, *, source: bool) -> str:
    normalized = str(value or ("auto" if source else "ru")).strip().lower()
    allowed = _LANGUAGE_IDS if source else _LANGUAGE_IDS - {"auto"}
    if normalized not in allowed:
        raise ValueError(f"неподдерживаемый язык: {normalized}")
    return normalized


def _clean_model(value: object, label: str) -> str:
    normalized = str(value or "").strip()
    if not _MODEL_RE.fullmatch(normalized):
        raise ValueError(f"{label}: некорректное имя модели")
    return normalized


def _clean_voice(value: object, label: str, *, allow_auto: bool = True) -> str:
    normalized = str(value or ("auto" if allow_auto else "")).strip()
    if allow_auto and normalized == "auto":
        return normalized
    if not _VOICE_RE.fullmatch(normalized):
        raise ValueError(f"{label}: некорректный идентификатор голоса")
    return normalized


def effective_settings(
    cfg: AppConfig,
    *,
    kind: str,
    profile_name: str,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_lang": str(cfg.source_lang or "auto"),
        "target_lang": str(cfg.target_lang or "ru"),
        "voice_gender": str(cfg.tts.voice_gender or "auto"),
    }
    if kind == "local":
        data.update(
            {
                "profile_id": profile_name,
                "voice_id": str(cfg.tts.voice_id or ""),
            }
        )
    else:
        data.update(
            {
                "stt_model": str(getattr(cfg.stt, "model", "") or ""),
                "translation_model": str(
                    getattr(cfg.translation, "model", "") or ""
                ),
                "tts_model": str(getattr(cfg.tts, "model", "") or ""),
                "tts_voice": str(getattr(cfg.tts, "voice", "auto") or "auto"),
            }
        )
    return data


def normalize_settings(
    payload: dict[str, Any],
    *,
    current: dict[str, Any],
    kind: str,
    profile_ids: set[str],
    local_voices: list[dict[str, Any]],
) -> dict[str, Any]:
    common_fields = {"source_lang", "target_lang", "voice_gender"}
    kind_fields = (
        {"profile_id", "voice_id"}
        if kind == "local"
        else {"stt_model", "translation_model", "tts_model", "tts_voice"}
    )
    unknown = set(payload) - common_fields - kind_fields
    if unknown:
        raise ValueError("неизвестные поля настроек: " + ", ".join(sorted(unknown)))
    for field_name, value in payload.items():
        if not isinstance(value, str):
            raise ValueError(f"{field_name} должно быть строкой")
    merged = {**current, **payload}
    normalized: dict[str, Any] = {
        "source_lang": _clean_language(merged.get("source_lang"), source=True),
        "target_lang": _clean_language(merged.get("target_lang"), source=False),
    }
    gender = str(merged.get("voice_gender") or "auto").strip().lower()
    if gender not in {"auto", "male", "female"}:
        raise ValueError("voice_gender должен быть auto, male или female")
    normalized["voice_gender"] = gender

    if kind == "local":
        profile_id = str(merged.get("profile_id") or "").strip()
        if profile_id not in profile_ids:
            raise ValueError("выбран неизвестный локальный профиль")
        normalized["profile_id"] = profile_id
        available_languages = {
            str(item.get("language"))
            for item in local_voices
            if item.get("installed") is True
        }
        if normalized["target_lang"] not in available_languages:
            raise ValueError(
                "для целевого языка нет установленного локального голоса"
            )
        requested_voice = str(merged.get("voice_id") or "").strip()
        if requested_voice:
            voice = next(
                (item for item in local_voices if item.get("id") == requested_voice),
                None,
            )
            if voice is None or voice.get("installed") is False:
                raise ValueError("выбранный локальный голос не установлен")
            if str(voice.get("language")) != normalized["target_lang"]:
                raise ValueError("локальный голос не подходит для целевого языка")
            normalized["voice_gender"] = str(voice.get("gender") or gender)
        normalized["voice_id"] = requested_voice
        return normalized

    normalized["stt_model"] = _clean_model(
        merged.get("stt_model"), "модель распознавания"
    )
    normalized["translation_model"] = _clean_model(
        merged.get("translation_model"), "модель перевода"
    )
    normalized["tts_model"] = _clean_model(
        merged.get("tts_model"), "модель озвучки"
    )
    if normalized["stt_model"] not in OPENAI_STT_MODELS:
        raise ValueError("эта модель распознавания не разрешена")
    if normalized["translation_model"] not in OPENAI_TRANSLATION_MODELS:
        raise ValueError("эта модель перевода не разрешена")
    allowed_tts_models = (
        OPENAI_TTS_MODELS if kind == "openai" else ELEVENLABS_TTS_MODELS
    )
    if kind in {"openai", "elevenlabs"} and normalized["tts_model"] not in allowed_tts_models:
        raise ValueError("эта модель озвучки не разрешена")
    voice = _clean_voice(merged.get("tts_voice"), "голос")
    if kind == "openai" and voice != "auto":
        compatible = OPENAI_VOICE_MODELS.get(normalized["tts_model"], ())
        if voice not in compatible:
            raise ValueError("выбранный голос не поддерживается этой моделью OpenAI")
    normalized["tts_voice"] = voice
    return normalized


def apply_settings(
    base_cfg: AppConfig,
    settings: dict[str, Any] | None,
    *,
    kind: str,
    base_profile_name: str,
    profiles: dict[str, AppConfig],
) -> tuple[AppConfig, str]:
    if not settings:
        return base_cfg.model_copy(deep=True), base_profile_name

    profile_name = base_profile_name
    if kind == "local":
        profile_name = str(settings.get("profile_id") or base_profile_name)
        selected = profiles.get(profile_name)
        if selected is None:
            return base_cfg.model_copy(deep=True), base_profile_name
        cfg = selected.model_copy(deep=True)
    else:
        cfg = base_cfg.model_copy(deep=True)

    cfg.source_lang = str(settings.get("source_lang") or cfg.source_lang)
    cfg.target_lang = str(settings.get("target_lang") or cfg.target_lang)
    cfg.tts.voice_gender = str(
        settings.get("voice_gender") or cfg.tts.voice_gender or "auto"
    )
    if kind == "local":
        cfg.tts.voice_id = str(settings.get("voice_id") or "") or None
    else:
        cfg.stt.model = str(settings.get("stt_model") or cfg.stt.model)
        cfg.translation.model = str(
            settings.get("translation_model") or cfg.translation.model
        )
        cfg.tts.model = str(settings.get("tts_model") or getattr(cfg.tts, "model", ""))
        cfg.tts.voice = str(settings.get("tts_voice") or cfg.tts.voice or "auto")
    return cfg, profile_name


def settings_catalog(
    cfg: AppConfig,
    *,
    kind: str,
    profiles: list[dict[str, Any]],
    voices: list[dict[str, Any]],
) -> dict[str, Any]:
    all_languages = [{"id": item[0], "label": item[1]} for item in LANGUAGES]
    source_ids = (
        PARAKEET_LANGUAGE_IDS
        if kind == "local" and str(cfg.stt.engine) == "parakeet-mlx"
        else _LANGUAGE_IDS
    )
    target_ids = (
        {str(voice.get("language")) for voice in voices if voice.get("installed") is True}
        if kind == "local"
        else _LANGUAGE_IDS - {"auto"}
    )
    catalog: dict[str, Any] = {
        "all_source_languages": all_languages,
        "source_languages": [item for item in all_languages if item["id"] in source_ids],
        "target_languages": [item for item in all_languages if item["id"] in target_ids],
        "voice_genders": [
            {"id": "auto", "label": "Авто по спикеру"},
            {"id": "male", "label": "Мужской"},
            {"id": "female", "label": "Женский"},
        ],
    }
    if kind == "local":
        catalog.update({"profiles": profiles, "voices": voices})
    elif kind == "openai":
        catalog.update(
            {
                "stt_models": list(OPENAI_STT_MODELS),
                "translation_models": list(OPENAI_TRANSLATION_MODELS),
                "tts_models": list(OPENAI_TTS_MODELS),
                "voices": [
                    {
                        "id": voice,
                        "label": voice.capitalize(),
                        "models": [
                            model
                            for model, model_voices in OPENAI_VOICE_MODELS.items()
                            if voice in model_voices
                        ],
                    }
                    for voice in OPENAI_VOICES
                ],
            }
        )
    elif kind == "elevenlabs":
        catalog.update(
            {
                "stt_models": list(OPENAI_STT_MODELS),
                "translation_models": list(OPENAI_TRANSLATION_MODELS),
                "tts_models": [
                    {
                        "id": "eleven_multilingual_v2",
                        "label": "Multilingual v2 — качество (проверено UVT)",
                    },
                    {
                        "id": "eleven_flash_v2_5",
                        "label": "Flash v2.5 — быстрее (экспериментально)",
                    },
                    {
                        "id": "eleven_turbo_v2_5",
                        "label": "Turbo v2.5 — баланс (экспериментально)",
                    },
                ],
                "voices": [],
            }
        )
    else:
        catalog.update(
            {
                "stt_models": [str(getattr(cfg.stt, "model", "") or "")],
                "translation_models": [
                    str(getattr(cfg.translation, "model", "") or "")
                ],
                "tts_models": [str(getattr(cfg.tts, "model", "") or "")],
                "voices": [],
            }
        )
    return catalog
