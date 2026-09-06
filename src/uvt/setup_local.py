"""Pinned, explicit model setup for the local Apple Silicon profiles.

Jobs must never discover and download model revisions on their own. This
module is the only supported download boundary: every Hugging Face artifact is
addressed by an immutable commit and each shipped profile sets
``allow_download: false``. ``preflight_mac_local`` uses the Hub cache in
``local_files_only`` mode and therefore performs no network access.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("uvt.setup.local")


@dataclass(frozen=True)
class HuggingFaceModelSpec:
    """One immutable Hugging Face snapshot required by a local preset."""

    key: str
    role: str
    repo_id: str
    revision: str
    required_files: tuple[str, ...]
    # Репозитории с несколькими чекпойнтами качаются точечно: полный snapshot
    # весов F5 — это десятки гигабайт вместо нужной пары файлов.
    allow_patterns: tuple[str, ...] = ()
    local_dir: str | None = None


@dataclass(frozen=True)
class LocalSetupPreset:
    """Artifacts installed together for a documented local quality level."""

    name: str
    model_keys: tuple[str, ...]
    stt_key: str
    translation_key: str
    requires_piper: bool = True


PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
TRANSLATEGEMMA_REPO = "mlx-community/translategemma-4b-it-4bit"
TRANSLATEGEMMA_REVISION = "5788ec08c047f3f2e17808101b8d9566ac930d58"
NLLB_REPO = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
NLLB_REVISION = "70f572adafa4794890ce7826156a4209717855af"
MLX_WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
MLX_WHISPER_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

# Chat-модель для контекстного перевода диалогов: в отличие от TranslateGemma
# её промпт вмещает соседние реплики, пол говорящего и глоссарий.
QWEN_CHAT_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
QWEN_CHAT_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"

# Русский файнтюн F5-TTS: базовый чекпойнт обучен на английском и китайском,
# для ru нужен именно он. Веса и словарь берутся из одной папки репозитория —
# несогласованная пара даёт кашу вместо речи.
F5_RU_REPO = "Misha24-10/F5-TTS_RUSSIAN"
F5_RU_REVISION = "ea166adeae4c80ec5ee423a671e2bdb83906cf84"
F5_RU_MODEL_FILE = "F5TTS_v1_Base/model_240000_inference.safetensors"
F5_RU_VOCAB_FILE = "F5TTS_v1_Base/vocab.txt"

# Piper itself is installed as a Python dependency. Voice files are fetched
# here directly from an immutable Hub commit instead of using
# ``piper.download_voices``, whose default endpoint can move over time.
PIPER_REPO = "rhasspy/piper-voices"
PIPER_REVISION = "39ab474be869e9181350af6a65e4953eef67aaa0"
PIPER_VOICES = (
    "uk_UA-mykyta-high",
    "uk_UA-tetiana-high",
    "ru_RU-dmitri-medium",
    "ru_RU-irina-medium",
)
PIPER_VOICE_PATHS: dict[str, str] = {
    "uk_UA-mykyta-high": "uk/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx",
    "uk_UA-tetiana-high": "uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
    "ru_RU-dmitri-medium": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
    "ru_RU-irina-medium": "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
}
_PIPER_PROVENANCE = ".uvt-piper-manifest.json"


LOCAL_MODEL_MANIFEST: dict[str, HuggingFaceModelSpec] = {
    "parakeet": HuggingFaceModelSpec(
        key="parakeet",
        role="stt",
        repo_id=PARAKEET_REPO,
        revision=PARAKEET_REVISION,
        required_files=(
            "config.json",
            "model.safetensors",
            "tokenizer.model",
            "tokenizer.vocab",
            "vocab.txt",
        ),
    ),
    "nllb": HuggingFaceModelSpec(
        key="nllb",
        role="translation",
        repo_id=NLLB_REPO,
        revision=NLLB_REVISION,
        required_files=(
            "config.json",
            "model.bin",
            "shared_vocabulary.json",
            "tokenizer.json",
        ),
    ),
    "translategemma": HuggingFaceModelSpec(
        key="translategemma",
        role="translation",
        repo_id=TRANSLATEGEMMA_REPO,
        revision=TRANSLATEGEMMA_REVISION,
        required_files=(
            "config.json",
            "chat_template.jinja",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
    ),
    "whisper": HuggingFaceModelSpec(
        key="whisper",
        role="stt",
        repo_id=MLX_WHISPER_REPO,
        revision=MLX_WHISPER_REVISION,
        required_files=("config.json", "weights.safetensors"),
    ),
    "qwen3-chat": HuggingFaceModelSpec(
        key="qwen3-chat",
        role="translation",
        repo_id=QWEN_CHAT_REPO,
        revision=QWEN_CHAT_REVISION,
        required_files=(
            "config.json",
            "chat_template.jinja",
            "model.safetensors",
            "tokenizer.json",
        ),
    ),
    "f5-ru": HuggingFaceModelSpec(
        key="f5-ru",
        role="tts",
        repo_id=F5_RU_REPO,
        revision=F5_RU_REVISION,
        required_files=(F5_RU_MODEL_FILE, F5_RU_VOCAB_FILE),
        allow_patterns=(F5_RU_MODEL_FILE, F5_RU_VOCAB_FILE),
    ),
}

# New optional runtimes keep their pinned files inside this checkout.
LOCAL_MODEL_MANIFEST.update({
    "hymt2": HuggingFaceModelSpec(
        key="hymt2", role="translation", repo_id="mlx-community/Hy-MT2-1.8B-4bit",
        revision="e5c6fe56c7b3bc77fae5ae92db31f2178f1e6912", local_dir=".models/hymt2",
        required_files=("config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
        allow_patterns=("*.json", "*.safetensors", "*.jinja", "LICENSE.txt"),
    ),
    "nemotron": HuggingFaceModelSpec(
        key="nemotron", role="stt", repo_id="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        revision="7279359e4481b5e9e185a318bd618e429c6d86cd", local_dir=".models/nemotron",
        required_files=("config.json", "model.safetensors", "tokenizer.model", "vocab.txt"),
        allow_patterns=("config.json", "model.safetensors", "tokenizer.model", "vocab.txt", "README.md"),
    ),
    "moss-tts": HuggingFaceModelSpec(
        key="moss-tts", role="tts", repo_id="OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
        revision="f52645cb467506d8e18e746ddd59482685b74e58", local_dir=".models/moss-tts",
        required_files=("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "tokenizer.model",
            "moss_tts_prefill.onnx", "moss_tts_decode_step.onnx", "moss_tts_global_shared.data",
            "moss_tts_local_decoder.onnx", "moss_tts_local_cached_step.onnx",
            "moss_tts_local_fixed_sampled_frame.onnx", "moss_tts_local_shared.data"),
        allow_patterns=("*.json", "tokenizer.model", "*.onnx", "*.data", "LICENSE*", "README.md"),
    ),
    "moss-codec": HuggingFaceModelSpec(
        key="moss-codec", role="codec", repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        revision="ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae", local_dir=".models/moss-codec",
        required_files=("codec_browser_onnx_meta.json", "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_encode.data", "moss_audio_tokenizer_decode_full.onnx",
            "moss_audio_tokenizer_decode_step.onnx", "moss_audio_tokenizer_decode_shared.data"),
        allow_patterns=("*.json", "*.onnx", "*.data", "LICENSE*", "README.md"),
    ),
})


LOCAL_SETUP_PRESETS: dict[str, LocalSetupPreset] = {
    "fast": LocalSetupPreset(
        name="fast",
        model_keys=("parakeet", "nllb"),
        stt_key="parakeet",
        translation_key="nllb",
    ),
    "balanced": LocalSetupPreset(
        name="balanced",
        model_keys=("parakeet", "translategemma"),
        stt_key="parakeet",
        translation_key="translategemma",
    ),
    "quality": LocalSetupPreset(
        name="quality",
        model_keys=("whisper", "translategemma"),
        stt_key="whisper",
        translation_key="translategemma",
    ),
    "natural": LocalSetupPreset(
        name="natural",
        model_keys=("parakeet", "qwen3-chat", "f5-ru"),
        stt_key="parakeet",
        translation_key="qwen3-chat",
    ),
    "hymt": LocalSetupPreset("hymt", ("parakeet", "hymt2"), "parakeet", "hymt2"),
    "moss": LocalSetupPreset("moss", ("parakeet", "hymt2", "moss-tts", "moss-codec"), "parakeet", "hymt2", requires_piper=False),
    "nemotron": LocalSetupPreset("nemotron", ("nemotron", "hymt2"), "nemotron", "hymt2"),
    "all": LocalSetupPreset(
        name="all",
        model_keys=(
            "parakeet",
            "nllb",
            "translategemma",
            "whisper",
            "qwen3-chat",
            "f5-ru",
            "hymt2", "nemotron", "moss-tts", "moss-codec",
        ),
        stt_key="whisper",
        translation_key="translategemma",
    ),
}


def preset_for_profile(cfg: Any) -> str | None:
    """Return the shipped setup preset matching a configured local route."""
    pair = (
        str(getattr(getattr(cfg, "stt", None), "engine", "") or ""),
        str(
            getattr(getattr(cfg, "translation", None), "engine", "") or ""
        ),
    )
    tts_engine = str(getattr(getattr(cfg, "tts", None), "engine", "") or "")
    if tts_engine == "moss-onnx":
        return "moss"
    if pair == ("nemotron-mlx", "hymt-mlx"):
        return "nemotron"
    if pair == ("parakeet-mlx", "hymt-mlx"):
        return "hymt"
    if tts_engine == "f5":
        # Клонирующая озвучка требует собственных весов, поэтому маршрут
        # опознаётся по ней, а не только по паре STT+перевод.
        return "natural"
    return {
        ("parakeet-mlx", "nllb-ct2"): "fast",
        ("parakeet-mlx", "translategemma-mlx"): "balanced",
        ("mlx-whisper", "translategemma-mlx"): "quality",
    }.get(pair)


def _preset(name: str) -> LocalSetupPreset:
    normalized = str(name or "").strip().lower()
    try:
        return LOCAL_SETUP_PRESETS[normalized]
    except KeyError:
        choices = ", ".join(LOCAL_SETUP_PRESETS)
        raise ValueError(
            f"неизвестный local preset '{name}'; доступны: {choices}"
        ) from None


def _validate_snapshot(spec: HuggingFaceModelSpec, path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{spec.repo_id}: Hub вернул не-каталог {path}")
    # snapshot_download without local_dir returns .../snapshots/<commit>.
    if spec.local_dir:
        for filename in spec.required_files:
            metadata = path / ".cache/huggingface/download" / (filename + ".metadata")
            try:
                downloaded_revision = metadata.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError) as exc:
                raise RuntimeError(f"{spec.repo_id}: нет подтверждения ревизии для {filename}; повторите setup") from exc
            if downloaded_revision != spec.revision:
                raise RuntimeError(f"{spec.repo_id}: неверная ревизия {filename}; повторите setup")
    elif spec.revision not in path.parts:
        raise RuntimeError(
            f"{spec.repo_id}: ожидался snapshot {spec.revision}, получен {path}"
        )
    missing = [
        name for name in spec.required_files
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"{spec.repo_id}@{spec.revision}: нет или пустые файлы {', '.join(missing)}"
        )


def _model_status(
    spec: HuggingFaceModelSpec,
    snapshot_download: Callable[..., str],
    *,
    local_only: bool,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "key": spec.key,
        "role": spec.role,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "ready": False,
        "path": None,
        "error": None,
    }
    try:
        kwargs: dict[str, Any] = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "local_files_only": local_only,
        }
        if spec.allow_patterns:
            kwargs["allow_patterns"] = list(spec.allow_patterns)
        if spec.local_dir:
            # Resolve against installed editable project, not a caller's cwd.
            local_path = Path(__file__).resolve().parents[2] / spec.local_dir
            if local_only:
                path = local_path
            else:
                kwargs["local_dir"] = str(local_path)
                path = Path(snapshot_download(**kwargs))
        else:
            path = Path(snapshot_download(**kwargs))
        _validate_snapshot(spec, path)
    except Exception as exc:  # noqa: BLE001 - report missing/corrupt artifact
        status["error"] = str(exc)
        return status
    status["ready"] = True
    status["path"] = path
    return status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _piper_expected_files() -> tuple[str, ...]:
    files: list[str] = []
    for voice in PIPER_VOICES:
        files.extend((f"{voice}.onnx", f"{voice}.onnx.json"))
    return tuple(files)


def _read_piper_provenance(voice_dir: Path) -> dict[str, Any]:
    path = voice_dir / _PIPER_PROVENANCE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Piper: нет корректного {path.name}; повторите setup"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Piper: {path.name} должен содержать JSON-object")
    return data


def _validate_piper_install(voice_dir: Path) -> dict[str, Any]:
    data = _read_piper_provenance(voice_dir)
    if data.get("repo_id") != PIPER_REPO or data.get("revision") != PIPER_REVISION:
        raise RuntimeError(
            "Piper: голоса установлены из другой ревизии; повторите setup"
        )
    recorded = data.get("files")
    if not isinstance(recorded, dict):
        raise RuntimeError("Piper: в provenance нет карты files")

    for filename in _piper_expected_files():
        path = voice_dir / filename
        expected = recorded.get(filename)
        if not path.is_file() or not isinstance(expected, dict):
            raise RuntimeError(f"Piper: нет проверенного файла {filename}")
        size = expected.get("size")
        sha256 = expected.get("sha256")
        if path.stat().st_size != size or _sha256(path) != sha256:
            raise RuntimeError(f"Piper: checksum не совпал для {filename}")

    # Validate the sidecar field consumed by the Piper runtime before a job.
    for voice in PIPER_VOICES:
        sidecar = voice_dir / f"{voice}.onnx.json"
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            sample_rate = metadata["audio"]["sample_rate"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Piper: {sidecar.name} не содержит audio.sample_rate"
            ) from exc
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise RuntimeError(
                f"Piper: {sidecar.name} содержит неверный audio.sample_rate"
            )
    return data


def _piper_status(voice_dir: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "repo_id": PIPER_REPO,
        "revision": PIPER_REVISION,
        "ready": False,
        "voice_dir": voice_dir,
        "voices": list(PIPER_VOICES),
        "error": None,
    }
    try:
        _validate_piper_install(voice_dir)
    except Exception as exc:  # noqa: BLE001 - preflight describes all failures
        status["error"] = str(exc)
        return status
    status["ready"] = True
    return status


def _install_piper_voices(
    voice_dir: Path,
    hf_hub_download: Callable[..., str],
) -> dict[str, Any]:
    voice_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for voice in PIPER_VOICES:
        model_path = PIPER_VOICE_PATHS[voice]
        for remote_path in (model_path, f"{model_path}.json"):
            source = Path(
                hf_hub_download(
                    repo_id=PIPER_REPO,
                    revision=PIPER_REVISION,
                    filename=remote_path,
                )
            )
            if PIPER_REVISION not in source.parts or not source.is_file():
                raise RuntimeError(
                    f"Piper: {remote_path} получен не из pinned revision "
                    f"{PIPER_REVISION}: {source}"
                )
            suffix = ".onnx.json" if remote_path.endswith(".json") else ".onnx"
            destination = voice_dir / f"{voice}{suffix}"
            temporary = voice_dir / f".{destination.name}.tmp"
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            files[destination.name] = {
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
            }

    provenance = {
        "schema": 1,
        "repo_id": PIPER_REPO,
        "revision": PIPER_REVISION,
        "voices": list(PIPER_VOICES),
        "files": files,
    }
    manifest_path = voice_dir / _PIPER_PROVENANCE
    temporary_manifest = voice_dir / f".{_PIPER_PROVENANCE}.tmp"
    temporary_manifest.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    _validate_piper_install(voice_dir)
    return provenance


def _result(
    preset: LocalSetupPreset,
    model_statuses: dict[str, dict[str, Any]],
    piper_status: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    ready = all(item["ready"] for item in model_statuses.values()) and bool(
        piper_status["ready"]
    )
    stt_path = model_statuses.get(preset.stt_key, {}).get("path")
    translation_path = model_statuses.get(preset.translation_key, {}).get("path")
    return {
        "preset": preset.name,
        "dry_run": dry_run,
        "ready": ready,
        "models": model_statuses,
        "piper": piper_status,
        # Backwards-compatible convenience keys used by the existing CLI.
        "stt_path": stt_path,
        "translation_path": translation_path,
        "voice_dir": piper_status.get("voice_dir"),
    }


def preflight_mac_local(
    preset: str = "balanced",
    *,
    voice_dir: Path | None = None,
) -> dict[str, Any]:
    """Check a preset without network access or artifact mutation."""

    selected = _preset(preset)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'preflight требует huggingface-hub: pip install -e ".[mac-local]"'
        ) from exc

    target_voice_dir = (
        voice_dir or Path.home() / ".local/share/uvt/piper"
    ).expanduser()
    statuses = {
        key: _model_status(
            LOCAL_MODEL_MANIFEST[key], snapshot_download, local_only=True
        )
        for key in selected.model_keys
    }
    return _result(
        selected,
        statuses,
        _piper_status(target_voice_dir) if selected.requires_piper else {"ready": True, "required": False},
        dry_run=True,
    )


def setup_mac_local(
    preset: str = "balanced",
    *,
    dry_run: bool = False,
    voice_dir: Path | None = None,
) -> dict[str, Any]:
    """Install or verify one pinned local preset.

    ``dry_run=True`` is an alias for :func:`preflight_mac_local`: it never
    downloads files. A regular call explicitly downloads only the artifacts
    declared by the selected preset, validates the immutable snapshot paths and
    records checksums for the flattened Piper voice files.
    """

    if dry_run:
        return preflight_mac_local(preset, voice_dir=voice_dir)

    selected = _preset(preset)
    try:
        if selected.requires_piper:
            import piper  # noqa: F401 - ensure the selected profiles can start
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'локальная установка требует: pip install -e ".[mac-local]"'
        ) from exc

    log.info("устанавливаю local-%s только из pinned revisions", selected.name)
    statuses: dict[str, dict[str, Any]] = {}
    for key in selected.model_keys:
        spec = LOCAL_MODEL_MANIFEST[key]
        log.info("модель %s: %s@%s", key, spec.repo_id, spec.revision)
        status = _model_status(spec, snapshot_download, local_only=False)
        if not status["ready"]:
            raise RuntimeError(status["error"])
        statuses[key] = status

    target_voice_dir = (
        voice_dir or Path.home() / ".local/share/uvt/piper"
    ).expanduser()
    if selected.requires_piper:
        log.info("Piper: %s@%s -> %s", PIPER_REPO, PIPER_REVISION, target_voice_dir)
        _install_piper_voices(target_voice_dir, hf_hub_download)
    result = _result(
        selected,
        statuses,
        _piper_status(target_voice_dir) if selected.requires_piper else {"ready": True, "required": False},
        dry_run=False,
    )
    if not result["ready"]:  # defensive: install helpers should already raise
        raise RuntimeError("локальный preset не прошёл итоговый preflight")
    return result


def format_setup_report(result: dict[str, Any]) -> str:
    """Return a concise human-readable setup/preflight report."""

    mark = "✓" if result.get("ready") else "✗"
    mode = "preflight" if result.get("dry_run") else "setup"
    lines = [f"{mark} local-{result.get('preset')} ({mode})"]
    for status in result.get("models", {}).values():
        item_mark = "✓" if status.get("ready") else "✗"
        location = status.get("path") or status.get("error") or "не найдено"
        lines.append(
            f"  {item_mark} {status['role']}: {status['repo_id']}@"
            f"{status['revision'][:12]} -> {location}"
        )
    piper_status = result.get("piper", {})
    if piper_status.get("required") is False:
        lines.append("  ✓ MOSS: встроенные голоса Adam/Bella; Piper не требуется")
        return "\n".join(lines)
    piper_mark = "✓" if piper_status.get("ready") else "✗"
    piper_location = (
        piper_status.get("voice_dir")
        if piper_status.get("ready")
        else piper_status.get("error") or "не найдено"
    )
    lines.append(
        f"  {piper_mark} tts: {piper_status.get('repo_id')}@"
        f"{str(piper_status.get('revision', ''))[:12]} -> {piper_location}"
    )
    if result.get("dry_run") and not result.get("ready"):
        lines.append("  Ничего не скачано; запустите explicit setup для этого preset.")
    return "\n".join(lines)
