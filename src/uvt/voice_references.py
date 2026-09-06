"""Private user-provided voice samples for reference-based local TTS."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

REFERENCE_ID = re.compile(r"ref_[a-f0-9]{32}")


def library_dir() -> Path:
    return Path(os.environ.get("UVT_VOICE_LIBRARY_DIR", "~/.local/share/uvt/voice-references")).expanduser().resolve()


def resolve_reference(voice_id: str) -> tuple[Path, str]:
    if not REFERENCE_ID.fullmatch(voice_id):
        raise ValueError("Неизвестный образец голоса")
    root = library_dir()
    metadata = root / (voice_id + ".json")
    path = root / (voice_id + ".wav")
    if not metadata.is_file() or metadata.stat().st_size > 16000 or not path.is_file():
        raise ValueError("Образец голоса удалён или недоступен")
    if metadata.resolve().parent != root or path.resolve().parent != root:
        raise ValueError("Образец голоса находится вне библиотеки")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("id") != voice_id or not isinstance(data.get("text"), str) or not data["text"].strip():
        raise ValueError("Повреждены данные образца голоса")
    return path, data["text"]


def catalog() -> list[dict]:
    root = library_dir()
    if not root.is_dir():
        return []
    result = []
    for metadata in sorted(root.glob("ref_*.json"))[:100]:
        try:
            resolve_reference(metadata.stem)
            data = json.loads(metadata.read_text(encoding="utf-8"))
            result.append({"id": metadata.stem, "label": str(data["label"])[:100],
                           "gender": data.get("gender", "auto"),
                           "language": data.get("language", "ru"),
                           "languages": data.get("languages", [data.get("language", "ru")]),
                           "engine": "f5", "installed": True, "reference": True})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result
