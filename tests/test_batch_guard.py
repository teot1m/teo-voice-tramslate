"""Детектор сбитой нумерации пакетного перевода (слабые локальные модели)."""
from uvt.engines.translate_openai import _misaligned


TEXTS = [
    "Первая фраза оригинала.",
    "Вторая фраза оригинала.",
    "Третья фраза оригинала.",
    "Четвёртая фраза оригинала.",
]


def test_shifted_by_one_is_caught():
    # классический сбой: «перевод» строки i — это исходная строка i+1
    parsed = {1: TEXTS[1], 2: TEXTS[2], 3: TEXTS[3], 4: "нормальный перевод"}
    assert _misaligned(TEXTS, parsed)


def test_normal_translation_passes():
    parsed = {i: f"translation {i}" for i in range(1, 5)}
    assert not _misaligned(TEXTS, parsed)


def test_passthrough_same_position_is_ok():
    # модель оставила строку без перевода НА СВОЁМ месте — это не сдвиг
    parsed = {1: TEXTS[0], 2: TEXTS[1], 3: "translation", 4: "translation"}
    assert not _misaligned(TEXTS, parsed)
