"""Нарезка пословных таймкодов Whisper на предложения (синхронность дубляжа)."""
from types import SimpleNamespace

from uvt.engines.stt_faster_whisper import words_to_spans


def _word(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end)


def test_split_by_punctuation():
    words = [
        _word(" Привет", 0.0, 0.4), _word(" мир.", 0.5, 0.9),
        _word(" Как", 2.0, 2.2), _word(" дела?", 2.3, 2.7),
        _word(" Отлично!", 5.0, 5.6),
    ]
    spans = words_to_spans(words, "ru")
    assert [s.text for s in spans] == ["Привет мир.", "Как дела?", "Отлично!"]
    assert spans[0].start == 0.0 and spans[0].end == 0.9
    assert spans[1].start == 2.0 and spans[1].end == 2.7
    assert all(s.language == "ru" for s in spans)


def test_split_long_run_without_punctuation():
    # 20 слов по секунде без знаков — режется предохранителем по длительности
    words = [_word(f" слово{i}", float(i), i + 0.9) for i in range(20)]
    spans = words_to_spans(words, "ru")
    assert len(spans) >= 2
    assert all(s.end - s.start <= 13.0 for s in spans)


def test_trailing_words_flushed():
    words = [_word(" Конец", 0.0, 0.5), _word(" без", 0.6, 0.8), _word(" точки", 0.9, 1.2)]
    spans = words_to_spans(words, None)
    assert len(spans) == 1
    assert spans[0].text == "Конец без точки"
