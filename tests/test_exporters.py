"""Экспорт истории: srt / vtt / txt / json (ТЗ §12)."""
import json

from uvt.history import HistoryEntry, to_json, to_srt, to_txt, to_vtt

ENTRIES = [
    HistoryEntry(start=0.0, end=1.5, original="hello", translated="привет",
                 language="en", target_lang="ru"),
    HistoryEntry(start=62.0, end=63.25, original="world", translated="мир",
                 language="en", target_lang="ru"),
]


def test_srt_format():
    srt = to_srt(ENTRIES)
    assert "1\n00:00:00,000 --> 00:00:01,500\nпривет" in srt
    assert "2\n00:01:02,000 --> 00:01:03,250\nмир" in srt


def test_srt_both_lines():
    srt = to_srt(ENTRIES, which="both")
    assert "hello\nпривет" in srt


def test_vtt_format():
    vtt = to_vtt(ENTRIES)
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_txt_format():
    txt = to_txt(ENTRIES)
    assert "[00:00:00,000] hello" in txt
    assert "→ привет" in txt


def test_json_roundtrip():
    data = json.loads(to_json(ENTRIES))
    assert data[0]["original"] == "hello"
    assert data[1]["end"] == 63.25
