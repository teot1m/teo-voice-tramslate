"""Бесплатный переводчик google-free: разбор ответа и пакетный режим."""
import httpx
import pytest

from uvt.config import TranslationConfig
from uvt.engines.translate_google import GoogleFreeTranslator


def _handler(request: httpx.Request) -> httpx.Response:
    query = request.url.params["q"]
    # эмулируем gtx: перевод = "X"+строка, переносы сохраняются
    translated = "\n".join("X" + line for line in query.split("\n"))
    return httpx.Response(200, json=[[[translated, query, None]], None, "en"])


@pytest.fixture
async def translator():
    engine = GoogleFreeTranslator(TranslationConfig(engine="google-free"))
    await engine.warmup()
    await engine._client.aclose()
    engine._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    yield engine
    await engine.close()


async def test_single_translate(translator):
    assert await translator.translate("hello", "en", "ru", []) == "Xhello"


async def test_batch_preserves_lines(translator):
    out = await translator.translate_batch(["hello", "big world"], "en", "ru")
    assert out == ["Xhello", "Xbig world"]


async def test_batch_falls_back_on_mismatch(translator):
    # хендлер, склеивающий строки — пачка "распадается", ждём пореплечный путь
    def bad_handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["q"]
        if "\n" in query:
            return httpx.Response(200, json=[[["merged into one", query, None]], None, "en"])
        return _handler(request)

    await translator._client.aclose()
    translator._client = httpx.AsyncClient(transport=httpx.MockTransport(bad_handler))
    out = await translator.translate_batch(["a", "b"], "en", "ru")
    assert out == ["Xa", "Xb"]
