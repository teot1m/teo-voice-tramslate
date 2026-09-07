"""Public Bunkr player metadata is parsed and signed without executing JavaScript."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import uvt.media_discovery as discovery


PAGE = "https://bunkr.cr/f/neutralRecording"
MEDIA = "https://c3pz-b.cdn.cr/storage/media/Neutral%20clip.mp4?download=1&lang=en&lang=uk&token=old&ex=1"
SIGN = "https://glb-apisign.cdn.cr/sign"
SIGNED = {"token": "neutral-test-token_123", "ex": 2000000000}


def declarations(media=MEDIA, sign=SIGN):
    # Actual player HTML escapes slashes inside JSON-like string literals.
    js = lambda value: json.dumps(value).replace("/", "\\/")
    return (f"var jsCDN = {js(media)}; var jsType = {js('video/mp4')}; "
            f"var signUrl = {js(sign)};")


def document(script=None):
    return f"<html><body><script>{declarations() if script is None else script}</script></body></html>"


def mock_http(monkeypatch, html=None, signer=None):
    original_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "glb-apisign.cdn.cr":
            assert request.url.path == "/sign"
            return signer(request) if signer else httpx.Response(200, json=SIGNED)
        assert len(requests) == 1, "Only the public page and pinned signing endpoint may be requested"
        return httpx.Response(200, text=html if html is not None else document(),
                              headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(discovery.httpx, "AsyncClient", lambda **kwargs:
                        original_client(transport=transport, **kwargs))
    return requests


@pytest.mark.parametrize("page", [PAGE, "https://www.bunkr.cr/f/neutralRecording"])
async def test_public_bunkr_signing_decodes_path_and_preserves_media_query(monkeypatch, page):
    requests = mock_http(monkeypatch)
    result = await discovery.discover_page_media(page)
    assert len(result) == 1
    result_url = urlsplit(result[0])
    assert result_url.scheme == "https"
    assert result_url.netloc == "c3pz-b.cdn.cr"
    assert result_url.path == "/storage/media/Neutral%20clip.mp4"
    assert parse_qs(result_url.query) == {
        "download": ["1"], "lang": ["en", "uk"],
        "token": [SIGNED["token"]], "ex": [str(SIGNED["ex"])],
    }
    assert len(requests) == 2
    signed_request = requests[1]
    assert signed_request.method == "GET"
    assert signed_request.url.copy_with(query=None) == httpx.URL(SIGN)
    assert dict(signed_request.url.params) == {"path": "/storage/media/Neutral clip.mp4"}
    assert "authorization" not in signed_request.headers
    assert "cookie" not in signed_request.headers
    assert requests[0].url == httpx.URL(page)


@pytest.mark.parametrize("status", [401, 403])
async def test_signer_access_denial_is_not_retried_or_bypassed(monkeypatch, status):
    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError) as error:
        await discovery.discover_page_media(PAGE)
    assert error.value.response.status_code == status
    assert len(requests) == 2


@pytest.mark.parametrize("location", ["https://other.test/sign", "http://127.0.0.1/private"])
async def test_signer_redirect_is_never_followed(monkeypatch, location):
    requests = mock_http(monkeypatch, signer=lambda request:
                         httpx.Response(302, headers={"location": location}))
    with pytest.raises(httpx.HTTPStatusError) as error:
        await discovery.discover_page_media(PAGE)
    assert error.value.response.status_code == 302
    assert len(requests) == 2


@pytest.mark.parametrize("payload", [
    None, [], {"token": "ok"}, {"ex": 2000000000},
    {"token": 123, "ex": 2000000000}, {"token": "", "ex": 2000000000},
    {"token": "line\nbreak", "ex": 2000000000},
    {"token": "ok", "ex": {}}, {"token": "ok", "ex": True},
    {"token": "ok", "ex": float("inf")}, {"token": "ok", "ex": -1},
])
async def test_invalid_signer_fields_fail_closed(monkeypatch, payload):
    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(
        200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"}))
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 2


async def test_malformed_json_signature_fails_closed(monkeypatch):
    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(
        200, content=b'{"token": broken', headers={"content-type": "application/json"}))
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 2


@pytest.mark.parametrize("page", [
    "https://other.test/f/neutralRecording",
    "https://bunkr.cr.other.test/f/neutralRecording",
    "https://notbunkr.cr/f/neutralRecording",
    "https://bunkr.cr/a/neutral-album",
])
async def test_unrelated_page_host_or_path_never_requests_signature(monkeypatch, page):
    requests = mock_http(monkeypatch)
    assert await discovery.discover_page_media(page) == []
    assert len(requests) == 1


@pytest.mark.parametrize("script", [
    "// " + declarations(),
    "/* " + declarations() + " */",
    "var help = " + json.dumps(declarations()) + ";",
    "function example() { " + declarations() + " }",
    "if (false) { " + declarations() + " }",
    "var jsCDN = " + json.dumps(MEDIA) + " + suffix; var jsType='video/mp4'; var signUrl=" + json.dumps(SIGN) + ";",
])
async def test_comments_strings_nested_or_dynamic_assignments_are_not_executed(monkeypatch, script):
    requests = mock_http(monkeypatch, html=document(script))
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 1


@pytest.mark.parametrize("attrs", ['src="/player.js"', 'type="application/ld+json"'])
async def test_non_inline_javascript_is_not_interpreted(monkeypatch, attrs):
    requests = mock_http(monkeypatch, html=f"<script {attrs}>{declarations()}</script>")
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 1


@pytest.mark.parametrize("media,sign", [
    ("http://127.0.0.1/storage/media/neutral.mp4", SIGN),
    ("https://cdn.cr.other.test/storage/media/neutral.mp4", SIGN),
    ("https://c3pz-b.cdn.cr/other/neutral.mp4", SIGN),
    (MEDIA, "http://127.0.0.1/sign"),
    (MEDIA, "https://other.test/sign"),
    (MEDIA, SIGN + "?path=attacker"),
])
async def test_media_and_signer_origins_are_restricted(monkeypatch, media, sign):
    requests = mock_http(monkeypatch, html=document(declarations(media, sign)))
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 1


class WaitingBody(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.sleep(10)
        yield b'{}'

    async def aclose(self):
        self.closed = True


async def test_cancelling_signature_fetch_closes_body_and_propagates(monkeypatch):
    body = WaitingBody()
    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(
        200, stream=body, headers={"content-type": "application/json"}))
    task = asyncio.create_task(discovery.discover_page_media(PAGE))
    try:
        await asyncio.wait_for(body.started.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert body.closed
    assert len(requests) == 2


async def test_signature_fetch_shares_total_discovery_deadline(monkeypatch):
    body = WaitingBody()
    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(
        200, stream=body, headers={"content-type": "application/json"}))
    monkeypatch.setattr(discovery, "DISCOVERY_TIMEOUT_SECONDS", 0.03)
    with pytest.raises(httpx.ReadTimeout):
        await asyncio.wait_for(discovery.discover_page_media(PAGE), timeout=1)
    assert body.started.is_set()
    assert body.closed
    assert len(requests) == 2


async def test_decimal_string_expiry_is_supported(monkeypatch):
    mock_http(monkeypatch, signer=lambda request: httpx.Response(
        200, json={"token": "neutral", "ex": "2000000000"}))
    result = await discovery.discover_page_media(PAGE)
    assert parse_qs(urlsplit(result[0]).query)["ex"] == ["2000000000"]


@pytest.mark.parametrize("advertise_length", [True, False])
async def test_oversized_signature_reply_is_bounded_and_closed(monkeypatch, advertise_length):
    class OversizedBody(httpx.AsyncByteStream):
        closed = False
        chunks = 0

        async def __aiter__(self):
            for _ in range(20):
                self.chunks += 1
                yield b"x" * 4096

        async def aclose(self):
            self.closed = True

    body = OversizedBody()
    headers = {"content-type": "application/json"}
    if advertise_length:
        headers["content-length"] = str(discovery.MAX_SIGN_BYTES + 1)
    requests = mock_http(monkeypatch, signer=lambda request:
                         httpx.Response(200, stream=body, headers=headers))
    assert await discovery.discover_page_media(PAGE) == []
    assert body.closed
    assert body.chunks < 20
    if advertise_length:
        assert body.chunks == 0
    assert len(requests) == 2


async def test_unsupported_page_download_uses_bunkr_signed_candidate(monkeypatch, tmp_path):
    import uvt.dub as dub
    import uvt.server as server

    requests = mock_http(monkeypatch)
    downloaded = []

    async def fail_ytdlp(cmd, timeout_s, what, on_line=None):
        if on_line:
            on_line("ERROR: Unsupported URL: " + PAGE)
        raise RuntimeError("yt-dlp does not support this public page")

    async def download_media(url, dest_dir, referer=None, out_name=None, **kwargs):
        downloaded.append((url, referer))
        path = dest_dir / (out_name or "neutral.m4a")
        path.write_bytes(b"neutral test audio")
        return path

    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "mock-yt-dlp")
    monkeypatch.setattr(server, "_run_process", fail_ytdlp)
    monkeypatch.setattr(server, "_download_media", download_media)
    result = await server._download_page(PAGE, tmp_path)
    assert result.read_bytes() == b"neutral test audio"
    assert len(requests) == 2
    assert len(downloaded) == 1
    media_url, referer = downloaded[0]
    assert referer == PAGE
    assert parse_qs(urlsplit(media_url).query)["token"] == [SIGNED["token"]]
    assert urlsplit(media_url).path == "/storage/media/Neutral%20clip.mp4"


async def test_signer_denial_reaches_download_error_without_exposing_signer_details(
    monkeypatch, tmp_path, caplog,
):
    import logging
    import uvt.dub as dub
    import uvt.server as server

    requests = mock_http(monkeypatch, signer=lambda request: httpx.Response(403))
    downloaded = []
    caplog.set_level(logging.INFO, logger="uvt.server")

    async def fail_ytdlp(cmd, timeout_s, what, on_line=None):
        if on_line:
            on_line("ERROR: Unsupported URL: " + PAGE)
        raise RuntimeError("yt-dlp does not support this public page")

    async def never_download(*args, **kwargs):
        downloaded.append(args)
        raise AssertionError("Access denial must not be bypassed")

    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "mock-yt-dlp")
    monkeypatch.setattr(server, "_run_process", fail_ytdlp)
    monkeypatch.setattr(server, "_download_media", never_download)
    with pytest.raises(RuntimeError) as error:
        await server._download_page(PAGE, tmp_path)
    assert "Публичный плеер не выдал поток (HTTP 403)" in str(error.value)
    assert len(requests) == 2
    assert downloaded == []
    for output in (str(error.value), caplog.text):
        assert SIGN not in output
        assert "token=" not in output


async def test_signer_receives_no_cookie_collected_during_page_redirects(monkeypatch):
    original_client = httpx.AsyncClient
    requests = []
    final_page = "https://bunkr.cr/f/neutralReady"

    def handler(request):
        requests.append(request)
        if str(request.url) == PAGE:
            return httpx.Response(302, headers={"location": "https://glb-apisign.cdn.cr/bootstrap"})
        if request.url.path == "/bootstrap":
            return httpx.Response(302, headers={"location": final_page,
                "set-cookie": "session=neutral-cookie; Path=/; Secure"})
        if str(request.url) == final_page:
            return httpx.Response(200, text=document(), headers={"content-type": "text/html"})
        assert request.url.copy_with(query=None) == httpx.URL(SIGN)
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers
        return httpx.Response(200, json=SIGNED)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(discovery.httpx, "AsyncClient", lambda **kwargs:
                        original_client(transport=transport, **kwargs))
    assert len(await discovery.discover_page_media(PAGE)) == 1
    assert len(requests) == 4


@pytest.mark.parametrize("html", [
    document("const decoy = /;" + declarations() + "/;"),
    document(declarations() + "jsCDN = runtimeValue;"),
    '<script>var jsCDN=' + json.dumps(MEDIA) + ';</script>'
        '<script>var jsType="video/mp4";var signUrl=' + json.dumps(SIGN) + ';</script>',
])
async def test_only_one_dedicated_literal_config_block_can_request_signature(monkeypatch, html):
    requests = mock_http(monkeypatch, html=html)
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 1


async def test_dedicated_config_allows_comments_and_unrelated_string_fields(monkeypatch):
    script = ('// Public player configuration only\n'
              'var videoCoverUrl="https://c3pz-b.cdn.cr/storage/images/neutral.jpg";'
              '/* Audio/video type and URL */' + declarations() +
              'const jsSlug="neutralRecording";')
    requests = mock_http(monkeypatch, html=document(script))
    result = await discovery.discover_page_media(PAGE)
    assert len(result) == 1
    assert len(requests) == 2
    assert urlsplit(result[0]).path == "/storage/media/Neutral%20clip.mp4"


@pytest.mark.parametrize("suffix", ['"unterminated', "/* unfinished comment"])
async def test_incomplete_script_after_config_does_not_produce_signature(monkeypatch, suffix):
    requests = mock_http(monkeypatch, html=document(declarations() + suffix))
    assert await discovery.discover_page_media(PAGE) == []
    assert len(requests) == 1
