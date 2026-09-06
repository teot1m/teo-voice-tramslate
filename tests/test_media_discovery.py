import asyncio
from pathlib import Path

import pytest
import httpx

from uvt.media_discovery import extract_media_urls, public_http_url


def test_playerjs_quality_options_choose_small_media_without_scraping_ads():
    document = '''<a href="https://cdn.test/ad.mp4">ad</a><script>
      const tracker = "https://cdn.test/tracker.mp4";
      var player = new Playerjs({id:"videoplayer",file:"[720p] https://cdn.test/film_720p.mp4,[240p] https://cdn.test/film_240p.mp4,[480p] https://cdn.test/film_480p.mp4"});
    </script>'''
    assert extract_media_urls(document, "https://site.test/watch") == [
        "https://cdn.test/film_240p.mp4", "https://cdn.test/film_480p.mp4", "https://cdn.test/film_720p.mp4",
    ]


def test_relative_sources_escaped_player_urls_and_signed_queries():
    document = r'''<video><source src="/master.m3u8?a=1&amp;b=2"></video>
    <script>new Playerjs({file:'[360p] https:\/\/cdn.test\/movie_360p.mp4?token=a,b\u0026expires=1'});</script>'''
    assert extract_media_urls(document, "https://site.test/watch") == [
        "https://site.test/master.m3u8?a=1&b=2", "https://cdn.test/movie_360p.mp4?token=a,b&expires=1",
    ]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/x.mp4", "http://[::1]/x", "http://localhost/x", "http://2130706433/x", "http://user:pass@cdn.test/x", "http://cdn.test/x\r\nY: 1"])
def test_non_public_player_urls_are_rejected(url):
    assert public_http_url(url, "https://site.test/watch") is None


@pytest.mark.asyncio
async def test_unsupported_ytdlp_uses_declared_media_with_referer(monkeypatch, tmp_path):
    import uvt.server as server
    import uvt.dub as dub
    import uvt.media_discovery as discovery

    page = "https://site.test/watch"
    async def fail_ytdlp(cmd, timeout_s, what, on_line=None):
        on_line("ERROR: Unsupported URL: " + page)
        raise RuntimeError("yt-dlp error")
    async def discover(url):
        assert url == page
        return ["https://cdn.test/film_240p.mp4"]
    async def download(url, dest_dir, referer=None, out_name=None, **kwargs):
        assert url == "https://cdn.test/film_240p.mp4"
        assert referer == page
        path = dest_dir / out_name
        path.write_bytes(b"audio")
        return path
    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(server, "_run_process", fail_ytdlp)
    monkeypatch.setattr(discovery, "discover_page_media", discover)
    monkeypatch.setattr(server, "_download_media", download)
    path = await server._download_page(page, tmp_path)
    assert path.read_bytes() == b"audio"


@pytest.mark.asyncio
async def test_access_denied_does_not_trigger_html_fallback(monkeypatch, tmp_path):
    import uvt.server as server
    import uvt.dub as dub
    import uvt.media_discovery as discovery
    async def fail(cmd, timeout_s, what, on_line=None):
        on_line("ERROR: HTTP Error 403: Forbidden")
        raise RuntimeError("yt-dlp error")
    async def never(url):
        raise AssertionError("No additional page fetching on access denial")
    monkeypatch.setattr(dub, "_find_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(server, "_run_process", fail)
    monkeypatch.setattr(discovery, "discover_page_media", never)
    with pytest.raises(RuntimeError, match="HTTP Error 403"):
        await server._download_page("https://site.test/private", tmp_path)


@pytest.mark.parametrize("url", [
    "http://[broken", "http://cdn.test:65536/a.mp4", "http://0x7f.0.0.1/a.mp4",
    "http://0177.0.0.1/a.mp4", "http://router/a.mp4", "http://machine.internal/a.mp4",
    "http://%31%32%37.0.0.1/a.mp4", "http://cdn.test/a\t.mp4", "\nhttps://cdn.test/a.mp4",
    "http://cdn.test\\@127.0.0.1/a.mp4", "http://@cdn.test/a.mp4",
])
def test_malformed_and_ambiguous_urls_do_not_escape_validation(url):
    assert public_http_url(url, "https://site.test/watch") is None


def test_only_static_top_level_player_file_is_used():
    document = '''<script>
      // new Playerjs({file:"https://cdn.test/comment.mp4"});
      /* new Playerjs({file:"https://cdn.test/block.mp4"}); */
      const help = 'new Playerjs({file:"https://cdn.test/string.mp4"})';
      new Playerjs({advert:{file:"https://cdn.test/ad.mp4"},
        title:'description says file:"https://cdn.test/title.mp4"',
        file:"https://cdn.test/movie.mp4", subtitles:{file:"https://cdn.test/sub.vtt"}});
      new Playerjs({file:"https://cdn.test/prefix.mp4" + token});
      new Playerjs({file:"https://cdn.test/old.mp4", file:currentFile});
    </script>
    <script type="application/ld+json">new Playerjs({file:"https://cdn.test/json.mp4"})</script>
    <script src="/script.js">new Playerjs({file:"https://cdn.test/ignored.mp4"})</script>
    <source src="https://cdn.test/orphan.mp4"><a href="https://cdn.test/link.mp4">movie</a>'''
    assert extract_media_urls(document, "https://site.test/watch") == ["https://cdn.test/movie.mp4"]


def test_quality_labels_and_signed_query_commas():
    document = '''<script>new Playerjs({"file":"[720p]https://cdn.test/high?sig=x,https://signature.test/value,[240p]https://cdn.test/low"});</script>'''
    assert extract_media_urls(document, "https://site.test/watch") == [
        "https://cdn.test/low", "https://cdn.test/high?sig=x,https://signature.test/value",
    ]


def test_candidate_deduplication_and_input_limits(monkeypatch):
    import uvt.media_discovery as discovery
    document = "".join(f'<video src="https://cdn.test/{number}.mp4"></video>' for number in range(12))
    document += '<video src="https://cdn.test/0.mp4"></video>'
    assert extract_media_urls(document, "https://site.test/watch") == [f"https://cdn.test/{number}.mp4" for number in range(6)]
    monkeypatch.setattr(discovery, "MAX_HTML_BYTES", 32)
    assert extract_media_urls(document, "https://site.test/watch") == []


def test_player_object_limit_fails_closed(monkeypatch):
    import uvt.media_discovery as discovery
    monkeypatch.setattr(discovery, "MAX_PLAYER_CHARS", 32)
    document = '<script>new Playerjs({padding:"' + "x" * 100 + '",file:"https://cdn.test/movie.mp4"});</script>'
    assert extract_media_urls(document, "https://site.test/watch") == []


def _mock_http(monkeypatch, handler):
    import uvt.media_discovery as discovery
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(discovery.httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))


@pytest.mark.asyncio
async def test_discovery_uses_redirect_base_url(monkeypatch):
    import uvt.media_discovery as discovery
    requests = []
    def handler(request):
        requests.append(str(request.url))
        if request.url.path == "/watch":
            return httpx.Response(302, headers={"Location": "/folder/player"})
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text='<video src="movie.mp4"></video>')
    _mock_http(monkeypatch, handler)
    assert await discovery.discover_page_media("https://site.test/watch") == ["https://site.test/folder/movie.mp4"]
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("location,expected_requests", [
    ("http://127.0.0.1/admin", 1), ("http://[broken", 1), ("", 1), ("/watch", 4),
])
async def test_discovery_bounds_redirects(monkeypatch, location, expected_requests):
    import uvt.media_discovery as discovery
    requests = []
    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": location})
    _mock_http(monkeypatch, handler)
    assert await discovery.discover_page_media("https://site.test/watch") == []
    assert len(requests) == expected_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {"Content-Type": "application/not-html"},
    {"Content-Type": "text/html"},
    {"Content-Type": "text/html", "Content-Length": "garbled"},
])
async def test_discovery_rejects_non_html_or_oversized_content(monkeypatch, headers):
    import uvt.media_discovery as discovery
    monkeypatch.setattr(discovery, "MAX_HTML_BYTES", 40)
    _mock_http(monkeypatch, lambda request: httpx.Response(200, headers=headers, content=b'<video src="https://cdn.test/movie.mp4"></video>'))
    assert await discovery.discover_page_media("https://site.test/watch") == []


@pytest.mark.asyncio
async def test_discovery_bounds_chunked_body_without_length(monkeypatch):
    import uvt.media_discovery as discovery
    monkeypatch.setattr(discovery, "MAX_HTML_BYTES", 40)
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(5):
                yield b"0123456789"
    _mock_http(monkeypatch, lambda request: httpx.Response(200, headers={"Content-Type": "text/html"}, stream=Body()))
    assert await discovery.discover_page_media("https://site.test/watch") == []


@pytest.mark.asyncio
async def test_discovery_total_deadline_closes_stream(monkeypatch):
    import uvt.media_discovery as discovery
    closed = asyncio.Event()
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b"never"
        async def aclose(self):
            closed.set()
    _mock_http(monkeypatch, lambda request: httpx.Response(200, headers={"Content-Type": "text/html"}, stream=Body()))
    monkeypatch.setattr(discovery, "DISCOVERY_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(httpx.ReadTimeout, match="time limit"):
        await discovery.discover_page_media("https://site.test/watch")
    assert closed.is_set()
