"""Read explicit HTML5 / Playerjs media without executing page JavaScript.

URL validation rejects private address literals; it is not a DNS/SSRF sandbox.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import re
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_MEDIA_CANDIDATES = 6
MAX_PLAYER_CHARS = 32768
DISCOVERY_TIMEOUT_SECONDS = 20
_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[/\\\"'])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LEGACY_IP = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*", re.I)


def _unescape_js(value: str) -> str:
    def replace(match):
        token = match.group(1)
        return chr(int(token[1:], 16)) if token.startswith(("u", "x")) else token
    return html.unescape(_ESCAPE.sub(replace, value))


def public_http_url(value: str, base: str) -> str | None:
    # urlsplit/urljoin discard some controls, so inspect the original first.
    if _CONTROL.search(value) or _CONTROL.search(base):
        return None
    try:
        value = urljoin(base, value.strip())
        value.encode("utf-8")
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
            return None
        if "\\" in parsed.netloc or "%" in host or host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            if _LEGACY_IP.fullmatch(host) or "." not in host:
                return None
            ascii_host = host.encode("idna").decode("ascii")
            if any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label, re.I) for label in ascii_host.split(".")):
                return None
        _ = parsed.port
    except (ValueError, UnicodeError):
        return None
    return value


class _MediaParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.scripts: list[str] = []
        self.in_script = False
        self.in_media = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"video", "audio"}:
            self.in_media += 1
            if attrs.get("src"):
                self.urls.append(attrs["src"])
        if tag == "source" and self.in_media and attrs.get("src"):
            self.urls.append(attrs["src"])
        if tag == "script":
            script_type = (attrs.get("type") or "").split(";", 1)[0].strip().lower()
            self.in_script = not attrs.get("src") and script_type in {"", "module", "text/javascript", "application/javascript"}

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        if tag in {"video", "audio"}:
            self.in_media = max(0, self.in_media - 1)

    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)


def _js_tokens(script: str):
    """Distinguish literals, comments and nesting; never evaluate expressions."""
    index = 0
    while index < len(script):
        start = index
        char = script[index]
        if char.isspace():
            index += 1
            continue
        if script.startswith("//", index):
            end = script.find("\n", index + 2)
            index = len(script) if end < 0 else end + 1
            continue
        if script.startswith("/*", index):
            end = script.find("*/", index + 2)
            index = len(script) if end < 0 else end + 2
            continue
        if char in "\"'`":
            index += 1
            while index < len(script):
                if script[index] == "\\":
                    index += 2
                elif script[index] == char:
                    break
                else:
                    index += 1
            if index >= len(script):
                return
            kind = "template" if char == "`" else "string"
            yield kind, _unescape_js(script[start + 1:index]), start
            index += 1
        elif char.isalpha() or char in "_$":
            index += 1
            while index < len(script) and (script[index].isalnum() or script[index] in "_$"):
                index += 1
            yield "identifier", script[start:index], start
        else:
            index += 1
            yield "punctuation", char, start


def _player_file(tokens, limit: int) -> str | None:
    """Accept only a literal top-level file property, never nested ad fields."""
    stack = ["{"]
    fields = []
    value = None

    def finish_field():
        nonlocal value
        if len(fields) >= 2 and fields[0][0] in {"identifier", "string"} and fields[0][1] == "file" and fields[1][1] == ":":
            value = fields[2][1] if len(fields) == 3 and fields[2][0] == "string" else None
        fields.clear()

    for token in tokens:
        kind, text, position = token
        if position >= limit:
            return None
        if kind == "punctuation":
            if text == "," and len(stack) == 1:
                finish_field()
                continue
            if text in "})]":
                if not stack or stack.pop() != {"}": "{", ")": "(", "]": "["}[text]:
                    return None
                if not stack:
                    finish_field()
                    return value
            elif text in "{([":
                stack.append(text)
        if len(fields) < 4:
            fields.append(token)
    return None


def _player_files(script: str):
    tokens = iter(_js_tokens(script))
    window = deque(maxlen=4)
    pattern = [("identifier", "new"), ("identifier", "Playerjs"), ("punctuation", "("), ("punctuation", "{")]
    for token in tokens:
        window.append(token)
        if [(kind, text) for kind, text, _ in window] == pattern:
            value = _player_file(tokens, token[2] + MAX_PLAYER_CHARS)
            if value is not None:
                yield value
            window.clear()


def extract_media_urls(document: str, page_url: str) -> list[str]:
    if len(document) > MAX_HTML_BYTES:
        return []
    parser = _MediaParser()
    parser.feed(document)
    candidates = [(url, None) for url in parser.urls]
    for script in parser.scripts:
        for value in _player_files(script):
            # Only a quality label separates alternatives. Signed query commas
            # must survive, even if a query contains another complete URL.
            for variant in re.split(r",\s*(?=\[[^\]]{1,40}\])", value):
                label = re.match(r"^\s*\[([^\]]{1,40})\]\s*", variant)
                quality = re.fullmatch(r"(\d{2,4})p?", label[1], re.I) if label else None
                raw = variant[label.end():] if label else variant
                candidates.append((raw, int(quality[1]) if quality else None))
    unique = {}
    for raw, label_quality in candidates:
        url = public_http_url(raw, page_url)
        if url and url not in unique:
            lower = url.lower()
            kind = 0 if re.search(r"\.(m3u8|mpd)(?:[?#]|$)", lower) else 1
            quality = re.search(r"(?:[/_\-])(240|360|480|720|1080|1440|2160)p?(?:[._/?#-]|$)", lower)
            unique[url] = (kind, label_quality if label_quality is not None else int(quality[1]) if quality else 9999)
    return sorted(unique, key=unique.__getitem__)[:MAX_MEDIA_CANDIDATES]


async def _fetch_page_media(page_url: str) -> list[str]:
    url = public_http_url(page_url, page_url)
    if not url:
        return []
    async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
        for _ in range(4):
            async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html", "Accept-Encoding": "identity"}) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    url = public_http_url(location, url) if location else None
                    if not url:
                        return []
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    return []
                try:
                    if int(response.headers.get("content-length", "0")) > MAX_HTML_BYTES:
                        return []
                except ValueError:
                    return []
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > MAX_HTML_BYTES:
                        return []
                    body.extend(chunk)
                return extract_media_urls(body.decode("utf-8", errors="replace"), str(response.url))
    return []


async def discover_page_media(page_url: str) -> list[str]:
    try:
        return await asyncio.wait_for(_fetch_page_media(page_url), timeout=DISCOVERY_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise httpx.ReadTimeout("HTML media discovery exceeded its time limit") from exc
