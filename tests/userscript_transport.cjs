/* Chrome fixtures use a GM API simulator, not a Tampermonkey installation.
 * UVT_LIVE_META=1 additionally bridges GET /meta to the already running server.
 * Every POST and every other path is mocked. No media/model jobs are started.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { chromium } = require(process.env.UVT_PLAYWRIGHT_MODULE || "playwright");
const script = process.argv[2];
const results = [];
const browserErrors = [];
const liveReads = [];
const fixtures = {
  api_version: 2, mode: "batch",
  capabilities: { profile_selection: true, voice_selection: true, tts_preview: true },
  profile: { name: "local-balanced", source_lang: "auto", target_lang: "ru",
    engines: { stt: "parakeet-mlx", translation: "translategemma-mlx", tts: "piper" } },
  defaults: { profile_id: "local-balanced", voice_gender: "auto", voice_id: null },
  profiles: [
    { id: "local-fast", label: "Быстро", installed: true,
      engines: { stt: "parakeet-mlx", translation: "nllb-ct2", tts: "piper" } },
    { id: "local-balanced", label: "Сбалансированный", installed: true,
      engines: { stt: "parakeet-mlx", translation: "translategemma-mlx", tts: "piper" } },
    { id: "local-quality", label: "Качество", installed: true,
      engines: { stt: "mlx-whisper", translation: "translategemma-mlx", tts: "piper" } },
    { id: "local-hymt", label: "Hy-MT2 · быстрый + Piper", installed: true,
      engines: { stt: "parakeet-mlx", translation: "hymt-mlx", tts: "piper" } },
  ],
  voices: [
    { id: "ru_RU-dmitri-medium", language: "ru", gender: "male", installed: true },
    { id: "ru_RU-irina-medium", language: "ru", gender: "female", installed: true },
  ],
  limits: { local_tts_languages: ["ru", "uk"] },
  model_readiness: { status: "ready", detail: "Проверочный сервер готов" },
  privacy: { data_leaves_device: false },
};

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  async function setup({ transport = "classic", behavior = "json", status = 200, live = false } = {}) {
    const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
    page.on("pageerror", error => browserErrors.push(error.message));
    await page.route("https://uvt-network.test/**", route => route.fulfill({
      status: 200, contentType: "text/html", headers: { "Content-Security-Policy": "connect-src 'none'" },
      body: '<!doctype html><html><head><title>UVT neutral network fixture</title></head><body><main><video id="main" controls style="width:640px;height:360px;background:#334155"></video></main></body></html>',
    }));
    await page.exposeFunction("__readOnlyMeta", async requestedUrl => {
      const url = new URL(requestedUrl);
      assert.equal(url.origin, "http://127.0.0.1:8765");
      assert.equal(url.pathname, "/meta");
      const response = await fetch(url, { signal: AbortSignal.timeout(20000), redirect: "error" });
      const body = await response.text();
      assert.equal(response.status, 200);
      liveReads.push(url.pathname + url.search);
      return body;
    });
    await page.goto("https://uvt-network.test/watch/neutral");
    assert.equal(await page.evaluate(async () => {
      try { await fetch("http://127.0.0.1:8765/meta"); return false; } catch (_) { return true; }
    }), true, "fixture must actually block page fetch through CSP");
    await page.evaluate(({ transport, behavior, status, live, fixtures }) => {
      window.__gmCalls = [];
      window.__gmAborts = 0;
      window.__nativeCalls = 0;
      window.__behavior = behavior;
      window.__status = status;
      const nativeFetch = window.fetch.bind(window);
      window.fetch = (...args) => { window.__nativeCalls += 1; return nativeFetch(...args); };
      const meta = url => {
        const copy = structuredClone(fixtures);
        const selected = new URL(url).searchParams.get("profile_id");
        if (selected) {
          const profile = copy.profiles.find(profile => profile.id === selected);
          if (profile) { copy.profile.name = profile.id; copy.profile.engines = profile.engines; }
        }
        return copy;
      };
      const send = details => {
        window.__gmCalls.push({ url: details.url, method: details.method, data: details.data,
          headers: details.headers, responseType: details.responseType, anonymous: details.anonymous,
          redirect: details.redirect, timeout: details.timeout });
        let resolvePromise, rejectPromise, timer, aborted = false;
        const promise = new Promise((resolve, reject) => { resolvePromise = resolve; rejectPromise = reject; });
        // Classic managers return a handle, so no caller consumes this promise.
        if (transport === "classic") promise.catch(() => {});
        const handle = transport === "classic" ? {} : promise;
        handle.abort = () => {
          window.__gmAborts += 1;
          aborted = true;
          clearTimeout(timer);
          details.onabort();
          rejectPromise(new DOMException("aborted", "AbortError"));
        };
        if (window.__behavior === "hang") return handle;
        timer = setTimeout(async () => {
          if (aborted) return;
          if (window.__behavior === "error") {
            if (transport !== "modern") details.onerror({ error: "fixture connection failure" });
            rejectPromise(new Error("fixture connection failure"));
            return;
          }
          try {
            const pathname = new URL(details.url).pathname;
            let body, type = "application/json";
            if (window.__behavior === "binary") {
              body = Uint8Array.from([82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69]).buffer;
              type = "audio/wav";
            } else if (window.__behavior === "http-error") {
              body = new TextEncoder().encode('Нет выбранной модели').buffer;
              type = "text/plain; charset=utf-8";
            } else {
              let text;
              if (pathname === "/meta") text = live ? await window.__readOnlyMeta(details.url) : JSON.stringify(meta(details.url));
              else if (pathname === "/dub") text = JSON.stringify({ id: "fixture-job", mode: "batch", status: "queued", stage: "queue", progress: 0 });
              else if (pathname === "/job/fixture-job") text = JSON.stringify({ status: "cancelled" });
              else text = JSON.stringify({ ok: true });
              body = new TextEncoder().encode(text).buffer;
            }
            if (aborted) return;
            const raw = { status: window.__status, response: body,
              responseHeaders: `Content-Type: ${type}\r\nX-Fixture: true\r\n` };
            if (transport !== "modern") details.onload(raw);
            resolvePromise(raw);
          } catch (error) {
            if (transport !== "modern") details.onerror(error);
            rejectPromise(error);
          }
        }, 0);
        return handle;
      };
      if (transport === "classic" || transport === "dual") window.GM_xmlhttpRequest = send;
      else if (transport === "modern") window.GM = { xmlHttpRequest: send };
    }, { transport, behavior, status, live, fixtures });
    // Fresh read sees concurrent production changes. Exports exist only in this
    // in-memory test copy; the distributed userscript remains an ordinary IIFE.
    const source = fs.readFileSync(script, "utf8");
    assert.match(source, /\}\)\(\);\s*$/);
    const instrumented = source.replace(/\}\)\(\);\s*$/, 'window.__uvtTest = { apiResponse, apiAudio, api, SCRIPT_VERSION };\n})();');
    await page.addScriptTag({ content: instrumented });
    await page.waitForFunction(() => !!window.__uvtTest && !!document.querySelector(".uvt-wrap"));
    return page;
  }
  async function scenario(name, options, run) {
    const page = await setup(options);
    try { await run(page); results.push(name); }
    finally { await page.close(); }
  }
  async function checkUi(page, live) {
    await page.getByRole("button", { name: /Настройки пакетного перевода:/ }).click();
    const profiles = page.getByLabel("Профиль:", { exact: true });
    await page.waitForFunction(() => {
      const select = [...document.querySelectorAll("select")].find(select => select.id.includes("local-profile"));
      return !!select && !select.disabled && [...select.options].some(option => option.value === "local-hymt");
    });
    assert.equal(await profiles.locator('option[value="local-hymt"]').count(), 1);
    await profiles.selectOption("local-fast");
    await page.waitForFunction(() => {
      const select = [...document.querySelectorAll("select")].find(select => select.id.includes("local-profile"));
      return select?.value === "local-fast" && window.__gmCalls.some(call => call.url.includes("profile_id=local-fast"));
    });
    assert.equal(await page.getByLabel("Использовать:", { exact: true }).inputValue(), "override");
    const version = await page.evaluate(() => window.__uvtTest.SCRIPT_VERSION);
    assert.match(await page.locator(".uvt-script-version").textContent(), new RegExp(version.replaceAll(".", "\\.")));
    // Close the panel first, then exercise the actual translation button.
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Подготовить пакетный перевод и синхронную аудиодорожку", exact: true }).click();
    await page.waitForFunction(() => window.__gmCalls.some(call => call.method === "POST" && new URL(call.url).pathname === "/dub"), null, { timeout: 15000 });
    const requests = await page.evaluate(() => window.__gmCalls.filter(call => call.method === "POST"));
    assert.equal(requests.length, 1, "single translation click must send one POST");
    const body = JSON.parse(requests[0].data);
    assert.equal(body.profile_id, "local-fast");
    assert.equal(body.settings_mode, "override");
    assert.equal(body.page_url, "https://uvt-network.test/watch/neutral");
    assert.equal(await page.evaluate(() => window.__nativeCalls), 0);
    if (live) assert.ok(liveReads.length >= 2);
  }
  try {
    await scenario("classic GM JSON bypasses blocked page fetch", {}, async page => {
      assert.equal(await page.evaluate(async () => (await window.__uvtTest.api("/meta")).capabilities.profile_selection), true);
      const calls = await page.evaluate(() => window.__gmCalls);
      assert.equal(calls.length, 1);
      assert.equal(calls[0].anonymous, true);
      assert.equal(calls[0].redirect, "error");
      assert.equal(calls[0].responseType, "arraybuffer");
      assert.equal(await page.evaluate(() => window.__nativeCalls), 0);
    });
    await scenario("modern promise-only GM JSON", { transport: "modern" }, async page => {
      assert.equal(await page.evaluate(async () => (await window.__uvtTest.api("/meta")).defaults.profile_id), "local-balanced");
      assert.equal(await page.evaluate(() => window.__nativeCalls), 0);
    });
    await scenario("manager using both callbacks and promise settles once", { transport: "dual" }, async page => {
      assert.deepEqual(await page.evaluate(async () => await window.__uvtTest.api("/fixture")), { ok: true });
      assert.equal(await page.evaluate(() => window.__gmCalls.length), 1);
    });
    for (const transport of ["classic", "modern"]) {
      await scenario(`${transport} GM preserves binary WAV bytes and MIME`, { transport, behavior: "binary" }, async page => {
        const result = await page.evaluate(async () => {
          const blob = await window.__uvtTest.apiAudio("/tts/preview", { method: "POST", body: JSON.stringify({ text: "Neutral test" }), headers: { "Content-Type": "application/json" } });
          return { type: blob.type, bytes: [...new Uint8Array(await blob.arrayBuffer())] };
        });
        assert.equal(result.type, "audio/wav");
        assert.deepEqual(result.bytes, [82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69]);
      });
    }
    await scenario("HTTP error retains server status and UTF-8 detail", { behavior: "http-error", status: 422 }, async page => {
      const message = await page.evaluate(async () => { try { await window.__uvtTest.api("/dub", { method: "POST" }); } catch (error) { return error.message; } });
      assert.match(message, /HTTP 422/);
      assert.match(message, /Нет выбранной модели/);
      assert.equal(await page.evaluate(() => window.__gmCalls.length), 1);
    });
    await scenario("abort before request never sends POST", { behavior: "hang" }, async page => {
      const name = await page.evaluate(async () => { const controller = new AbortController(); controller.abort(); try { await window.__uvtTest.api("/dub", { method: "POST", signal: controller.signal }); } catch (error) { return error.name; } });
      assert.equal(name, "AbortError");
      assert.equal(await page.evaluate(() => window.__gmCalls.length), 0);
    });
    for (const transport of ["classic", "modern"]) {
      await scenario(`${transport} abort during request stops transport`, { transport, behavior: "hang" }, async page => {
        const result = await page.evaluate(async () => {
          const controller = new AbortController();
          const pending = window.__uvtTest.api("/dub", { method: "POST", signal: controller.signal }).catch(error => error.name);
          controller.abort();
          return { name: await pending, aborts: window.__gmAborts, calls: window.__gmCalls.length };
        });
        assert.deepEqual(result, { name: "AbortError", aborts: 1, calls: 1 });
      });
    }
    await scenario("own timeout aborts a GM request whose native timeout is ignored", { behavior: "hang" }, async page => {
      const result = await page.evaluate(async () => {
        try { await window.__uvtTest.apiResponse("/dub", { method: "POST" }, undefined, 30); }
        catch (error) { return { message: error.message, code: error.code, aborts: window.__gmAborts, calls: window.__gmCalls.length, nativeCalls: window.__nativeCalls }; }
      });
      assert.match(result.message, /не ответил вовремя/);
      assert.equal(result.code, "UVT_CONNECTION");
      assert.equal(result.aborts, 1);
      assert.equal(result.calls, 1);
      assert.equal(result.nativeCalls, 0);
    });
    for (const transport of ["classic", "modern"]) {
      await scenario(`${transport} failed POST is not retried through native fetch`, { transport, behavior: "error" }, async page => {
        const result = await page.evaluate(async () => {
          try { await window.__uvtTest.api("/dub", { method: "POST", body: "{}" }); }
          catch (error) { return { message: error.message, calls: window.__gmCalls.length, nativeCalls: window.__nativeCalls }; }
        });
        assert.match(result.message, /Tampermonkey/);
        assert.equal(result.calls, 1);
        assert.equal(result.nativeCalls, 0);
      });
    }
    await scenario("missing GM API reports blocked native connection with installed-version guidance", { transport: "none" }, async page => {
      const result = await page.evaluate(async () => {
        try { await window.__uvtTest.api("/meta"); }
        catch (error) { return { message: error.message, code: error.code, version: window.__uvtTest.SCRIPT_VERSION, nativeCalls: window.__nativeCalls }; }
      });
      assert.equal(result.code, "UVT_CONNECTION");
      assert.match(result.message, /Обновите userscript/);
      assert.ok(result.message.includes(result.version));
      assert.equal(result.nativeCalls, 1);
    });
    await scenario("actual profile UI selects model and submits override while CSP blocks fetch", {}, page => checkUi(page, false));
    if (process.env.UVT_LIVE_META === "1") {
      await scenario("read-only real UVT metadata through GM bridge fills UI and selected profile reaches mocked POST", { live: true }, page => checkUi(page, true));
    }
    assert.deepEqual(browserErrors, []);
    console.log(JSON.stringify({ passed: results.length, scenarios: results, browserErrors,
      transport: "GM API simulator in real Chrome", liveMetadataReads: liveReads,
      realTranslationRequests: 0 }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
