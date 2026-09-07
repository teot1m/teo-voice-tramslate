/* Real settings UI with neutral DOM and in-memory, engine-specific metadata. */
const assert = require('node:assert/strict');
const { chromium } = require(process.env.UVT_PLAYWRIGHT_MODULE || 'playwright');
const script = process.argv[2];

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  page.setDefaultTimeout(7000);
  const errors = [], passed = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('https://uvt-voices.test/**', route => route.fulfill({
    status: 200, contentType: 'text/html',
    body: '<!doctype html><html><head><title>UVT neutral voice settings</title></head><body><video muted style="display:block;width:640px;height:360px;background:#334155"></video></body></html>',
  }));
  await page.goto('https://uvt-voices.test/watch/demo');
  await page.evaluate(() => {
    window.__voiceFixture = { requests: [], hold: '', pending: [], fail: '' };
    const store = new Map();
    window.GM_getValue = (key, fallback) => store.has(key) ? store.get(key) : fallback;
    window.GM_setValue = (key, value) => store.set(key, value);
    const profiles = [
      { id: 'local-moss', label: 'MOSS', installed: true, engines: { stt: 'parakeet-mlx', translation: 'hymt-mlx', tts: 'moss' } },
      { id: 'local-nemotron', label: 'Nemotron', installed: true, engines: { stt: 'nemotron', translation: 'hymt-mlx', tts: 'piper' } },
      { id: 'local-quality', label: 'Качество', installed: true, engines: { stt: 'mlx-whisper', translation: 'translategemma-mlx', tts: 'piper' } },
    ];
    const catalogs = {
      moss: ['Adam', 'Nathan', 'Ava', 'Bella'].map((label, i) => ({ id: label.toLowerCase(), label, language: 'ru', gender: i < 2 ? 'male' : 'female', installed: true })),
      piper: [
        { id: 'ru_RU-dmitri-medium', label: 'Дмитрий', language: 'ru', gender: 'male', installed: true },
        { id: 'ru_RU-irina-medium', label: 'Ирина', language: 'ru', gender: 'female', installed: true },
        { id: 'uk_UA-mykyta-high', label: 'Микита', language: 'uk', gender: 'male', installed: true },
        { id: 'uk_UA-tetiana-high', label: 'Тетяна', language: 'uk', gender: 'female', installed: true },
      ],
    };
    window.GM_xmlhttpRequest = options => {
      const url = new URL(options.url);
      const data = options.data ? JSON.parse(options.data) : undefined;
      window.__voiceFixture.requests.push({ path: url.pathname, query: url.search, data });
      let aborted = false;
      const deliver = () => {
        if (aborted) return;
        // Saved global settings win for legacy requests, just like UVT.
        const overrides = url.searchParams.get('settings_mode') === 'override';
        const selected = overrides ? url.searchParams.get('profile_id') || 'local-moss' : 'local-moss';
        let status = 200, response;
        if (url.pathname === '/meta') {
          if (window.__voiceFixture.fail === selected) {
            status = 422; response = 'Выбранный профиль сейчас недоступен';
          } else {
            const profile = profiles.find(item => item.id === selected);
            response = {
              api_version: 2, mode: 'batch',
              profile: { name: selected, source_lang: 'en', target_lang: url.searchParams.get('target_lang') || 'ru', engines: profile.engines },
              defaults: { profile_id: 'local-moss', voice_gender: 'auto', voice_id: 'adam' },
              profiles, voices: catalogs[profile.engines.tts],
              limits: { local_tts_languages: ['ru', 'uk'] },
              capabilities: { profile_selection: true, voice_selection: true, tts_preview: true },
              model_readiness: { status: 'ready', detail: 'Модели готовы' }, privacy: { data_leaves_device: false },
            };
          }
        } else if (url.pathname === '/dub') {
          // Match the server's strict string contract: null must never reach it.
          if ('voice_id' in data && typeof data.voice_id !== 'string') {
            status = 422; response = 'voice_id должно быть строкой';
          } else response = { id: 'voice-test-job', mode: 'batch', status: 'queued' };
        } else if (url.pathname === '/tts/preview') {
          // Hold audio response: this test checks the serialized request only.
          return;
        } else response = { id: 'voice-test-job', status: 'cancelled' };
        options.onload({ status, responseHeaders: 'Content-Type: application/json',
          response: new TextEncoder().encode(typeof response === 'string' ? response : JSON.stringify(response)).buffer });
      };
      if (url.pathname === '/meta' && url.searchParams.get('profile_id') === window.__voiceFixture.hold) {
        window.__voiceFixture.pending.push(deliver);
      } else setTimeout(deliver, 0);
      return { abort() { aborted = true; if (options.onabort) options.onabort(); } };
    };
  });
  await page.addScriptTag({ path: script });
  const profile = page.getByLabel('Профиль:', { exact: true });
  const concrete = page.getByLabel('Конкретный:', { exact: true });
  const target = page.getByLabel('На какой:', { exact: true });
  const values = () => concrete.locator('option').evaluateAll(options => options.map(option => option.value));
  async function open() {
    await page.waitForTimeout(720);
    await page.locator('video').hover();
    await page.getByRole('button', { name: /Настройки пакетного перевода:/ }).click();
    await concrete.waitFor();
    await page.waitForFunction(() => !document.querySelector('[id^="uvt-voice-model"]').disabled);
  }
  async function ready(id, voice) {
    await page.waitForFunction(({ id, voice }) => {
      const profile = document.querySelector('[id^="uvt-local-profile"]');
      const concrete = document.querySelector('[id^="uvt-voice-model"]');
      return profile.value === id && !concrete.disabled && [...concrete.options].some(option => option.value === voice);
    }, { id, voice });
  }
  async function close() {
    await page.locator('.uvt-panel').focus(); await page.keyboard.press('Escape');
    await page.locator('.uvt-panel').waitFor({ state: 'detached' });
  }
  try {
    await open();
    assert.deepEqual(await values(), ['', 'adam', 'nathan', 'ava', 'bella']);
    assert.equal(await concrete.inputValue(), 'adam');
    passed.push('global MOSS defaults display MOSS voices');

    await page.evaluate(() => { window.__voiceFixture.hold = 'local-nemotron'; });
    await profile.selectOption('local-nemotron');
    await page.waitForFunction(() => window.__voiceFixture.pending.length === 1);
    assert.equal(await concrete.isDisabled(), true);
    assert.deepEqual(await values(), [''], 'previous engine voices disappear immediately');
    await profile.selectOption('local-moss');
    await ready('local-moss', 'adam');
    await page.evaluate(() => { window.__voiceFixture.hold = ''; window.__voiceFixture.pending.splice(0).forEach(resolve => resolve()); });
    await page.waitForTimeout(150);
    assert.deepEqual(await values(), ['', 'adam', 'nathan', 'ava', 'bella']);
    passed.push('pending catalog hides stale voices; late response cannot replace newer profile');

    await page.evaluate(() => { window.__voiceFixture.fail = 'local-nemotron'; });
    await profile.selectOption('local-nemotron');
    await page.getByText(/Не удалось получить голоса профиля/, { exact: false }).first().waitFor({ state: 'attached' });
    assert.equal(await concrete.isDisabled(), true);
    assert.deepEqual(await values(), ['']);
    assert.equal(await profile.isDisabled(), false);
    passed.push('failed profile refresh exposes no stale MOSS options and keeps profile selectable');

    await page.evaluate(() => { window.__voiceFixture.fail = ''; });
    await profile.selectOption('local-moss'); await ready('local-moss', 'adam');
    await profile.selectOption('local-nemotron'); await ready('local-nemotron', 'ru_RU-dmitri-medium');
    assert.deepEqual(await values(), ['', 'ru_RU-dmitri-medium', 'ru_RU-irina-medium']);
    assert.equal(await concrete.inputValue(), '');
    await close();
    await page.waitForTimeout(720); await page.locator('video').hover();
    await page.getByRole('button', { name: 'Подготовить пакетный перевод и синхронную аудиодорожку', exact: true }).click();
    await page.waitForFunction(() => window.__voiceFixture.requests.some(request => request.path === '/dub'));
    const body = await page.evaluate(() => window.__voiceFixture.requests.find(request => request.path === '/dub').data);
    assert.equal(body.profile_id, 'local-nemotron');
    assert.equal(body.voice_id, '', 'automatic voice explicitly clears incompatible inherited ID with string');
    await page.waitForFunction(() => document.querySelector('.uvt-progress').hidden);
    passed.push('MOSS to Piper uses correct catalog and sends valid empty voice string');

    await open();
    await concrete.selectOption('ru_RU-irina-medium');
    await profile.selectOption('local-quality'); await ready('local-quality', 'ru_RU-irina-medium');
    assert.equal(await concrete.inputValue(), 'ru_RU-irina-medium', 'compatible same-engine choice remains');
    await target.selectOption('uk'); await ready('local-quality', 'uk_UA-tetiana-high');
    assert.deepEqual(await values(), ['', 'uk_UA-mykyta-high', 'uk_UA-tetiana-high']);
    assert.equal(await concrete.inputValue(), '');
    passed.push('same-engine profile preserves compatible voice; language change removes incompatible voice');

    await page.getByText('Прослушать локальные голоса', { exact: true }).click();
    const previewButton = page.getByRole('button', { name: /Прослушать мужской/ });
    await previewButton.click();
    await page.waitForFunction(() => window.__voiceFixture.requests.some(request => request.path === '/tts/preview'));
    const previewBody = await page.evaluate(() => window.__voiceFixture.requests.find(request => request.path === '/tts/preview').data);
    assert.equal(previewBody.voice_id, '');
    assert.equal(previewBody.profile_id, 'local-quality');
    await close();
    passed.push('voice preview also sends string clearing instead of null');

    await open();
    await page.getByRole('button', { name: 'Вернуть значения из глобальных настроек для этого видео', exact: true }).click();
    await ready('local-moss', 'adam');
    assert.equal(await concrete.inputValue(), 'adam');
    assert.equal(await target.inputValue(), 'ru');
    passed.push('reset restores global MOSS catalog and selected voice');

    assert.deepEqual(errors, []);
    process.stdout.write(JSON.stringify({ passed: passed.length, scenarios: passed, browserErrors: errors }) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
