/* Per-video profile selection with a neutral page and mocked UVT metadata. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.UVT_PLAYWRIGHT_MODULE || "playwright");
const script = process.argv[2];

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  page.setDefaultTimeout(6000);
  const browserErrors = [];
  const results = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  let fixture = fs.readFileSync(path.join(path.dirname(script), "tests/userscript-fixture.html"), "utf8")
    .replace('<video controls muted></video>', '<video id="first" muted style="height:250px"></video><video id="second" muted style="height:250px"></video>')
    .replace('<script src="../uvt.user.js"></script>', '')
    .replace("id:'local-quality',label:'Качество',installed:false", "id:'local-quality',label:'Качество',installed:true")
    .replace("window.fetch = async", "window.uvtFixture.profiles = profiles;\nwindow.fetch = async");
  await page.route("https://uvt-settings.test/**", route => route.fulfill({
    status: 200, contentType: "text/html", body: fixture
  }));
  const wrapper = id => page.locator(`.uvt-wrap[data-video-id="${id}"]`);
  const profile = () => page.locator('[id^="uvt-local-profile"]');
  async function open(id) {
    await page.waitForTimeout(720); // userscript debounces repeated pointer actions
    await page.locator(`#${id}`).hover();
    await wrapper(id).getByRole("button", { name: /Настройки пакетного перевода:/ }).click();
    await page.locator('.uvt-panel').waitFor();
    await page.waitForFunction(() => !document.querySelector('.uvt-panel').textContent.includes('Проверяю сервер и модели…'));
  }
  async function close() {
    await page.locator(".uvt-panel").focus();
    await page.keyboard.press("Escape");
    await page.locator(".uvt-panel").waitFor({ state: "detached" });
  }
  async function latestJob(id) {
    await page.waitForTimeout(720);
    await page.locator(`#${id}`).hover();
    const before = await page.evaluate(() => window.uvtFixture.requests.filter(r => r.path === '/dub').length);
    await wrapper(id).getByRole("button", { name: "Подготовить пакетный перевод и синхронную аудиодорожку", exact: true }).click();
    await page.waitForFunction(before => window.uvtFixture.requests.filter(r => r.path === '/dub').length > before, before);
    const body = await page.evaluate(() => window.uvtFixture.requests.filter(r => r.path === '/dub').at(-1).body);
    await wrapper(id).getByRole("button", { name: "Отменить подготовку пакетного перевода", exact: true }).click();
    await page.waitForFunction(id => document.querySelector(`.uvt-wrap[data-video-id="${id}"] .uvt-progress`).hidden, id);
    return body;
  }
  try {
    await page.goto("https://uvt-settings.test/watch/example");
    await page.addScriptTag({ path: script });
    await page.waitForFunction(() => document.querySelectorAll('.uvt-wrap').length === 2);
    await page.locator('.uvt-wrap').evaluateAll(wrappers => wrappers.forEach(w => w.dataset.videoId = w.__uvtVideo.id));
    await open("first");
    assert.equal(await profile().inputValue(), "local-balanced");
    assert.equal(await page.locator('[id^="uvt-source"]').inputValue(), "en");
    assert.equal(await page.locator('[id^="uvt-target"]').inputValue(), "uk");
    assert.equal(await page.locator('[id^="uvt-voice-model"]').inputValue(), "uk_UA-tetiana-high");
    for (const id of ["local-fast", "local-balanced", "local-quality", "local-natural", "unknown"]) {
      assert.equal(await profile().locator(`option[value="${id}"]`).isDisabled(), false);
    }
    assert.equal(await profile().textContent().then(text => text.includes("проверка")), false);
    results.push("global defaults displayed, installed and unverified profiles selectable");

    await profile().selectOption("local-quality");
    await page.waitForFunction(() => document.querySelector('.uvt-profile-guide').textContent.startsWith('Качество:'));
    assert.equal(await page.locator('[id^="uvt-settings-mode"]').inputValue(), "override");
    assert.equal(await page.locator('[id^="uvt-target"]').inputValue(), "uk");
    await close();
    await open("second");
    assert.equal(await profile().inputValue(), "local-balanced");
    await profile().selectOption("local-fast");
    await page.waitForFunction(() => document.querySelector('.uvt-profile-guide').textContent.startsWith('Быстро:'));
    await page.locator('[id^="uvt-target"]').selectOption("ru");
    await page.waitForFunction(() => document.querySelector('[id^="uvt-target"]').value === 'ru');
    await close();
    await open("first");
    assert.equal(await profile().inputValue(), "local-quality");
    assert.equal(await page.locator('[id^="uvt-target"]').inputValue(), "uk");
    assert.equal(await page.evaluate(() => GM_getValue('uvt.localProfile', 'unwritten')), "unwritten");
    assert.equal(await page.evaluate(() => GM_getValue('uvt.target', 'unwritten')), "unwritten");
    if (process.env.UVT_SETTINGS_SCREENSHOT) {
      await page.locator('.uvt-panel').evaluate(panel => { panel.scrollTop = panel.querySelector('.uvt-profile-guide').offsetTop - 150; });
      await page.screenshot({ path: process.env.UVT_SETTINGS_SCREENSHOT });
    }
    await close();
    const first = await latestJob("first");
    assert.equal(first.settings_mode, "override");
    assert.equal(first.profile_id, "local-quality");
    assert.equal(first.source_lang, "en");
    assert.equal(first.target_lang, "uk");
    assert.equal(first.voice_id, "uk_UA-tetiana-high");
    const second = await latestJob("second");
    assert.equal(second.profile_id, "local-fast");
    assert.equal(second.target_lang, "ru");
    assert.equal(second.voice_id, null, "incompatible inherited voice is not sent to a different language");
    results.push("two videos retain independent overrides and correct submission payloads without shared preference writes");

    await open("first");
    await page.getByRole("button", { name: "Вернуть значения из глобальных настроек для этого видео", exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[id^="uvt-local-profile"]').value === 'local-balanced');
    assert.equal(await page.locator('[id^="uvt-settings-mode"]').inputValue(), "server");
    await close();
    const defaults = await latestJob("first");
    assert.equal(defaults.settings_mode, "server");
    for (const field of ['source_lang', 'target_lang', 'voice_gender', 'voice_id', 'profile_id']) assert.equal(field in defaults, false);
    results.push("reset restores globals and omits manual fields from job");

    await page.evaluate(() => { window.uvtFixture.unavailable = true; });
    await open("first");
    assert.equal(await profile().isDisabled(), false);
    await profile().selectOption("local-natural");
    await page.waitForFunction(() => document.querySelector('.uvt-panel').textContent.includes('Сервер недоступен.'));
    assert.equal(await profile().inputValue(), "local-natural");
    await close();
    await page.evaluate(() => { window.uvtFixture.unavailable = false; });
    const offlineChoice = await latestJob("first");
    assert.equal(offlineChoice.profile_id, "local-natural");
    assert.equal(offlineChoice.target_lang, "uk");
    assert.equal(offlineChoice.source_lang, "en");
    results.push("profile can be chosen offline and inherits server defaults when starting after reconnect");

    await open("second");
    await page.locator('[id^="uvt-settings-route"]').selectOption("cloud");
    await page.waitForFunction(() => document.querySelector('[id^="uvt-local-profile"]').disabled);
    await close();
    const cloud = await latestJob("second");
    assert.equal(cloud.profile_id, null, "local profile is not submitted to a cloud route");
    assert.equal(cloud.target_lang, "ru");
    await open("second");
    await page.locator('[id^="uvt-settings-route"]').selectOption("free");
    await page.waitForFunction(() => document.querySelector('[id^="uvt-local-profile"]').value === 'local-fast' && !document.querySelector('[id^="uvt-local-profile"]').disabled);
    await close();
    results.push("cloud route ignores local profile and switching back restores the selected local profile");

    await open("first");
    await profile().selectOption("local-quality");
    await page.waitForFunction(() => document.querySelector('.uvt-profile-guide').textContent.startsWith('Качество:'));
    await close();
    await page.evaluate(() => { window.uvtFixture.profiles.find(p => p.id === 'local-quality').installed = false; });
    const requestCount = await page.evaluate(() => window.uvtFixture.requests.filter(r => r.path === '/dub').length);
    await wrapper('first').getByRole("button", { name: "Подготовить пакетный перевод и синхронную аудиодорожку", exact: true }).click();
    await page.getByText(/Модели выбранного профиля ещё не установлены/).first().waitFor();
    assert.equal(await page.evaluate(() => window.uvtFixture.requests.filter(r => r.path === '/dub').length), requestCount);
    results.push("profile removed after selection produces a useful error without silently switching models");
    assert.deepEqual(browserErrors, []);
    console.log(JSON.stringify({ passed: results.length, scenarios: results, browserErrors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
