/* Real DOM fixtures for all-player visibility and idle controls. No external media. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.UVT_PLAYWRIGHT_MODULE || "playwright");
const script = process.argv[2];

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  const results = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  let fixture = "";
  await page.route("https://uvt-preview.test/**", route => route.fulfill({
    status: 200, contentType: "text/html", body: fixture
  }));
  const video = (id, attrs = "controls", style = "width:640px;height:360px") =>
    `<video id="${id}" ${attrs} style="${style}"></video>`;
  const selected = () => page.locator(".uvt-wrap").evaluateAll(
    wrappers => wrappers.map(wrapper => wrapper.__uvtVideo.id).sort());
  async function waitFor(expected) {
    await page.waitForFunction(expected => {
      const ids = [...document.querySelectorAll(".uvt-wrap")].map(w => w.__uvtVideo.id).sort();
      return JSON.stringify(ids) === JSON.stringify(expected);
    }, expected, { timeout: 4000 });
    assert.deepEqual(await selected(), expected);
  }
  async function opacity(value) {
    await page.waitForFunction(value => getComputedStyle(document.querySelector(".uvt-wrap")).opacity === value,
      value, { timeout: 5000 });
  }
  async function scenario(name, body, expected, further = null) {
    fixture = `<html><head><title>UVT neutral player test</title><style>body{margin:0}video{display:block;background:#334155}
      .uvt-wrap{font-family:sans-serif}</style></head><body>${body}</body></html>`;
    await page.goto("https://uvt-preview.test/watch/demo");
    await page.addScriptTag({ path: script });
    await waitFor(expected);
    if (further) await further();
    results.push(name);
  }
  try {
    await scenario("deep generic player", `<main><section><div>${video("deep", "muted")}</div></section></main>`, ["deep"]);
    await scenario("YouTube watch columns and sidebar", `<ytd-watch-flexy><div style="display:grid;grid-template-columns:2fr 1fr"><div><div id="movie_player">${video("youtube", "muted")}</div><h1>Example video</h1><a href="/channel/demo">Channel</a></div><aside><a href="/watch/next">Next</a></aside></div></ytd-watch-flexy>`, ["youtube"]);
    await scenario("custom player beside recommendation sidebar", `<div style="display:grid;grid-template-columns:2fr 1fr"><section><div class="player">${video("watch", "muted")}</div><a href="/author">Author</a></section><aside><a href="/next">Next</a></aside></div>`, ["watch"]);
    await scenario("outer recommendation labels do not hide video", `<div class="related-content video-list-page"><main>${video("page")}</main></div>`, ["page"]);
    await scenario("linked video is eligible again", `<a href="/watch/other">${video("linked")}</a>`, ["linked"]);
    await scenario("preview class cannot hide a real player", `<div class="video-preview jwplayer">${video("named-preview", "muted")}</div>`, ["named-preview"]);
    await scenario("all separate visible players receive controls", `<main>${video("first", "muted")}<div class="video-card">${video("second", "muted", "width:320px;height:180px")}</div></main>`, ["first", "second"]);
    await scenario("video collection remains eligible", `<div class="videos-grid" role="feed">${video("feed", "muted")}</div>`, ["feed"]);
    await scenario("proprietary player", `<div class="mgp_videoWrapper">${video("custom", "muted autoplay")}</div>`, ["custom"]);
    await scenario("muted autoplay short loop", `<main>${video("loop", "controls muted autoplay loop")}</main>`, ["loop"]);
    await scenario("small player", `<section>${video("small", "muted", "width:240px;height:160px")}</section>`, ["small"]);
    await scenario("hidden videos excluded", `${video("display-none", "", "display:none;width:640px;height:360px")}${video("invisible", "", "visibility:hidden;width:640px;height:360px")}${video("transparent", "", "opacity:0;width:640px;height:360px")}`, []);
    await scenario("offscreen player appears when scrolled into view", `<div style="height:1200px"></div>${video("below")}`, [], async () => {
      await page.locator("video").scrollIntoViewIfNeeded();
      await waitFor(["below"]);
    });
    await scenario("reparenting does not discard video controls", `<div id="host">${video("moving")}</div><div class="video-card" id="card"></div>`, ["moving"], async () => {
      await page.evaluate(() => document.querySelector("#card").append(document.querySelector("video")));
      await waitFor(["moving"]);
      await page.evaluate(() => document.querySelector("video").remove());
      await waitFor([]);
    });
    await scenario("preview classes changing do not hide controls", `<div>${video("dynamic")}</div>`, ["dynamic"], async () => {
      await page.evaluate(() => { document.querySelector("video").className = "hoverPreview"; });
      await waitFor(["dynamic"]);
    });
    await scenario("SPA navigation resets reused player state", `${video("spa")}`, ["spa"], async () => {
      await page.evaluate(() => {
        document.querySelector(".uvt-wrap").dataset.oldJob = "yes";
        history.pushState({}, "", "/watch/new-video");
        document.querySelector("video").setAttribute("data-preview", "true");
      });
      await page.waitForFunction(() => !document.querySelector(".uvt-wrap[data-old-job]"));
      await waitFor(["spa"]);
    });
    await scenario("stacked players only attach to active video", `<div style="position:relative">${video("idle", "", "position:absolute;width:640px;height:360px")}${video("active", "", "position:absolute;width:640px;height:360px")}</div>`, ["idle"], async () => {
      await page.evaluate(() => Object.defineProperty(document.querySelector("#active"), "paused", {get: () => false}));
      await waitFor(["active"]);
    });
    await scenario("small inset is a separate player, not a duplicate", `<div style="position:relative;width:640px;height:360px">${video("background", "muted")}${video("inset", "muted", "position:absolute;right:0;bottom:0;width:240px;height:160px")}</div>`, ["background", "inset"]);
    await scenario("fullscreen excludes another playing video", `<div id="full">${video("full-main", "muted")}<button id="expand">Open video</button></div>${video("sidebar", "muted", "width:320px;height:180px")}`, ["full-main", "sidebar"], async () => {
      await page.evaluate(() => {
        Object.defineProperty(document.querySelector("#sidebar"), "paused", {get: () => false});
        document.querySelector("#expand").onclick = () => document.querySelector("#full").requestFullscreen();
      });
      await page.locator("#expand").click();
      await page.waitForFunction(() => !!document.fullscreenElement);
      await waitFor(["full-main"]);
      await page.evaluate(() => document.exitFullscreen());
      await waitFor(["full-main", "sidebar"]);
    });
    await scenario("fullscreen keeps accessible controls", `<div id="full">${video("full-video")}<button id="expand">Open video</button></div>`, ["full-video"], async () => {
      await page.evaluate(() => document.querySelector("#expand").onclick = () => document.querySelector("#full").requestFullscreen());
      await page.locator("#expand").click();
      await page.waitForFunction(() => document.querySelector(".uvt-wrap").parentElement === document.fullscreenElement);
      await page.evaluate(() => document.exitFullscreen());
      await page.waitForFunction(() => document.querySelector(".uvt-wrap").parentElement === document.body);
    });
    await scenario("nested iframe player", `<iframe id="embed-frame" style="width:640px;height:400px" srcdoc="<main><section><div><video id='iframe-video' muted style='width:600px;height:340px'></video></div></section></main>"></iframe>`, [], async () => {
      const frame = page.frames().find(frame => frame !== page.mainFrame());
      await frame.addScriptTag({ path: script });
      await frame.waitForFunction(() => document.querySelector(".uvt-wrap")?.__uvtVideo.id === "iframe-video");
      assert.equal(await frame.locator(".uvt-wrap").count(), 1);
    });
    await scenario("idle over controls fully hides, passes clicks, then movement restores", `<main>${video("idle", "muted")}</main>`, ["idle"], async () => {
      const chip = page.getByRole("button", { name: /Настройки пакетного перевода:/ });
      const box = await chip.boundingBox();
      const point = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
      await page.mouse.move(point.x, point.y);
      await opacity("1");
      if (process.env.UVT_SCREENSHOT_DIR) await page.screenshot({path: `${process.env.UVT_SCREENSHOT_DIR}/uvt-controls-visible.png`});
      await opacity("0");
      assert.equal(await page.evaluate(p => !!document.elementFromPoint(p.x, p.y)?.closest(".uvt-wrap"), point), false);
      assert.equal(await chip.evaluate(el => getComputedStyle(el).pointerEvents), "none");
      await page.waitForTimeout(600);
      assert.equal(await page.locator(".uvt-wrap").evaluate(el => getComputedStyle(el).opacity), "0", "stationary pointer must not wake the layer");
      if (process.env.UVT_SCREENSHOT_DIR) await page.screenshot({path: `${process.env.UVT_SCREENSHOT_DIR}/uvt-controls-hidden.png`});
      await page.mouse.move(point.x + 2, point.y + 2);
      await opacity("1");
      assert.equal(await page.evaluate(p => !!document.elementFromPoint(p.x + 2, p.y + 2)?.closest(".uvt-wrap"), point), true);
    });
    await scenario("keyboard focus wakes and retains controls", `<main>${video("keyboard", "muted")}</main>`, ["keyboard"], async () => {
      await page.mouse.move(1000, 800);
      await opacity("0");
      await page.keyboard.press("Tab");
      await opacity("1");
      assert.equal(await page.evaluate(() => !!document.activeElement.closest(".uvt-wrap")), true);
      await page.waitForTimeout(3300);
      assert.equal(await page.locator(".uvt-wrap").evaluate(el => getComputedStyle(el).opacity), "1");
    });
    await page.setViewportSize({width:390,height:844});
    await scenario("mobile controls fit the player", `<main>${video("mobile", "muted", "width:100%;height:220px")}</main>`, ["mobile"], async () => {
      assert.equal(await page.locator(".uvt-wrap").evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth; }), true);
      if (process.env.UVT_SCREENSHOT_DIR) await page.screenshot({path: `${process.env.UVT_SCREENSHOT_DIR}/uvt-controls-mobile.png`});
    });
    assert.deepEqual(errors, [], "userscript emitted browser errors");
    console.log(JSON.stringify({ passed: results.length, scenarios: results, browserErrors: errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
