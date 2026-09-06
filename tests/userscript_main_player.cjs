/* Neutral real-DOM regression fixtures; no external media or server needed. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.UVT_PLAYWRIGHT_MODULE || "playwright");
const script = process.argv[2];

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  const results = [];
  page.on("pageerror", error => errors.push(error.message));
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
  async function scenario(name, body, expected, further = null) {
    fixture = `<html><head><style>body{margin:0}video{display:block;background:#334155}
      .uvt-wrap{font-family:sans-serif}</style></head><body>${body}</body></html>`;
    await page.goto("https://uvt-preview.test/watch/demo");
    await page.addScriptTag({ path: script });
    await waitFor(expected);
    if (further) await further();
    results.push(name);
  }
  try {
    await scenario("large linked card, even with controls", `<a href="/watch/other">${video("card")}</a>`, []);
    await scenario("hover preview carrying a player class", `<div class="video-preview jwplayer">${video("hover")}</div>`, []);
    await scenario("single lazy hover video in a grid", `<div class="videos-grid"><div>${video("grid", "muted autoplay loop")}</div></div>`, []);
    await scenario("semantic feed excludes unlabelled video entries", `<section role="feed"><article>${video("feed")}</article></section>`, []);
    await scenario("YouTube custom-element preview card", `<ytd-rich-grid-media><div id="movie_player">${video("yt-card")}</div></ytd-rich-grid-media>`, []);
    await scenario("YouTube full watch player stays eligible", `<ytd-watch-flexy><div id="movie_player">${video("yt-main", "muted")}</div></ytd-watch-flexy>`, ["yt-main"]);
    await scenario("semantic list with unlabelled cards", `<ul><li>${video("list")}<a href="/watch/other">Title</a></li><li><a href="/watch/another">Next</a></li></ul>`, []);
    await scenario("CSS grid with unlabelled cards", `<div style="display:grid;grid-template-columns:1fr 1fr"><div>${video("css-grid")}<a href="/watch/other">Title</a></div><div><a href="/watch/another">Next</a></div></div>`, []);
    await scenario("muted autoplay native main player", `<main>${video("native", "controls muted autoplay loop")}</main>`, ["native"]);
    await scenario("hover control state is not a hover preview", `<main><div class="player hover">${video("hover-state", "muted")}</div></main>`, ["hover-state"]);
    await scenario("generic card styling around full player", `<main><section class="card"><div class="player">${video("styled")}</div></section></main>`, ["styled"]);
    await scenario("clickable grid cards without href links", `<div style="display:grid"><div class="card" role="link">${video("js-card")}</div><div class="card">Next</div></div>`, []);
    await scenario("custom main player without native controls", `<main><div class="jwplayer">${video("custom", "muted")}</div></main>`, ["custom"]);
    await scenario("proprietary camelCase main player", `<main><div class="mgp_videoWrapper">${video("proprietary", "muted autoplay")}</div></main>`, ["proprietary"]);
    await scenario("small standalone embed without native controls", `<div>${video("embed", "muted", "width:320px;height:180px")}</div>`, ["embed"]);
    await scenario("main player with large preview beside it", `<main><div class="player">${video("main")}</div><div class="video-card">${video("related")}</div></main>`, ["main"]);
    await scenario("portrait short clip", `<main>${video("short", "controls muted loop", "width:240px;height:420px")}</main>`, ["short"]);
    await scenario("article embed remains usable", `<main><article><h1>Lesson</h1>${video("lesson")}<p><a href="/next">Next lesson</a></p></article></main>`, ["lesson"]);
    await scenario("explicit ad preview rejected", `<div class="ad-video player">${video("ad")}</div>`, []);
    await scenario("unrelated silent thumbnail does not receive overlay", `<main>${video("main")}<aside>${video("thumb", "muted loop", "width:260px;height:140px")}</aside></main>`, ["main"]);
    await scenario("below-fold giant video does not steal overlay", `${video("main")}<div style="margin-top:1100px">${video("below", "controls", "width:1100px;height:600px")}</div>`, ["main"]);
    await scenario("popup opens from a card context", `<div class="video-card"><div role="dialog" aria-modal="true"><div class="player">${video("popup")}</div></div></div>`, ["popup"]);
    await scenario("grid previews inside a dialog stay excluded", `<div role="dialog"><div class="video-grid">${video("popup-preview")}</div></div>`, []);
    await scenario("reparenting main player into a card removes controls", `<div class="player" id="host">${video("moving")}</div><div class="video-card" id="card"></div>`, ["moving"], async () => {
      await page.evaluate(() => document.querySelector("#card").append(document.querySelector("#moving")));
      await waitFor([]);
      await page.evaluate(() => document.querySelector("#host").append(document.querySelector("#moving")));
      await waitFor(["moving"]);
    });
    await scenario("lazy preview class changes trigger a fresh decision", `<div class="player">${video("dynamic")}</div>`, ["dynamic"], async () => {
      await page.evaluate(() => { document.querySelector("video").className = "hoverPreview"; });
      await waitFor([]);
      await page.evaluate(() => { document.querySelector("video").className = ""; });
      await waitFor(["dynamic"]);
    });
    await scenario("SPA navigation replaces stale overlay on reused video", `<div class="player">${video("spa")}</div>`, ["spa"], async () => {
      await page.evaluate(() => {
        document.querySelector(".uvt-wrap").dataset.oldJob = "yes";
        history.pushState({}, "", "/watch/new-video");
        document.querySelector("video").setAttribute("data-preview", "true");
      });
      await waitFor([]);
      await page.evaluate(() => document.querySelector("video").removeAttribute("data-preview"));
      await waitFor(["spa"]);
      assert.equal(await page.locator(".uvt-wrap[data-old-job]").count(), 0);
    });
    await scenario("fullscreen expansion overrides card classification", `<div class="video-card" id="full">${video("full-video")}<button id="expand">Open video</button></div>`, [], async () => {
      await page.evaluate(() => document.querySelector("#expand").onclick = () => document.querySelector("#full").requestFullscreen());
      await page.locator("#expand").click();
      await page.waitForFunction(() => !!document.fullscreenElement);
      await waitFor(["full-video"]);
      await page.evaluate(() => document.exitFullscreen());
      await waitFor([]);
    });
    await scenario("plain page layout named list-page is not a preview", `<main class="list-page"><div class="player">${video("plain")}</div></main>`, ["plain"]);
    await scenario("legitimate nested iframe player", `<iframe id="embed-frame" style="width:640px;height:400px" srcdoc="<main><section><div><video id='iframe-video' muted style='width:600px;height:340px'></video></div></section></main>"></iframe>`, [], async () => {
      const frame = page.frames().find(frame => frame !== page.mainFrame());
      await frame.addScriptTag({ path: script });
      await frame.waitForFunction(() => document.querySelector(".uvt-wrap")?.__uvtVideo.id === "iframe-video");
      assert.equal(await frame.locator(".uvt-wrap").count(), 1);
    });
    assert.deepEqual(errors, [], "userscript emitted browser errors");
    console.log(JSON.stringify({ passed: results.length, scenarios: results, browserErrors: errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
