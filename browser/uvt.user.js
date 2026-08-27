// ==UserScript==
// @name         UVT — закадровый перевод видео
// @namespace    uvt
// @version      0.15.0
// @description  Пакетный закадровый перевод через личный UVT: Free, GPT или ElevenLabs, не live-перевод
// @match        *://*/*
// @grant        GM_getValue
// @grant        GM_setValue
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  // --- настройки ---
  // `uvt serve-personal` поднимает три маршрута. Для удалённого личного VPS
  // замените только URL: каждый маршрут указывает на свой reverse proxy.
  // API-ключи OpenAI/ElevenLabs остаются только на сервере.
  const SERVERS = Object.freeze({
    free: {
      url: "http://127.0.0.1:8765",
      label: "Бесплатный / локальный — профиль UVT-сервера",
      shortLabel: "Free",
    },
    cloud: {
      url: "http://127.0.0.1:8766",
      label: "GPT Cloud — OpenAI",
      shortLabel: "GPT",
    },
    eleven: {
      url: "http://127.0.0.1:8767",
      label: "GPT-перевод + озвучка ElevenLabs",
      shortLabel: "ElevenLabs",
    },
  });
  const LOCAL_PROFILES = Object.freeze({
    "local-fast": {
      label: "Быстро",
      detail: "Parakeet → NLLB INT8 → Piper",
    },
    "local-balanced": {
      label: "Сбалансированный",
      detail: "Parakeet → TranslateGemma 4-bit → Piper",
    },
    "local-quality": {
      label: "Качество",
      detail: "Whisper large-v3-turbo → TranslateGemma → Piper",
    },
  });
  const LOCAL_VOICES = Object.freeze([
    { id: "ru_RU-dmitri-medium", label: "Дмитрий", language: "ru", gender: "male" },
    { id: "ru_RU-irina-medium", label: "Ирина", language: "ru", gender: "female" },
    { id: "uk_UA-mykyta-high", label: "Микита", language: "uk", gender: "male" },
    { id: "uk_UA-tetiana-high", label: "Тетяна", language: "uk", gender: "female" },
  ]);
  // Задайте тот же секрет, что и UVT_API_TOKEN на удалённом сервере. Для
  // localhost оставьте пустую строку. Это личный доступ, не биллинг-аккаунт.
  const UVT_API_TOKEN = "";
  const DEFAULT_DUCK = 0.15;   // громкость оригинала ВО ВРЕМЯ реплики (по умолчанию)
  const DEFAULT_VOICE_VOL = 1; // громкость переведённого голоса (по умолчанию)
  const DRIFT_S = 0.12;      // допустимый рассинхрон перевода с видео
  const SEEK_BIAS_S = 0.05;  // компенсация задержки старта аудиоэлемента
  const WINDOW_LEAD_S = 0.15;  // приглушать чуть раньше начала реплики
  const WINDOW_TAIL_S = 0.3;   // и отпускать чуть позже её конца
  const SPEECH_RATE_S = 0.07;  // оценка длительности озвучки: секунд на символ
  const MIN_VIDEO_WIDTH = 200; // на мелкие превью кнопку не вешаем

  const LANG_NAMES = {
    auto: "авто", ru: "русский", en: "английский", uk: "украинский",
    de: "немецкий", fr: "французский", es: "испанский", it: "итальянский",
    pt: "португальский", pl: "польский", bg: "болгарский", cs: "чешский",
    da: "датский", el: "греческий", et: "эстонский", fi: "финский",
    hr: "хорватский", hu: "венгерский", lt: "литовский", lv: "латышский",
    mt: "мальтийский", nl: "нидерландский", ro: "румынский",
    sk: "словацкий", sl: "словенский", sv: "шведский", ja: "японский",
    zh: "китайский", ko: "корейский", tr: "турецкий", ar: "арабский",
    hi: "хинди",
  };
  const LOCAL_PIPELINE_LANGUAGES = new Set([
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "uk",
  ]);

  function readPref(key, fallback = "") {
    try {
      if (typeof GM_getValue === "function") {
        const value = GM_getValue(key, undefined);
        if (value !== undefined && value !== null) return value;
      }
    } catch (_) { /* прямой запуск без userscript API */ }
    try {
      const legacy = localStorage.getItem(key);
      return legacy === null ? fallback : legacy;
    } catch (_) {
      return fallback;
    }
  }

  function writePref(key, value) {
    try {
      if (typeof GM_setValue === "function") GM_setValue(key, value);
    } catch (_) { /* прямой запуск без userscript API */ }
    try { localStorage.setItem(key, String(value)); } catch (_) { /* storage закрыт */ }
  }

  // Разовая миграция: до v0.8 голос по умолчанию был "male" — переводим на авто.
  if (!readPref("uvt.voice.v2", "")) {
    writePref("uvt.voice.v2", "1");
    writePref("uvt.voice", "auto");
  }

  const prefs = {
    get route() {
      const value = readPref("uvt.route", "free");
      return Object.prototype.hasOwnProperty.call(SERVERS, value) ? value : "free";
    },
    set route(v) { writePref("uvt.route", v); },
    get source() { return readPref("uvt.source", "auto"); },
    set source(v) { writePref("uvt.source", v); },
    get target() { return readPref("uvt.target", "ru"); },
    set target(v) { writePref("uvt.target", v); },
    get voice() { return readPref("uvt.voice", "auto"); },
    set voice(v) { writePref("uvt.voice", v); },
    get voiceId() { return readPref("uvt.voiceId", ""); },
    set voiceId(v) { writePref("uvt.voiceId", v || ""); },
    get localProfile() {
      const value = String(readPref("uvt.localProfile", "") || "");
      return /^[a-z0-9][a-z0-9._-]{0,63}$/i.test(value) ? value : "";
    },
    set localProfile(v) { writePref("uvt.localProfile", v); },
    get duck() {
      const v = parseFloat(readPref("uvt.duck", ""));
      return Number.isFinite(v) ? v : DEFAULT_DUCK;
    },
    set duck(v) { writePref("uvt.duck", String(v)); },
    get voiceVol() {
      const v = parseFloat(readPref("uvt.voiceVol", ""));
      return Number.isFinite(v) ? v : DEFAULT_VOICE_VOL;
    },
    set voiceVol(v) { writePref("uvt.voiceVol", String(v)); },
  };

  const state = new WeakMap();
  let controlId = 0;
  // Browser focus stays on a clicked button, so focus alone must not pin the
  // overlay forever. Remember the latest input modality to preserve controls
  // for keyboard users while letting pointer controls disappear like native
  // video controls.
  let lastInteractionWasKeyboard = false;
  document.addEventListener("keydown", () => { lastInteractionWasKeyboard = true; }, true);
  document.addEventListener("pointerdown", () => { lastInteractionWasKeyboard = false; }, true);

  const STAGE_NAMES = {
    queue: "очередь",
    download: "получение звука",
    transcribe: "распознавание",
    translate: "перевод",
    synthesize: "озвучка",
    mix: "сборка дорожки",
    done: "готово",
    cancelled: "отменено",
    error: "ошибка",
  };

  const CHIP_STYLE = {
    padding: "4px 10px",
    font: "600 12px/1.4 -apple-system, system-ui, sans-serif",
    color: "#fff",
    background: "rgba(20, 20, 20, 0.75)",
    border: "1px solid rgba(255,255,255,0.25)",
    borderRadius: "8px",
    cursor: "pointer",
    userSelect: "none",
    whiteSpace: "nowrap",
    appearance: "none",
    lineHeight: "1.4",
    textAlign: "center",
  };

  function nextControlId(prefix) {
    controlId += 1;
    return `uvt-${prefix}-${controlId}`;
  }

  function setButton(btn, text, bg, label) {
    btn.textContent = text;
    if (bg) btn.style.background = bg;
    if (label) btn.setAttribute("aria-label", label);
  }

  function serverForRoute(route) {
    const value = Object.prototype.hasOwnProperty.call(SERVERS, route) ? route : "free";
    const server = SERVERS[value];
    const configuredUrl = readPref(`uvt.server.${value}`, server.url);
    return { ...server, key: value, url: configuredUrl };
  }

  function urlFor(server, path) {
    const relativePath = String(path || "").replace(/^\/+/, "");
    return new URL(
      relativePath,
      server.url.endsWith("/") ? server.url : server.url + "/"
    ).toString();
  }

  async function apiResponse(path, options, server) {
    const activeServer = server || serverForRoute(prefs.route);
    const headers = new Headers((options && options.headers) || {});
    if (UVT_API_TOKEN) headers.set("X-UVT-Token", UVT_API_TOKEN);
    const response = await fetch(urlFor(activeServer, path), { ...options, headers });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(
        activeServer.shortLabel + ": сервер UVT ответил HTTP " + response.status
        + (detail ? " — " + detail.slice(0, 360) : "")
      );
    }
    return response;
  }

  async function api(path, options, server) {
    const response = await apiResponse(path, options, server);
    return response.json();
  }

  async function apiAudio(path, options, server) {
    const response = await apiResponse(path, options, server);
    return response.blob();
  }

  function makeNotice(wrapper, className, role) {
    const panel = document.createElement("section");
    panel.className = className;
    panel.setAttribute("role", role);
    panel.setAttribute("aria-live", role === "alert" ? "assertive" : "polite");
    panel.setAttribute("aria-atomic", "true");
    Object.assign(panel.style, {
      position: "absolute",
      top: "38px",
      left: "50%",
      transform: "translateX(-50%)",
      width: "min(360px, calc(100vw - 24px))",
      boxSizing: "border-box",
      padding: "9px 10px",
      background: "rgba(15, 15, 15, 0.96)",
      border: "1px solid rgba(255,255,255,0.28)",
      borderRadius: "10px",
      color: "#f4f4f4",
      font: "12px/1.45 -apple-system, system-ui, sans-serif",
      textAlign: "left",
      whiteSpace: "normal",
      boxShadow: "0 8px 26px rgba(0,0,0,.38)",
      zIndex: "2147483647",
    });
    wrapper.appendChild(panel);
    return panel;
  }

  function clearError(video) {
    const s = state.get(video);
    if (!s) return;
    if (s.errorPanel) s.errorPanel.remove();
    s.errorPanel = null;
  }

  function showError(video, error) {
    const s = state.get(video);
    if (!s || !s.wrapper) return;
    clearError(video);
    const panel = makeNotice(s.wrapper, "uvt-error", "alert");
    panel.style.borderColor = "rgba(255, 112, 112, .8)";
    panel.style.background = "rgba(74, 18, 22, .97)";
    s.errorPanel = panel;

    const heading = document.createElement("strong");
    const activeRoute = s.server || serverForRoute(prefs.route);
    heading.textContent = `Не удалось подготовить перевод · ${activeRoute.shortLabel}`;
    heading.style.display = "block";
    panel.appendChild(heading);
    const body = document.createElement("div");
    body.textContent = String(error && error.message ? error.message : error);
    body.style.marginTop = "3px";
    panel.appendChild(body);
    const help = document.createElement("div");
    const errorText = String(error && error.message ? error.message : error);
    const billingError = /HTTP\s+40[123]|доступ или оплату|ключ, тариф|balance/i.test(errorText);
    const rateLimitError = /HTTP\s+429|частот[уы]|Too Many Requests/i.test(errorText);
    const paidRouteError = (billingError || rateLimitError) && activeRoute.key !== "free";
    help.textContent = paidRouteError
      ? rateLimitError
        ? `${activeRoute.shortLabel} временно ограничил частоту запросов. Повторите через минуту или выберите Free: исходный звук уже в общем кэше.`
        : `${activeRoute.shortLabel} отклонил запрос по ключу или квоте. Проверьте остаток символов/баланс либо выберите Free: исходный звук уже в общем кэше.`
      : "Проверьте, что запущен uvt serve-personal, ссылка доступна без DRM, а выбранные движки настроены.";
    help.style.color = "#f2c9c9";
    help.style.marginTop = "4px";
    panel.appendChild(help);

    const actions = document.createElement("div");
    Object.assign(actions.style, { display: "flex", gap: "6px", marginTop: "8px" });
    if (paidRouteError) {
      const useFree = document.createElement("button");
      useFree.type = "button";
      setButton(useFree, "Переключить на Free", "rgba(30, 120, 65, .9)", "Переключить на бесплатный локальный маршрут и повторить");
      Object.assign(useFree.style, CHIP_STYLE, { padding: "3px 7px" });
      useFree.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        prefs.route = "free";
        syncRouteControls("free");
        beginTranslation(video);
      });
      actions.appendChild(useFree);
    }
    const retry = document.createElement("button");
    retry.type = "button";
    setButton(retry, "Повторить", "rgba(255,255,255,.14)", "Повторить пакетную подготовку перевода");
    Object.assign(retry.style, CHIP_STYLE, { padding: "3px 7px" });
    retry.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      beginTranslation(video);
    });
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    setButton(dismiss, "Закрыть", "transparent", "Закрыть сообщение об ошибке");
    Object.assign(dismiss.style, CHIP_STYLE, { padding: "3px 7px" });
    dismiss.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      clearError(video);
    });
    actions.append(retry, dismiss);
    panel.appendChild(actions);
    retry.focus();
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    const rounded = Math.max(1, Math.round(seconds));
    if (rounded < 60) return `~${rounded} с`;
    return `~${Math.floor(rounded / 60)} мин ${rounded % 60} с`;
  }

  function renderJobStatus(video, info) {
    const s = state.get(video);
    if (!s) return;
    const stage = STAGE_NAMES[info.stage] || "подготовка";
    const stageProgress = Number(info.stage_progress);
    const overallPct = Math.round((Number(info.progress) || 0) * 100);
    const stagePct = Number.isFinite(stageProgress)
      ? Math.round(stageProgress * 100)
      : overallPct;
    const routeLabel = (info.route && info.route.label)
      || (s.server && s.server.shortLabel)
      || serverForRoute(prefs.route).shortLabel;
    let message;
    if (info.stage === "queue" || info.status === "queued") {
      const position = Number(info.queue_position);
      message = Number.isFinite(position) && position > 1
        ? `В очереди: перед вами ${position - 1}.`
        : "В очереди: задача следующая.";
    } else {
      message = `Этап: ${stage} (${stagePct}%). Общая готовность: ${overallPct}%.`;
    }
    const eta = Number(info.eta_seconds);
    const etaText = info.eta_is_estimate && Number.isFinite(eta)
      ? `Оценка до готовности ${formatDuration(eta)}.`
      : "";
    setButton(
      s.button,
      `${routeLabel} · ${info.stage === "queue" ? "очередь" : `${stage} ${stagePct}%`}`,
      "rgba(120, 90, 0, 0.85)",
      `Пакетный перевод: ${message} ${etaText}`.trim()
    );
    s.button.setAttribute("aria-busy", "true");
  }

  // --- окна реплик: когда приглушать оригинал ---
  function buildWindows(entries) {
    const raw = (entries || []).map((e) => {
      // Новые batch-ответы несут фактическое окно TTS. Со старым сервером
      // сохраняем прежнюю оценку по тексту, поэтому обновление совместимо.
      const ttsStart = typeof e.tts_start === "number" && Number.isFinite(e.tts_start)
        ? e.tts_start : null;
      const ttsEnd = typeof e.tts_end === "number" && Number.isFinite(e.tts_end)
        ? e.tts_end : null;
      if (ttsStart !== null && ttsEnd !== null && ttsEnd >= ttsStart) {
        return [Math.max(0, ttsStart - WINDOW_LEAD_S), ttsEnd + WINDOW_TAIL_S];
      }
      const start = typeof e.start === "number" && Number.isFinite(e.start) ? e.start : 0;
      const end = typeof e.end === "number" && Number.isFinite(e.end) ? e.end : start;
      const spoken = Math.max(end - start, SPEECH_RATE_S * (e.translated || "").length);
      return [Math.max(0, start - WINDOW_LEAD_S), start + spoken + WINDOW_TAIL_S];
    }).sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const w of raw) {
      const last = merged[merged.length - 1];
      if (last && w[0] <= last[1] + 0.1) last[1] = Math.max(last[1], w[1]);
      else merged.push(w);
    }
    return merged;
  }

  function inWindow(windows, t) {
    for (const [a, b] of windows) {
      if (t < a) return false;
      if (t <= b) return true;
    }
    return false;
  }

  // --- приглушение оригинала ---
  // Единственный безопасный путь — Web Audio GainNode. Мы принципиально не
  // меняем video.volume и не перехватываем volumechange: на стороннем плеере
  // это могло сохранить приглушение после выключения UVT или перетереть выбор
  // пользователя. Если MediaElementSource небезопасен, оригинал не трогаем.
  function createDucker(video) {
    const s = state.get(video);
    const src = video.currentSrc || video.src || "";
    // ТОЛЬКО blob/MSE (YouTube и т.п.): маршрутизация обычного кросс-доменного
    // src через Web Audio глушит элемент навсегда по правилам CORS.
    const safeForWebAudio = src.startsWith("blob:");

    if (safeForWebAudio) {
      try {
        if (!s.webaudio) {
          const Ctx = window.AudioContext || window.webkitAudioContext;
          const ctx = new Ctx();
          const source = ctx.createMediaElementSource(video);
          const gain = ctx.createGain();
          source.connect(gain);
          gain.connect(ctx.destination);
          s.webaudio = { ctx, gain };
        }
        s.webaudio.ctx.resume();
        const gainParam = s.webaudio.gain.gain;
        console.info("[UVT] приглушение через Web Audio");
        return {
          mode: "webaudio",
          set(mult) { if (Math.abs(gainParam.value - mult) > 0.01) gainParam.value = mult; },
          release() { gainParam.value = 1; },
        };
      } catch (err) {
        console.warn("[UVT] Web Audio недоступен; оригинальный звук останется без ducking:", err);
      }
    }

    console.info("[UVT] исходник нельзя безопасно приглушить; громкость оригинала не меняется");
    return {
      mode: "none",
      set() {},
      release() {},
    };
  }

  // --- синхронное воспроизведение готовой дорожки ---

  function attachAudio(video, audioUrl, entries, server) {
    const s = state.get(video);
    const audio = new Audio(urlFor(server, audioUrl));
    audio.preload = "auto";
    s.audio = audio;
    s.windows = buildWindows(entries);
    s.ducker = createDucker(video);
    s.audioErrorHandler = () => {
      if (!s.on) return;
      detachAudio(video);
      setRetryButton(video);
      showError(video, new Error("готовая аудиодорожка не загрузилась с выбранного UVT-сервера"));
    };
    audio.addEventListener("error", s.audioErrorHandler, { once: true });

    // Пока перевод не включён (s.on=false), оригинал никто не трогает:
    // приглушение действует только внутри окон реплик работающего перевода.
    const applyDuck = () => {
      if (!s.on) return;
      s.ducker.set(inWindow(s.windows, video.currentTime) ? prefs.duck : 1);
    };
    const sync = () => {
      if (!s.on) return;
      if (Math.abs(audio.currentTime - video.currentTime) > DRIFT_S) {
        // небольшое упреждение компенсирует задержку старта аудиоэлемента
        audio.currentTime = video.currentTime + SEEK_BIAS_S;
      }
      applyDuck();
    };

    s.handlers = {
      play: () => { audio.play(); sync(); },
      pause: () => audio.pause(),
      seeked: sync,
      timeupdate: sync, // ~4 раза в секунду: и синхрон, и приглушение
      ratechange: () => { audio.playbackRate = video.playbackRate; },
      // Синхронизируем только mute. UVT не читает и не меняет video.volume.
      volumechange: () => { audio.muted = video.muted; },
    };
    for (const [event, fn] of Object.entries(s.handlers)) video.addEventListener(event, fn);
    s.timer = setInterval(sync, 100);
    s.on = true;

    audio.playbackRate = video.playbackRate;
    audio.muted = video.muted;
    audio.volume = prefs.voiceVol;
    if (!video.paused) { audio.currentTime = video.currentTime; audio.play(); }
    applyDuck();
  }

  function detachAudio(video) {
    const s = state.get(video);
    if (!s || !s.audio) return;
    // Сначала вернуть GainNode в 1, пока перевод ещё формально включён. Для
    // fallback это no-op; user-selected video.volume всегда остаётся нетронут.
    if (s.ducker) { s.ducker.release(); s.ducker = null; }
    s.on = false;
    clearInterval(s.timer);
    for (const [event, fn] of Object.entries(s.handlers || {})) video.removeEventListener(event, fn);
    if (s.audioErrorHandler) s.audio.removeEventListener("error", s.audioErrorHandler);
    s.audioErrorHandler = null;
    s.audio.pause();
    s.audio.src = "";
    s.audio = null;
  }

  function mediaUrlOf(video) {
    const src = video.currentSrc || video.src || "";
    return src && !src.startsWith("blob:") ? src : null;
  }

  // SPA-сайты меняют видео без перезагрузки страницы — запоминаем момент
  // перехода, чтобы не отдать серверу потоки ПРЕДЫДУЩЕГО ролика.
  let lastNavigation = 0;
  const markNavigation = () => { lastNavigation = performance.now(); };
  window.addEventListener("popstate", markNavigation);
  for (const method of ["pushState", "replaceState"]) {
    const original = history[method];
    history[method] = function (...args) {
      markNavigation();
      return original.apply(this, args);
    };
  }

  // Плееры с MSE (src=blob:) прячут настоящий адрес потока, но манифесты
  // (.m3u8/.mpd) видны в сетевых ресурсах страницы — отдаём их серверу,
  // ffmpeg скачает поток напрямую, даже если yt-dlp сайт не знает.
  function findMediaCandidates() {
    const streams = [];
    const files = [];
    const freshStreams = [];
    const freshFiles = [];
    try {
      for (const entry of performance.getEntriesByType("resource")) {
        const url = entry.name;
        const fresh = entry.startTime >= lastNavigation;
        // манифест не всегда оканчивается на .m3u8 — ловим и по пути/параметрам
        if (/m3u8|\.mpd([?#]|$)|\/(hls|dash)\//i.test(url)) {
          streams.push(url);
          if (fresh) freshStreams.push(url);
        } else if (/\.(mp4|webm|m4a|mp3)([?#]|$)/i.test(url)) {
          files.push(url);
          if (fresh) freshFiles.push(url);
        }
      }
    } catch (_) { /* performance API недоступен — не страшно */ }
    // потоки текущего видео приоритетнее; манифесты раньше файлов; свежие — первыми
    const useFresh = freshStreams.length > 0 || freshFiles.length > 0;
    const pickStreams = useFresh ? freshStreams : streams;
    const pickFiles = useFresh ? freshFiles : files;
    return {
      list: [...pickStreams.reverse(), ...pickFiles.reverse()].slice(0, 6),
      fresh: useFresh,
    };
  }

  // Не запускаем, не ставим на паузу и не меняем mute плеера ради поиска
  // манифеста. До готовой дорожки UVT вообще не должен менять аудио-состояние
  // сайта; если поток ещё не замечен, сервер сам применит page_url fallback.
  async function ensureCandidates(video) {
    const found = findMediaCandidates();
    if (mediaUrlOf(video)) return found.list; // есть прямой src — этого хватит
    if (found.list.length && (found.fresh || lastNavigation === 0)) return found.list;
    return [];
  }

  // --- запуск перевода ---

  function setIdleButton(video) {
    const s = state.get(video);
    if (!s || !s.button) return;
    setButton(
      s.button,
      "UVT · перевести",
      "rgba(20, 20, 20, 0.75)",
      "Подготовить пакетный перевод и синхронную аудиодорожку"
    );
    s.button.setAttribute("aria-pressed", "false");
    s.button.setAttribute("aria-busy", "false");
    s.button.title = "Подготовить готовую дорожку перевода (не live-перевод)";
  }

  function setEnabledButton(video) {
    const s = state.get(video);
    if (!s || !s.button) return;
    setButton(
      s.button,
      "UVT · выключить",
      "rgba(20, 110, 50, 0.85)",
      "Выключить готовую дорожку перевода"
    );
    s.button.setAttribute("aria-pressed", "true");
    s.button.setAttribute("aria-busy", "false");
    s.button.title = "Готовая дорожка включена; нажмите, чтобы выключить";
  }

  function setRetryButton(video) {
    const s = state.get(video);
    if (!s || !s.button) return;
    setButton(
      s.button,
      "UVT · повторить",
      "rgba(150, 30, 30, 0.85)",
      "Повторить пакетную подготовку перевода"
    );
    s.button.setAttribute("aria-pressed", "false");
    s.button.setAttribute("aria-busy", "false");
  }

  function snapshotPrefs(video) {
    const route = prefs.route;
    const videoState = video && state.get(video);
    const settingsMode = videoState && videoState.settingsMode === "override"
      ? "override" : "server";
    return {
      route,
      server: serverForRoute(route),
      settingsMode,
      source: prefs.source,
      target: prefs.target,
      voice: prefs.voice,
      voiceId: route === "free" ? prefs.voiceId : "",
      profileId: route === "free" ? prefs.localProfile : "",
    };
  }

  async function resolveJobPrefs(jobPrefs, signal) {
    if (jobPrefs.settingsMode !== "override") return jobPrefs;
    const requestOptions = signal ? { signal } : undefined;
    const baseMeta = await api("/meta", requestOptions, jobPrefs.server);
    const capabilities = baseMeta.capabilities || {};
    const profiles = Array.isArray(baseMeta.profiles) ? baseMeta.profiles : [];
    const advertisedProfile = capabilities.profile_selection
      && profiles.some((item) => item.id === jobPrefs.profileId && item.installed !== false)
      ? jobPrefs.profileId
      : "";

    let selectedMeta = baseMeta;
    const baseProfile = baseMeta.profile && baseMeta.profile.name;
    const baseTarget = baseMeta.profile && baseMeta.profile.target_lang;
    if ((advertisedProfile && advertisedProfile !== baseProfile) || jobPrefs.target !== baseTarget) {
      const query = new URLSearchParams({ target_lang: jobPrefs.target });
      if (advertisedProfile) query.set("profile_id", advertisedProfile);
      selectedMeta = await api(`/meta?${query.toString()}`, requestOptions, jobPrefs.server);
    }

    const voices = Array.isArray(selectedMeta.voices) ? selectedMeta.voices : [];
    const exactVoice = jobPrefs.route === "free"
      && selectedMeta.capabilities
      && selectedMeta.capabilities.voice_selection
      && voices.some((voice) => (
        voice.id === jobPrefs.voiceId
        && voice.language === jobPrefs.target
        && voice.installed !== false
      ))
      ? jobPrefs.voiceId
      : "";

    return { ...jobPrefs, profileId: advertisedProfile, voiceId: exactVoice };
  }

  function beginTranslation(video) {
    const s = state.get(video);
    if (!s || s.busy || s.on) return;
    if (s.jobAbort) s.jobAbort.abort();
    const jobAbort = new AbortController();
    s.jobAbort = jobAbort;
    clearError(video);
    // Новый (в том числе повторный) запуск не должен оставлять рядом открытые
    // ползунки настроек. Прогресс новой задачи отражается только в кнопке.
    closeLangPanel(s.wrapper, s.chip, false);
    s.busy = true;
    s.cancelled = false;
    s.jobId = null;
    // Все настройки фиксируются на весь job: другая вкладка/кнопка не должна
    // менять модель, язык или голос после асинхронного поиска медиапотока.
    s.jobPrefs = snapshotPrefs(video);
    s.server = s.jobPrefs.server;
    s.routeSelect.disabled = true;
    s.cancelButton.hidden = false;
    s.cancelButton.disabled = false;
    renderJobStatus(video, { status: "queued", stage: "queue", progress: 0 });
    translate(video, jobAbort.signal).finally(() => {
      const current = state.get(video);
      if (!current) return;
      if (current.jobAbort === jobAbort) current.jobAbort = null;
      current.busy = false;
      current.routeSelect.disabled = false;
      current.cancelButton.hidden = true;
      current.cancelButton.disabled = false;
      if (!current.on && !current.errorPanel) setIdleButton(video);
    });
  }

  async function translate(video, signal) {
    const s = state.get(video);
    const jobPrefs = s.jobPrefs || snapshotPrefs(video);
    try {
      const [candidates, resolvedPrefs] = await Promise.all([
        ensureCandidates(video),
        jobPrefs.settingsMode === "override"
          ? resolveJobPrefs(jobPrefs, signal)
          : Promise.resolve(jobPrefs),
      ]);
      if (s.cancelled) return;
      const requestBody = {
        page_url: location.href,
        media_url: mediaUrlOf(video),
        media_candidates: candidates,
        // длительность из плеера — сервер отбрасывает потоки-превью
        duration_hint: Number.isFinite(video.duration) ? video.duration : null,
        settings_mode: resolvedPrefs.settingsMode,
      };
      if (resolvedPrefs.settingsMode === "override") {
        Object.assign(requestBody, {
          source_lang: resolvedPrefs.source,
          target_lang: resolvedPrefs.target,
          voice_gender: resolvedPrefs.voice,
          voice_id: resolvedPrefs.voiceId || null,
          profile_id: resolvedPrefs.profileId || null,
        });
      }
      const job = await api("/dub", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal,
      }, s.server);
      if (s.cancelled) {
        if (job.id) await api("/job/" + job.id + "/cancel", { method: "POST" }, s.server).catch(() => {});
        return;
      }
      if (!job.id) throw new Error("сервер UVT не вернул идентификатор задачи");
      if (job.mode && job.mode !== "batch") {
        throw new Error("этот userscript ожидает пакетный сервер UVT, а не live-режим");
      }
      s.jobId = job.id;
      renderJobStatus(video, job);
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1250));
        if (s.cancelled) return;
        const info = await api("/job/" + job.id, { signal }, s.server);
        if (info.status === "done") {
          if (!info.audio_url) throw new Error("сервер отметил задачу готовой, но не отдал аудиодорожку");
          attachAudio(video, info.audio_url, info.entries, s.server);
          setEnabledButton(video);
          return;
        }
        if (info.status === "cancelled") {
          return;
        }
        if (info.status === "error") throw new Error(info.detail || "ошибка сервера");
        renderJobStatus(video, info);
      }
    } catch (err) {
      if (s.cancelled) return;
      console.warn("[UVT]", err);
      setRetryButton(video);
      showError(video, err);
    }
  }

  async function cancelJob(video) {
    const s = state.get(video);
    if (!s || !s.busy) return;
    s.cancelled = true;
    s.cancelButton.disabled = true;
    if (s.jobAbort) s.jobAbort.abort();
    if (s.jobId) {
      try {
        await api("/job/" + s.jobId + "/cancel", { method: "POST" }, s.server);
      } catch (_) { /* сервер мог уже завершить задачу */ }
    }
  }

  // --- выбор языков ---

  function syncRouteControls(route, keepPanelFor = null) {
    for (const wrapper of document.querySelectorAll(".uvt-wrap")) {
      const video = wrapper.__uvtVideo;
      const s = video && state.get(video);
      if (!s) continue;
      s.routeSelect.value = route;
      const panelRoute = s.settingsPanel
        && s.settingsPanel.querySelector(".uvt-panel-route");
      if (panelRoute) panelRoute.value = route;
      if (wrapper !== keepPanelFor && s.settingsPanel && s.settingsPanel.isConnected) {
        closeLangPanel(wrapper, s.chip, false);
      }
    }
  }

  function syncChipLabels() {
    for (const wrapper of document.querySelectorAll(".uvt-wrap")) {
      const video = wrapper.__uvtVideo;
      const s = video && state.get(video);
      if (!s || !s.chip) continue;
      s.chip.textContent = chipLabel(video);
      s.chip.setAttribute(
        "aria-label",
        `Настройки пакетного перевода: ${chipLabel(video)}`
      );
    }
  }

  function chipLabel(video) {
    const videoState = video && state.get(video);
    return videoState && videoState.settingsMode === "override"
      ? prefs.source + " → " + prefs.target
      : "Настройки панели";
  }

  function makeRouteSelect(onChange, id) {
    const select = document.createElement("select");
    if (id) select.id = id;
    Object.assign(select.style, CHIP_STYLE, {
      padding: "4px 7px",
      maxWidth: "150px",
      textOverflow: "ellipsis",
      appearance: "auto",
    });
    for (const [value, server] of Object.entries(SERVERS)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = server.shortLabel;
      option.title = server.label;
      if (value === prefs.route) option.selected = true;
      select.appendChild(option);
    }
    select.title = "Выберите тип перевода перед запуском";
    select.setAttribute("aria-label", "Модель перевода");
    select.addEventListener("change", () => onChange(select.value));
    select.addEventListener("click", (event) => event.stopPropagation());
    return select;
  }

  function makeSelect(current, withAuto, onChange, id) {
    const select = document.createElement("select");
    if (id) select.id = id;
    Object.assign(select.style, {
      font: "12px -apple-system, system-ui, sans-serif",
      background: "#222",
      color: "#fff",
      border: "1px solid #555",
      borderRadius: "6px",
      padding: "2px 4px",
    });
    for (const code of Object.keys(LANG_NAMES)) {
      if (!withAuto && code === "auto") continue;
      const option = document.createElement("option");
      option.value = code;
      option.textContent = code + " — " + LANG_NAMES[code];
      if (code === current) option.selected = true;
      select.appendChild(option);
    }
    select.addEventListener("change", () => onChange(select.value));
    select.addEventListener("click", (e) => e.stopPropagation());
    return select;
  }

  function makeSlider(min, max, step, value, onInput, id) {
    const input = document.createElement("input");
    input.type = "range";
    if (id) input.id = id;
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.style.width = "100%";
    input.style.minWidth = "0";
    input.addEventListener("input", () => onInput(parseFloat(input.value)));
    for (const event of ["click", "mousedown", "mousemove"]) {
      input.addEventListener(event, (e) => e.stopPropagation());
    }
    return input;
  }

  function cleanupSettingsState(wrapper) {
    const s = wrapper && state.get(wrapper.__uvtVideo);
    if (!s) return;
    if (s.settingsAbort) s.settingsAbort.abort();
    s.settingsAbort = null;
    if (s.settingsTimer) clearTimeout(s.settingsTimer);
    s.settingsTimer = null;
    if (s.previewAbort) s.previewAbort.abort();
    s.previewAbort = null;
    if (s.previewAudio) {
      s.previewAudio.pause();
      s.previewAudio.removeAttribute("src");
    }
    s.previewAudio = null;
    if (s.previewUrl) URL.revokeObjectURL(s.previewUrl);
    s.previewUrl = "";
    if (s.settingsPanel) s.settingsPanel.remove();
    s.settingsPanel = null;
  }

  function closeLangPanel(wrapper, chip, focusChip) {
    cleanupSettingsState(wrapper);
    chip.removeAttribute("aria-controls");
    chip.setAttribute("aria-expanded", "false");
    if (focusChip) chip.focus();
  }

  function toggleLangPanel(wrapper, chip, video) {
    const currentState = state.get(video);
    const existing = currentState && currentState.settingsPanel;
    if (existing) {
      closeLangPanel(wrapper, chip, false);
      return;
    }

    for (const other of document.querySelectorAll(".uvt-wrap")) {
      if (other === wrapper) continue;
      const otherVideo = other.__uvtVideo;
      const otherState = otherVideo && state.get(otherVideo);
      if (otherState && otherState.settingsPanel) {
        closeLangPanel(other, otherState.chip, false);
      }
    }

    const s = state.get(video);
    if (!s) return;
    cleanupSettingsState(wrapper);
    s.settingsAbort = new AbortController();

    const panel = document.createElement("section");
    panel.className = "uvt-panel";
    panel.id = nextControlId("settings");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Настройки пакетного перевода UVT");
    panel.tabIndex = -1;
    chip.setAttribute("aria-controls", panel.id);
    chip.setAttribute("aria-expanded", "true");
    const panelOffset = 38;
    Object.assign(panel.style, {
      // A fixed portal is not clipped by a site's overflow:hidden player and
      // can be clamped to the actual viewport even for edge-aligned embeds.
      position: "fixed",
      top: "8px",
      bottom: "auto",
      left: "8px",
      display: "grid",
      gridTemplateColumns: "minmax(78px, auto) minmax(0, 1fr)",
      gap: "8px",
      alignItems: "center",
      width: "min(460px, calc(100vw - 16px))",
      maxHeight: "min(620px, calc(100vh - 16px))",
      overflowY: "auto",
      boxSizing: "border-box",
      padding: "11px",
      background: "rgba(15, 15, 15, 0.97)",
      border: "1px solid rgba(255,255,255,0.32)",
      borderRadius: "10px",
      color: "#ddd",
      font: "12px -apple-system, system-ui, sans-serif",
      zIndex: "2147483647",
      whiteSpace: "normal",
      boxShadow: "0 8px 26px rgba(0,0,0,.38)",
    });

    const positionPanel = () => {
      if (!panel.isConnected && s.settingsPanel !== panel) return;
      const wrapperRect = wrapper.getBoundingClientRect();
      const panelWidth = Math.min(460, Math.max(160, window.innerWidth - 16));
      const desiredLeft = wrapperRect.left + wrapperRect.width / 2 - panelWidth / 2;
      const maxLeft = Math.max(8, window.innerWidth - panelWidth - 8);
      const clampedLeft = Math.max(8, Math.min(desiredLeft, maxLeft));
      const belowTop = wrapperRect.top + panelOffset;
      const aboveBottom = wrapperRect.top - 8;
      const spaceBelow = window.innerHeight - belowTop - 8;
      const spaceAbove = aboveBottom - 8;
      const placeAbove = spaceBelow < 260 && spaceAbove > spaceBelow;
      const available = Math.max(
        80,
        Math.floor(Math.min(window.innerHeight * 0.72, placeAbove ? spaceAbove : spaceBelow))
      );
      panel.style.width = `${panelWidth}px`;
      panel.style.left = `${clampedLeft}px`;
      panel.style.maxHeight = `${Math.min(620, available)}px`;
      if (placeAbove) {
        panel.style.top = "auto";
        panel.style.bottom = `${Math.max(8, window.innerHeight - aboveBottom)}px`;
      } else {
        panel.style.top = `${Math.max(8, belowTop)}px`;
        panel.style.bottom = "auto";
      }
    };
    window.addEventListener("resize", positionPanel, { signal: s.settingsAbort.signal });
    window.addEventListener("scroll", positionPanel, {
      capture: true,
      passive: true,
      signal: s.settingsAbort.signal,
    });

    const selectStyle = {
      width: "100%",
      minHeight: "30px",
      font: "12px -apple-system, system-ui, sans-serif",
      background: "#222",
      color: "#fff",
      border: "1px solid #555",
      borderRadius: "6px",
      padding: "3px 5px",
    };
    const sectionTitle = (text) => {
      const heading = document.createElement("strong");
      heading.textContent = text;
      Object.assign(heading.style, {
        gridColumn: "1 / -1",
        color: "#fff",
        borderTop: "1px solid rgba(255,255,255,.14)",
        paddingTop: "7px",
        marginTop: "2px",
      });
      panel.appendChild(heading);
    };
    const appendLabel = (text, control) => {
      const label = document.createElement("label");
      label.htmlFor = control.id;
      label.textContent = text;
      panel.appendChild(label);
      panel.appendChild(control);
    };
    const refreshChip = () => syncChipLabels();

    let voiceCatalog = [...LOCAL_VOICES];
    let metaSequence = 0;
    let previewAllowed = false;
    let localSourceRestricted = false;
    let localTargetLanguages = null;

    sectionTitle("Источник настроек");
    const settingsModeId = nextControlId("settings-mode");
    const settingsModeSelect = document.createElement("select");
    settingsModeSelect.id = settingsModeId;
    Object.assign(settingsModeSelect.style, selectStyle);
    for (const [value, label] of [
      ["server", "Web-панель сервера (по умолчанию)"],
      ["override", "Свои настройки для этого видео"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      settingsModeSelect.appendChild(option);
    }
    settingsModeSelect.value = s.settingsMode === "override" ? "override" : "server";
    appendLabel("Использовать:", settingsModeSelect);

    const settingsModeStatus = document.createElement("div");
    settingsModeStatus.setAttribute("role", "status");
    settingsModeStatus.setAttribute("aria-live", "polite");
    Object.assign(settingsModeStatus.style, {
      gridColumn: "1 / -1",
      padding: "7px 8px",
      borderRadius: "7px",
      background: "rgba(75, 130, 210, .16)",
      color: "#dbeafe",
    });
    panel.appendChild(settingsModeStatus);

    const settingsModeActions = document.createElement("div");
    Object.assign(settingsModeActions.style, {
      gridColumn: "1 / -1",
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
    });
    const useDashboardDefaults = document.createElement("button");
    useDashboardDefaults.type = "button";
    setButton(
      useDashboardDefaults,
      "↩ Вернуться к настройкам web-панели",
      "rgba(35, 95, 155, .9)",
      "Отменить настройки этого видео и снова использовать настройки web-панели"
    );
    Object.assign(useDashboardDefaults.style, CHIP_STYLE, {
      minHeight: "32px",
      padding: "4px 8px",
    });
    const openDashboard = document.createElement("button");
    openDashboard.type = "button";
    setButton(
      openDashboard,
      "Открыть web-панель",
      "rgba(255,255,255,.12)",
      "Открыть web-панель настроек текущего UVT-маршрута"
    );
    Object.assign(openDashboard.style, CHIP_STYLE, {
      minHeight: "32px",
      padding: "4px 8px",
    });
    settingsModeActions.append(useDashboardDefaults, openDashboard);
    panel.appendChild(settingsModeActions);

    const setSettingsMode = (mode, { reloadMeta = true } = {}) => {
      s.settingsMode = mode === "override" ? "override" : "server";
      settingsModeSelect.value = s.settingsMode;
      useDashboardDefaults.disabled = s.settingsMode === "server";
      settingsModeStatus.textContent = s.settingsMode === "server"
        ? "Следующий перевод возьмёт модель, языки и голос из web-панели. Значения ниже не отправляются серверу."
        : "Для следующего перевода этого видео будут отправлены выбранные ниже язык, локальный профиль и голос.";
      syncChipLabels();
      if (reloadMeta) refreshMeta();
    };
    const activateManualOverride = () => {
      if (s.settingsMode !== "override") setSettingsMode("override", { reloadMeta: false });
    };
    settingsModeSelect.addEventListener("change", () => {
      stopPreviewAudio();
      setSettingsMode(settingsModeSelect.value);
    });
    useDashboardDefaults.addEventListener("click", () => {
      stopPreviewAudio();
      setSettingsMode("server");
    });
    openDashboard.addEventListener("click", () => {
      window.open(serverForRoute(prefs.route).url, "_blank", "noopener,noreferrer");
    });
    setSettingsMode(s.settingsMode, { reloadMeta: false });

    sectionTitle("Перевод и модели");
    const panelRouteId = nextControlId("settings-route");
    const panelRoute = makeRouteSelect((value) => {
      stopPreviewAudio();
      prefs.route = value;
      localSourceRestricted = false;
      localTargetLanguages = null;
      refreshLanguageOptions();
      syncRouteControls(value, wrapper);
      profileSelect.disabled = value !== "free";
      serverUrlInput.value = serverForRoute(value).url;
      refreshMeta();
    }, panelRouteId);
    panelRoute.classList.add("uvt-panel-route");
    Object.assign(panelRoute.style, selectStyle);
    appendLabel("Маршрут:", panelRoute);

    const sourceId = nextControlId("source");
    const sourceSelect = makeSelect(
      prefs.source,
      true,
      (value) => {
        if (
          localSourceRestricted && value !== "auto"
          && !LOCAL_PIPELINE_LANGUAGES.has(value)
        ) {
          sourceSelect.value = prefs.source;
          readiness.textContent = "Этот локальный профиль принимает 25 языков Parakeet; выберите auto или доступный язык.";
          return;
        }
        prefs.source = value;
        activateManualOverride();
        refreshChip();
        refreshMeta();
      },
      sourceId
    );
    Object.assign(sourceSelect.style, selectStyle);
    appendLabel("С какого:", sourceSelect);

    const targetId = nextControlId("target");
    const targetSelect = makeSelect(
      prefs.target,
      false,
      (value) => {
        if (localTargetLanguages && !localTargetLanguages.has(value)) {
          targetSelect.value = prefs.target;
          readiness.textContent = "Для локального Piper доступны только установленные целевые языки; выберите RU/UK или облачный маршрут.";
          return;
        }
        stopPreviewAudio();
        prefs.target = value;
        activateManualOverride();
        refreshChip();
        refreshVoiceOptions();
        refreshMeta();
      },
      targetId
    );
    Object.assign(targetSelect.style, selectStyle);
    appendLabel("На какой:", targetSelect);

    function refreshLanguageOptions() {
      for (const option of sourceSelect.options) {
        option.disabled = localSourceRestricted
          && option.value !== "auto"
          && !LOCAL_PIPELINE_LANGUAGES.has(option.value);
      }
      for (const option of targetSelect.options) {
        option.disabled = !!localTargetLanguages
          && !localTargetLanguages.has(option.value);
      }
    }
    refreshLanguageOptions();

    const profileId = nextControlId("local-profile");
    const profileSelect = document.createElement("select");
    profileSelect.id = profileId;
    Object.assign(profileSelect.style, selectStyle);
    const renderProfileOptions = (profiles, selectedProfile) => {
      const catalog = profiles.length
        ? profiles
        : Object.entries(LOCAL_PROFILES).map(([id, info]) => ({ id, ...info }));
      profileSelect.replaceChildren();
      for (const item of catalog) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label || item.id;
        if (item.installed === false) {
          option.textContent += " · не установлен";
          option.disabled = true;
        }
        const engines = item.engines || {};
        option.title = item.detail
          || [engines.stt, engines.translation, engines.tts].filter(Boolean).join(" → ");
        profileSelect.appendChild(option);
      }
      profileSelect.value = selectedProfile;
    };
    renderProfileOptions([], prefs.localProfile || "local-balanced");
    profileSelect.disabled = prefs.route !== "free";
    profileSelect.addEventListener("change", () => {
      stopPreviewAudio();
      prefs.localProfile = profileSelect.value;
      activateManualOverride();
      localSourceRestricted = false;
      localTargetLanguages = null;
      refreshLanguageOptions();
      refreshMeta();
    });
    appendLabel("Локально:", profileSelect);

    const readiness = document.createElement("div");
    readiness.setAttribute("role", "status");
    readiness.setAttribute("aria-live", "polite");
    Object.assign(readiness.style, {
      gridColumn: "1 / -1",
      padding: "7px 8px",
      borderRadius: "7px",
      background: "rgba(255,255,255,.07)",
      color: "#cbd5e1",
      overflowWrap: "anywhere",
    });
    readiness.textContent = "Проверяю сервер и модели…";
    panel.appendChild(readiness);

    sectionTitle("Голос");
    const voiceModeId = nextControlId("voice-mode");
    const voiceSelect = document.createElement("select");
    voiceSelect.id = voiceModeId;
    Object.assign(voiceSelect.style, selectStyle);
    for (const [value, label] of [
      ["auto", "Авто по каждой реплике"],
      ["male", "Мужской"],
      ["female", "Женский"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = value === prefs.voice;
      voiceSelect.appendChild(option);
    }
    voiceSelect.addEventListener("change", () => {
      stopPreviewAudio();
      prefs.voice = voiceSelect.value;
      prefs.voiceId = "";
      voiceModelSelect.value = "";
      activateManualOverride();
    });
    appendLabel("Режим:", voiceSelect);

    const voiceModelId = nextControlId("voice-model");
    const voiceModelSelect = document.createElement("select");
    voiceModelSelect.id = voiceModelId;
    Object.assign(voiceModelSelect.style, selectStyle);
    voiceModelSelect.addEventListener("change", () => {
      stopPreviewAudio();
      prefs.voiceId = voiceModelSelect.value;
      activateManualOverride();
      const selected = voiceCatalog.find((voice) => voice.id === prefs.voiceId);
      if (selected) {
        prefs.voice = selected.gender;
        voiceSelect.value = selected.gender;
      }
    });
    appendLabel("Конкретный:", voiceModelSelect);

    function refreshVoiceOptions({ clearInvalid = false } = {}) {
      const compatible = voiceCatalog.filter(
        (voice) => voice.language === prefs.target && voice.installed !== false
      );
      const previous = prefs.voiceId;
      voiceModelSelect.replaceChildren();
      const automatic = document.createElement("option");
      automatic.value = "";
      automatic.textContent = "По режиму выше";
      voiceModelSelect.appendChild(automatic);
      for (const voice of compatible) {
        const option = document.createElement("option");
        option.value = voice.id;
        option.textContent = `${voice.label} · ${voice.gender === "female" ? "женский" : "мужской"}`;
        voiceModelSelect.appendChild(option);
      }
      const stillCompatible = compatible.some((voice) => voice.id === previous);
      // LOCAL_VOICES is only an offline fallback.  A custom Piper voice stays
      // pending until a successful Free /meta response authoritatively says it
      // is unavailable; switching to GPT/ElevenLabs must not erase it.
      if (clearInvalid && !stillCompatible) prefs.voiceId = "";
      voiceModelSelect.value = stillCompatible ? previous : "";
      voiceModelSelect.disabled = prefs.route !== "free" || compatible.length === 0;
    }
    refreshVoiceOptions();

    const duckId = nextControlId("original-volume");
    const duckLabel = document.createElement("label");
    duckLabel.htmlFor = duckId;
    const refreshDuckLabel = () => {
      duckLabel.textContent = `Оригинал: ${Math.round(prefs.duck * 100)}%`;
    };
    refreshDuckLabel();
    duckLabel.title = "Громкость основной дорожки, пока звучит перевод";
    panel.appendChild(duckLabel);
    panel.appendChild(makeSlider(0, 0.6, 0.05, prefs.duck, (value) => {
      prefs.duck = value;
      refreshDuckLabel();
    }, duckId));

    const volumeId = nextControlId("translation-volume");
    const volLabel = document.createElement("label");
    volLabel.htmlFor = volumeId;
    const refreshVolLabel = () => {
      volLabel.textContent = `Перевод: ${Math.round(prefs.voiceVol * 100)}%`;
    };
    refreshVolLabel();
    panel.appendChild(volLabel);
    panel.appendChild(makeSlider(0.2, 1, 0.05, prefs.voiceVol, (value) => {
      prefs.voiceVol = value;
      refreshVolLabel();
      if (s.audio) s.audio.volume = value;
    }, volumeId));

    sectionTitle("Проба локального голоса");
    const previewText = document.createElement("textarea");
    previewText.id = nextControlId("preview-text");
    previewText.maxLength = 240;
    previewText.rows = 2;
    previewText.value = prefs.target === "uk"
      ? "Привіт! Так звучатиме локальний переклад UVT."
      : "Здравствуйте! Так будет звучать локальный перевод UVT.";
    Object.assign(previewText.style, {
      ...selectStyle,
      gridColumn: "1 / -1",
      resize: "vertical",
      minHeight: "52px",
      boxSizing: "border-box",
    });
    previewText.setAttribute("aria-label", "Текст для пробы голоса");
    panel.appendChild(previewText);

    const previewActions = document.createElement("div");
    Object.assign(previewActions.style, {
      gridColumn: "1 / -1",
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
    });
    const malePreview = document.createElement("button");
    const femalePreview = document.createElement("button");
    const stopPreview = document.createElement("button");
    for (const button of [malePreview, femalePreview, stopPreview]) {
      button.type = "button";
      Object.assign(button.style, CHIP_STYLE, { minHeight: "32px", padding: "4px 8px" });
      previewActions.appendChild(button);
    }
    setButton(malePreview, "▶ Мужской", "rgba(30, 100, 150, .9)", "Прослушать мужской локальный голос");
    setButton(femalePreview, "▶ Женский", "rgba(125, 55, 130, .9)", "Прослушать женский локальный голос");
    setButton(stopPreview, "■ Стоп", "rgba(255,255,255,.12)", "Отменить подготовку или остановить пробу голоса");
    panel.appendChild(previewActions);

    const previewStatus = document.createElement("div");
    previewStatus.setAttribute("role", "status");
    previewStatus.setAttribute("aria-live", "polite");
    Object.assign(previewStatus.style, { gridColumn: "1 / -1", color: "#aeb6c2" });
    previewStatus.textContent = "Введите короткую фразу и выберите пример.";
    panel.appendChild(previewStatus);

    const applyPreviewButtonState = () => {
      const compatible = voiceCatalog.filter(
        (voice) => voice.language === prefs.target && voice.installed !== false
      );
      malePreview.disabled = !previewAllowed
        || !compatible.some((voice) => voice.gender === "male");
      femalePreview.disabled = !previewAllowed
        || !compatible.some((voice) => voice.gender === "female");
    };

    let previewSequence = 0;
    const stopPreviewAudio = () => {
      previewSequence += 1;
      if (s.previewAbort) s.previewAbort.abort();
      s.previewAbort = null;
      if (s.previewAudio) s.previewAudio.pause();
      s.previewAudio = null;
      if (s.previewUrl) URL.revokeObjectURL(s.previewUrl);
      s.previewUrl = "";
    };
    const playPreview = async (gender) => {
      stopPreviewAudio();
      const sequence = previewSequence;
      if (!previewAllowed || prefs.route !== "free") {
        previewStatus.textContent = "Проба доступна на локальном маршруте Free с обновлённым сервером.";
        return;
      }
      const text = previewText.value.trim();
      if (!text) {
        previewStatus.textContent = "Введите текст для пробы голоса.";
        previewText.focus();
        return;
      }
      previewStatus.textContent = "Готовлю локальный пример…";
      malePreview.disabled = true;
      femalePreview.disabled = true;
      const selectedVoice = voiceCatalog.find(
        (voice) => voice.id === prefs.voiceId && voice.gender === gender
      );
      const previewAbort = new AbortController();
      s.previewAbort = previewAbort;
      const previewPrefs = {
        route: "free",
        server: serverForRoute("free"),
        settingsMode: "override",
        source: prefs.source,
        target: prefs.target,
        voice: gender,
        voiceId: selectedVoice ? selectedVoice.id : "",
        profileId: prefs.localProfile,
      };
      try {
        const resolvedPreview = await resolveJobPrefs(previewPrefs, previewAbort.signal);
        const blob = await apiAudio("/tts/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            settings_mode: "override",
            target_lang: resolvedPreview.target,
            profile_id: resolvedPreview.profileId || null,
            voice_gender: gender,
            voice_id: resolvedPreview.voiceId || null,
          }),
          signal: previewAbort.signal,
        }, resolvedPreview.server);
        if (!panel.isConnected || sequence !== previewSequence) return;
        s.previewUrl = URL.createObjectURL(blob);
        s.previewAudio = new Audio(s.previewUrl);
        s.previewAudio.volume = prefs.voiceVol;
        s.previewAudio.addEventListener("ended", () => {
          if (sequence !== previewSequence) return;
          previewStatus.textContent = "Пример завершён.";
          stopPreviewAudio();
        }, { once: true });
        await s.previewAudio.play();
        previewStatus.textContent = gender === "female"
          ? "Играет женский голос." : "Играет мужской голос.";
      } catch (error) {
        if (error && error.name === "AbortError") return;
        if (sequence !== previewSequence) return;
        previewStatus.textContent = String(error && error.message ? error.message : error);
      } finally {
        if (sequence === previewSequence) s.previewAbort = null;
        if (panel.isConnected && sequence === previewSequence) {
          applyPreviewButtonState();
        }
      }
    };
    malePreview.addEventListener("click", () => playPreview("male"));
    femalePreview.addEventListener("click", () => playPreview("female"));
    stopPreview.addEventListener("click", () => {
      stopPreviewAudio();
      applyPreviewButtonState();
      previewStatus.textContent = "Проба остановлена.";
    });

    sectionTitle("Подключение");
    const connection = document.createElement("details");
    connection.style.gridColumn = "1 / -1";
    const connectionSummary = document.createElement("summary");
    connectionSummary.textContent = "Адрес текущего маршрута";
    connectionSummary.style.cursor = "pointer";
    connection.appendChild(connectionSummary);
    const serverUrlInput = document.createElement("input");
    serverUrlInput.type = "url";
    serverUrlInput.value = serverForRoute(prefs.route).url;
    serverUrlInput.setAttribute("aria-label", "Адрес UVT-сервера текущего маршрута");
    Object.assign(serverUrlInput.style, { ...selectStyle, marginTop: "7px" });
    connection.appendChild(serverUrlInput);
    const connectionActions = document.createElement("div");
    Object.assign(connectionActions.style, { display: "flex", gap: "6px", marginTop: "6px" });
    const saveServer = document.createElement("button");
    saveServer.type = "button";
    setButton(saveServer, "Сохранить и проверить", "rgba(30, 105, 65, .9)", "Сохранить адрес и проверить UVT-сервер");
    Object.assign(saveServer.style, CHIP_STYLE);
    connectionActions.appendChild(saveServer);
    connection.appendChild(connectionActions);
    const connectionHint = document.createElement("div");
    connectionHint.textContent = "Для сети используйте HTTPS reverse proxy и UVT_API_TOKEN; секрет в эту панель не вводится.";
    Object.assign(connectionHint.style, { color: "#aeb6c2", marginTop: "5px" });
    connection.appendChild(connectionHint);
    panel.appendChild(connection);
    saveServer.addEventListener("click", () => {
      try {
        stopPreviewAudio();
        const parsed = new URL(serverUrlInput.value.trim());
        if (!/^https?:$/.test(parsed.protocol)) throw new Error("нужен адрес http:// или https://");
        const loopback = ["localhost", "127.0.0.1", "::1"].includes(parsed.hostname);
        if (!loopback && parsed.protocol !== "https:") {
          throw new Error("сетевой сервер должен открываться по HTTPS");
        }
        writePref(`uvt.server.${prefs.route}`, parsed.toString());
        refreshMeta();
      } catch (error) {
        readiness.textContent = String(error && error.message ? error.message : error);
      }
    });

    const hint = document.createElement("div");
    hint.textContent = "По умолчанию модель, язык и голос берутся из web-панели. Изменение поля выше включает переопределение только для этого видео; громкости всегда остаются настройкой браузера.";
    Object.assign(hint.style, { gridColumn: "1 / -1", color: "#aeb6c2" });
    panel.appendChild(hint);

    async function refreshMeta() {
      const sequence = ++metaSequence;
      if (s.settingsTimer) clearTimeout(s.settingsTimer);
      if (readiness.textContent !== "Проверяю сервер и модели…") {
        readiness.textContent = "Проверяю сервер и модели…";
      }
      previewAllowed = false;
      applyPreviewButtonState();
      try {
        const activeRoute = prefs.route;
        const activeServer = serverForRoute(activeRoute);
        const baseMeta = await api("/meta", { signal: s.settingsAbort.signal }, activeServer);
        if (!panel.isConnected || sequence !== metaSequence) return;

        const usingServerSettings = s.settingsMode !== "override";
        const baseProfiles = Array.isArray(baseMeta.profiles) ? baseMeta.profiles : [];
        const canSelectProfile = activeRoute === "free"
          && !!(baseMeta.capabilities && baseMeta.capabilities.profile_selection);
        const preferred = baseProfiles.find(
          (item) => item.id === prefs.localProfile && item.installed !== false
        );
        const defaultProfile = (baseMeta.defaults && baseMeta.defaults.profile_id)
          || (baseMeta.profile && baseMeta.profile.name)
          || "";
        const selectedProfile = usingServerSettings
          ? defaultProfile
          : canSelectProfile && preferred ? preferred.id : defaultProfile;

        let meta = baseMeta;
        const baseProfile = baseMeta.profile && baseMeta.profile.name;
        const baseTarget = baseMeta.profile && baseMeta.profile.target_lang;
        if (!usingServerSettings && (
          (canSelectProfile && selectedProfile !== baseProfile) || prefs.target !== baseTarget
        )) {
          const query = new URLSearchParams({ target_lang: prefs.target });
          if (canSelectProfile && selectedProfile) query.set("profile_id", selectedProfile);
          meta = await api(`/meta?${query.toString()}`, { signal: s.settingsAbort.signal }, activeServer);
          if (!panel.isConnected || sequence !== metaSequence) return;
        }

        const engines = meta.profile && meta.profile.engines ? meta.profile.engines : {};
        const profileInfo = (meta.profiles || []).find((item) => item.id === selectedProfile);
        const engineInfo = profileInfo && profileInfo.engines ? profileInfo.engines : engines;
        localSourceRestricted = activeRoute === "free" && (
          ["local-fast", "local-balanced", "local-quality"].includes(selectedProfile)
          || engineInfo.stt === "parakeet-mlx"
          || engineInfo.translation === "translategemma-mlx"
        );
        const advertisedTargets = meta.limits
          && Array.isArray(meta.limits.local_tts_languages)
          ? meta.limits.local_tts_languages : [];
        localTargetLanguages = activeRoute === "free" && engineInfo.tts === "piper"
          ? new Set(advertisedTargets) : null;
        if (!usingServerSettings &&
          localSourceRestricted && prefs.source !== "auto"
          && !LOCAL_PIPELINE_LANGUAGES.has(prefs.source)
        ) {
          prefs.source = "auto";
          sourceSelect.value = "auto";
          refreshChip();
        }
        if (!usingServerSettings && localTargetLanguages && !localTargetLanguages.has(prefs.target)) {
          const fallbackTarget = localTargetLanguages.has("ru")
            ? "ru" : [...localTargetLanguages][0];
          if (fallbackTarget) {
            prefs.target = fallbackTarget;
            targetSelect.value = fallbackTarget;
            refreshChip();
            refreshVoiceOptions();
            readiness.textContent = `Локальный Piper не озвучивает выбранный язык; переключено на ${fallbackTarget}.`;
            setTimeout(refreshMeta, 0);
            return;
          }
        }
        refreshLanguageOptions();
        const readinessInfo = meta.model_readiness || {};
        const readinessText = readinessInfo.detail || "сервер отвечает";
        const settingsLabel = usingServerSettings ? "Web-панель" : "Для этого видео";
        const statusText = `${settingsLabel} · ${profileInfo ? profileInfo.label : selectedProfile}: ${engineInfo.stt || "?"} → ${engineInfo.translation || "?"} → ${engineInfo.tts || "?"}. ${readinessText}`;
        if (readiness.textContent !== statusText) readiness.textContent = statusText;

        renderProfileOptions(baseProfiles, selectedProfile);
        profileSelect.disabled = !canSelectProfile;
        voiceCatalog = activeRoute === "free" && Array.isArray(meta.voices)
          ? meta.voices : [];
        refreshVoiceOptions({
          clearInvalid: !usingServerSettings && activeRoute === "free",
        });
        const compatible = voiceCatalog.filter(
          (voice) => voice.language === prefs.target && voice.installed !== false
        );
        const hasMale = compatible.some((voice) => voice.gender === "male");
        const hasFemale = compatible.some((voice) => voice.gender === "female");
        const automaticVoice = [...voiceSelect.options].find(
          (option) => option.value === "auto"
        );
        const piperNeedsBothVoices = activeRoute === "free"
          && engineInfo.tts === "piper" && (!hasMale || !hasFemale);
        if (automaticVoice) automaticVoice.disabled = piperNeedsBothVoices;
        if (!usingServerSettings && piperNeedsBothVoices && prefs.voice === "auto") {
          const availableGender = hasFemale ? "female" : hasMale ? "male" : "";
          if (availableGender) {
            prefs.voice = availableGender;
            voiceSelect.value = availableGender;
          }
        }
        previewAllowed = !!(
          meta.capabilities && meta.capabilities.tts_preview && activeRoute === "free"
        );
        applyPreviewButtonState();
        if (!previewAllowed) {
          previewStatus.textContent = activeRoute === "free"
            ? `Для языка ${prefs.target} нет установленного локального Piper-голоса.`
            : "Проба доступна только для локального Piper.";
        }

        if (["pending", "checking", "loading", "in-use", "idle"].includes(readinessInfo.status)) {
          s.settingsTimer = setTimeout(refreshMeta, 1600);
        }
      } catch (error) {
        if (error && error.name === "AbortError") return;
        readiness.textContent = String(error && error.message ? error.message : error);
      }
    }

    const close = document.createElement("button");
    close.type = "button";
    setButton(close, "Закрыть настройки", "transparent", "Закрыть настройки пакетного перевода");
    Object.assign(close.style, CHIP_STYLE, {
      gridColumn: "1 / -1",
      minHeight: "34px",
      justifySelf: "end",
    });
    close.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeLangPanel(wrapper, chip, true);
    });
    panel.appendChild(close);

    panel.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeLangPanel(wrapper, chip, true);
      }
    });
    for (const eventName of [
      "pointerdown", "pointerup", "mousedown", "mouseup",
      "click", "dblclick", "touchstart", "touchend",
    ]) {
      panel.addEventListener(eventName, (event) => event.stopPropagation(), {
        signal: s.settingsAbort.signal,
      });
    }
    document.body.appendChild(panel);
    s.settingsPanel = panel;
    positionPanel();
    refreshMeta();
    panelRoute.focus();
  }

  // --- кнопка на плеере ---

  function addButton(video) {
    if (state.has(video)) return;
    const parent = video.parentElement;
    if (!parent) return;
    if (getComputedStyle(parent).position === "static") parent.style.position = "relative";

    const wrapper = document.createElement("div");
    wrapper.className = "uvt-wrap";
    wrapper.__uvtVideo = video; // для уборки кнопок исчезнувших видео (реклама)
    Object.assign(wrapper.style, {
      position: "absolute",
      top: "10px",
      left: "50%",
      transform: "translateX(-50%)",
      zIndex: "2147483647",
      display: "flex",
      flexWrap: "wrap",
      justifyContent: "center",
      maxWidth: "calc(100vw - 12px)",
      gap: "6px",
      alignItems: "flex-start",
      opacity: "1",
      pointerEvents: "auto",
      transition: "opacity 0.25s ease",
    });

    const btn = document.createElement("button");
    btn.type = "button";
    Object.assign(btn.style, CHIP_STYLE);
    btn.setAttribute("aria-pressed", "false");

    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = chipLabel(video);
    Object.assign(chip.style, CHIP_STYLE);
    chip.title = "Языки, голос и громкости пакетного перевода";
    chip.setAttribute("aria-haspopup", "dialog");
    chip.setAttribute("aria-expanded", "false");
    chip.setAttribute("aria-label", `Настройки пакетного перевода: ${chipLabel(video)}`);

    const routeId = nextControlId("route");
    const routeSelect = makeRouteSelect((value) => {
      prefs.route = value;
      syncRouteControls(value);
    }, routeId);

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.textContent = "✕";
    Object.assign(cancelBtn.style, CHIP_STYLE);
    cancelBtn.hidden = true;
    cancelBtn.title = "Отменить подготовку пакетного перевода";
    cancelBtn.setAttribute("aria-label", "Отменить подготовку пакетного перевода");

    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      const s = state.get(video);
      if (s.on) {
        detachAudio(video);
        closeLangPanel(s.wrapper, s.chip, false);
        setIdleButton(video);
      } else if (!s.busy) {
        beginTranslation(video);
      }
    });

    cancelBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      cancelJob(video);
    });

    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      toggleLangPanel(wrapper, chip, video);
    });

    wrapper.appendChild(btn);
    wrapper.appendChild(routeSelect);
    wrapper.appendChild(chip);
    wrapper.appendChild(cancelBtn);
    parent.appendChild(wrapper);
    state.set(video, {
      wrapper,
      button: btn,
      routeSelect,
      chip,
      cancelButton: cancelBtn,
      on: false,
      busy: false,
      cancelled: false,
      jobId: null,
      jobPrefs: null,
      settingsMode: "server",
      server: null,
      errorPanel: null,
      settingsAbort: null,
      settingsTimer: null,
      previewAudio: null,
      previewUrl: "",
      previewAbort: null,
      jobAbort: null,
      settingsPanel: null,
      cleanupAutoHide: null,
    });
    setIdleButton(video);
    state.get(video).cleanupAutoHide = setupAutoHide(video, wrapper);
  }

  function setupAutoHide(video, wrapper) {
    const container = video.parentElement;
    const abortController = new AbortController();
    const listenerOptions = { signal: abortController.signal };
    let timer = null;
    let keyboardFocus = false;
    let pointerOverControls = false;

    const hasKeyboardFocus = () => {
      try {
        // Pointer click leaves :focus, but only keyboard navigation normally
        // gets :focus-visible. This is the reliable path in modern browsers.
        return !!wrapper.querySelector(":focus-visible");
      } catch (_) {
        // Older engines have no :focus-visible selector; retain the modality
        // fallback rather than hiding a keyboard user's active control.
        return keyboardFocus && wrapper.contains(document.activeElement);
      }
    };

    // Обычный статус подготовки не держит слой на экране сам по себе. Ошибка,
    // открытые настройки и настоящий keyboard focus остаются видимыми, чтобы
    // причину сбоя и элементы управления нельзя было потерять без мыши.
    const mustStay = () => {
      const s = state.get(video);
      return !!(
        pointerOverControls ||
        (s && s.settingsPanel && s.settingsPanel.isConnected) ||
        (s && s.errorPanel) ||
        hasKeyboardFocus()
      );
    };
    const show = () => {
      wrapper.style.opacity = "1";
      wrapper.style.pointerEvents = "auto";
    };
    const hide = () => {
      if (mustStay()) {
        clearTimeout(timer);
        timer = setTimeout(hide, 1500); // проверим снова, когда панель закроют
        return;
      }
      // Click leaves native focus on a button. Blur only pointer focus before
      // dimming so the next Space/Enter cannot repeat a pointer action.
      const active = document.activeElement;
      if (active instanceof HTMLElement && wrapper.contains(active)) active.blur();
      // Keep the controls visibly present and hit-testable. YouTube can render
      // transparent overlays above <video>; disabling pointer events here made
      // the next click fall through to the player and pause the video.
      wrapper.style.opacity = "0.55";
      wrapper.style.pointerEvents = "auto";
    };
    const poke = () => {
      show();
      clearTimeout(timer);
      timer = setTimeout(hide, 2800);
    };

    // YouTube and similar players render transparent sibling overlays above
    // <video>. Capture pointer movement at document level so a hidden UVT layer
    // wakes even when the event never bubbles through video.parentElement.
    document.addEventListener("pointermove", (event) => {
      const rect = video.getBoundingClientRect();
      if (
        event.clientX >= rect.left && event.clientX <= rect.right &&
        event.clientY >= rect.top && event.clientY <= rect.bottom
      ) {
        poke();
      }
    }, { capture: true, passive: true, signal: abortController.signal });

    for (const el of [container, video]) {
      el.addEventListener("pointerenter", poke, listenerOptions);
    }
    wrapper.addEventListener("pointerenter", () => {
      pointerOverControls = true;
      poke();
    }, listenerOptions);
    wrapper.addEventListener("pointerleave", () => {
      pointerOverControls = false;
      clearTimeout(timer);
      timer = setTimeout(hide, 400);
    }, listenerOptions);

    // Do not let player-level pointer/mouse handlers pause the video while a
    // UVT control is being pressed. Individual controls still handle click.
    for (const eventName of [
      "pointerdown", "pointerup", "mousedown", "mouseup",
      "click", "dblclick", "touchstart", "touchend",
    ]) {
      wrapper.addEventListener(eventName, (event) => event.stopPropagation(), listenerOptions);
    }
    wrapper.addEventListener("focusin", () => {
      keyboardFocus = lastInteractionWasKeyboard;
      poke();
    }, listenerOptions);
    wrapper.addEventListener("focusout", (event) => {
      if (event.relatedTarget && wrapper.contains(event.relatedTarget)) return;
      keyboardFocus = false;
      clearTimeout(timer);
      timer = setTimeout(hide, 300);
    }, listenerOptions);
    container.addEventListener("mouseleave", () => {
      clearTimeout(timer);
      timer = setTimeout(hide, 400);
    }, listenerOptions);
    video.addEventListener("pause", poke, listenerOptions);
    video.addEventListener("play", poke, listenerOptions);
    poke();

    return () => {
      clearTimeout(timer);
      abortController.abort();
    };
  }

  // Видео исчезло/спрятано: удалено из DOM, схлопнуто, display:none,
  // visibility:hidden или полностью прозрачно (так сайты прячут рекламу)
  function videoGone(video) {
    if (!video || !video.isConnected) return true;
    if (video.offsetWidth === 0 || video.getClientRects().length === 0) return true;
    const style = getComputedStyle(video);
    if (style.visibility === "hidden" || style.display === "none") return true;
    if (parseFloat(style.opacity) === 0) return true;
    return false;
  }

  function rectsOverlap(a, b) {
    const w = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
    const h = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    const smaller = Math.min(a.width * a.height, b.width * b.height);
    return smaller > 0 && (w * h) / smaller > 0.7;
  }

  function livelinessScore(video) {
    return (video.paused ? 0 : 2) + (video.readyState >= 2 ? 1 : 0) + (video.currentTime > 0 ? 1 : 0);
  }

  function removeWrapperFor(video, wrapper) {
    const s = state.get(video);
    if (s) {
      s.cancelled = true; // остановить опрос задачи, если шёл
      if (s.jobAbort) s.jobAbort.abort();
      if (s.busy && s.jobId && s.server) {
        api(`/job/${s.jobId}/cancel`, { method: "POST" }, s.server).catch(() => {});
      }
      if (s.cleanupAutoHide) s.cleanupAutoHide();
    }
    cleanupSettingsState(wrapper);
    detachAudio(video);
    state.delete(video);
    wrapper.remove();
  }

  function scan() {
    // Уборка: кнопки видео, которых больше нет или которые спрятаны (реклама)
    for (const wrapper of document.querySelectorAll(".uvt-wrap")) {
      if (videoGone(wrapper.__uvtVideo)) {
        removeWrapperFor(wrapper.__uvtVideo, wrapper);
      }
    }

    // Дедупликация: из видео, перекрывающих друг друга в одном плеере
    // (контент + спрятанная реклама), кнопку получает только «живое»
    const visible = [...document.querySelectorAll("video")]
      .filter((v) => !videoGone(v) && v.offsetWidth >= MIN_VIDEO_WIDTH);
    const chosen = [];
    for (const video of visible) {
      const rect = video.getBoundingClientRect();
      const clash = chosen.findIndex((c) => rectsOverlap(rect, c.rect));
      if (clash === -1) {
        chosen.push({ video, rect });
      } else if (livelinessScore(video) > livelinessScore(chosen[clash].video)) {
        chosen[clash] = { video, rect };
      }
    }
    const keep = new Set(chosen.map((c) => c.video));
    for (const wrapper of document.querySelectorAll(".uvt-wrap")) {
      const video = wrapper.__uvtVideo;
      if (keep.has(video)) continue;
      const s = state.get(video);
      if (s && (s.on || s.busy)) continue; // активный перевод не трогаем
      removeWrapperFor(video, wrapper);
    }
    for (const video of keep) addButton(video);
  }

  scan();
  setInterval(scan, 2000);
})();
