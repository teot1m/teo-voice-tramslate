// ==UserScript==
// @name         UVT — закадровый перевод видео
// @namespace    uvt
// @version      0.9.5
// @description  Кнопка UVT на любом видео: перевод и замена голоса через локальный сервер uvt serve (аналог voice-over-translation, но со своим движком)
// @match        *://*/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  // --- настройки ---
  const SERVER = "http://127.0.0.1:8765"; // адрес uvt serve
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
    pt: "португальский", ja: "японский", zh: "китайский", ko: "корейский",
    tr: "турецкий", pl: "польский", ar: "арабский", hi: "хинди",
  };

  // Разовая миграция: до v0.8 голос по умолчанию был "male" — переводим на авто
  if (!localStorage.getItem("uvt.voice.v2")) {
    localStorage.setItem("uvt.voice.v2", "1");
    localStorage.setItem("uvt.voice", "auto");
  }

  const prefs = {
    get source() { return localStorage.getItem("uvt.source") || "auto"; },
    set source(v) { localStorage.setItem("uvt.source", v); },
    get target() { return localStorage.getItem("uvt.target") || "ru"; },
    set target(v) { localStorage.setItem("uvt.target", v); },
    get voice() { return localStorage.getItem("uvt.voice") || "auto"; },
    set voice(v) { localStorage.setItem("uvt.voice", v); },
    get duck() {
      const v = parseFloat(localStorage.getItem("uvt.duck"));
      return Number.isFinite(v) ? v : DEFAULT_DUCK;
    },
    set duck(v) { localStorage.setItem("uvt.duck", String(v)); },
    get voiceVol() {
      const v = parseFloat(localStorage.getItem("uvt.voiceVol"));
      return Number.isFinite(v) ? v : DEFAULT_VOICE_VOL;
    },
    set voiceVol(v) { localStorage.setItem("uvt.voiceVol", String(v)); },
  };

  const state = new WeakMap();

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
  };

  function setButton(btn, text, bg) {
    btn.textContent = text;
    if (bg) btn.style.background = bg;
  }

  async function api(path, options) {
    const response = await fetch(SERVER + path, options);
    if (!response.ok) throw new Error("сервер UVT: HTTP " + response.status);
    return response.json();
  }

  // --- окна реплик: когда приглушать оригинал ---
  function buildWindows(entries) {
    const raw = (entries || []).map((e) => {
      const spoken = Math.max(e.end - e.start, SPEECH_RATE_S * (e.translated || "").length);
      return [Math.max(0, e.start - WINDOW_LEAD_S), e.start + spoken + WINDOW_TAIL_S];
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
  // Основной путь — Web Audio GainNode: работает поверх плеера, даже если сайт
  // (YouTube) перетирает video.volume. Для источников, где MediaElementSource
  // заглушил бы звук (чужой origin без CORS), — резерв через video.volume.
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
        console.warn("[UVT] Web Audio недоступен, приглушаю через volume:", err);
      }
    }

    const fb = { base: video.volume, setting: false, ducked: false };
    const onVolumeChange = () => {
      if (fb.setting) return; // наша же правка
      fb.base = fb.ducked
        ? Math.min(1, video.volume / Math.max(prefs.duck, 0.05))
        : video.volume;
    };
    video.addEventListener("volumechange", onVolumeChange);
    console.info("[UVT] приглушение через video.volume (резервный режим)");
    return {
      mode: "volume",
      set(mult) {
        fb.ducked = mult < 1;
        const target = Math.min(1, fb.base * mult);
        if (Math.abs(video.volume - target) > 0.01) {
          fb.setting = true;
          video.volume = target;
          fb.setting = false;
        }
      },
      release() {
        video.removeEventListener("volumechange", onVolumeChange);
        fb.setting = true;
        video.volume = fb.base;
        fb.setting = false;
      },
    };
  }

  // --- синхронное воспроизведение перевода ---

  function attachAudio(video, audioUrl, entries) {
    const s = state.get(video);
    const audio = new Audio(SERVER + audioUrl);
    audio.preload = "auto";
    s.audio = audio;
    s.windows = buildWindows(entries);
    s.ducker = createDucker(video);

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
    s.on = false;
    clearInterval(s.timer);
    for (const [event, fn] of Object.entries(s.handlers || {})) video.removeEventListener(event, fn);
    s.audio.pause();
    s.audio.src = "";
    s.audio = null;
    if (s.ducker) { s.ducker.release(); s.ducker = null; }
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

  // Если плеер ещё не загрузил поток (видео не играли) — беззвучно «трогаем»
  // воспроизведение на пару секунд, чтобы манифест появился в ресурсах.
  async function ensureCandidates(video) {
    let found = findMediaCandidates();
    if (mediaUrlOf(video)) return found.list; // есть прямой src — этого хватит
    if (found.list.length && (found.fresh || lastNavigation === 0)) return found.list;

    const wasPaused = video.paused;
    const wasMuted = video.muted;
    try {
      video.muted = true;
      await video.play().catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } finally {
      if (wasPaused) video.pause();
      video.muted = wasMuted;
    }
    return findMediaCandidates().list;
  }

  // --- запуск перевода ---

  async function translate(video, btn) {
    const s = state.get(video);
    s.cancelled = false;
    s.jobId = null;
    setButton(btn, "UVT ⏳ 0%", "rgba(120, 90, 0, 0.8)");
    try {
      const candidates = await ensureCandidates(video);
      const job = await api("/dub", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          page_url: location.href,
          media_url: mediaUrlOf(video),
          media_candidates: candidates,
          // длительность из плеера — сервер отбрасывает потоки-превью
          duration_hint: Number.isFinite(video.duration) ? video.duration : null,
          source_lang: prefs.source,
          target_lang: prefs.target,
          voice_gender: prefs.voice,
        }),
      });
      s.jobId = job.id;
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        if (s.cancelled) { setButton(btn, "UVT", "rgba(20, 20, 20, 0.75)"); return; }
        const info = await api("/job/" + job.id);
        if (info.status === "done") {
          attachAudio(video, info.audio_url, info.entries);
          setButton(btn, "UVT ✓ выкл?", "rgba(20, 110, 50, 0.85)");
          return;
        }
        if (info.status === "cancelled") {
          setButton(btn, "UVT", "rgba(20, 20, 20, 0.75)");
          return;
        }
        if (info.status === "error") throw new Error(info.detail || "ошибка сервера");
        const pct = Math.round((info.progress || 0) * 100);
        const label = pct === 0 && info.detail ? "загрузка" : pct + "%";
        setButton(btn, "UVT ⏳ " + label);
      }
    } catch (err) {
      console.warn("[UVT]", err);
      setButton(btn, "UVT ✗", "rgba(150, 30, 30, 0.85)");
      btn.title = String(err) + " — запущен ли uvt serve?";
    }
  }

  async function cancelJob(video, btn) {
    const s = state.get(video);
    s.cancelled = true;
    if (s.jobId) {
      try {
        await api("/job/" + s.jobId + "/cancel", { method: "POST" });
      } catch (_) { /* сервер мог уже завершить задачу */ }
    }
    setButton(btn, "UVT", "rgba(20, 20, 20, 0.75)");
  }

  // --- выбор языков ---

  function chipLabel() {
    return prefs.source + " → " + prefs.target;
  }

  function makeSelect(current, withAuto, onChange) {
    const select = document.createElement("select");
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

  function makeSlider(min, max, step, value, onInput) {
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.style.width = "140px";
    input.addEventListener("input", () => onInput(parseFloat(input.value)));
    for (const event of ["click", "mousedown", "mousemove"]) {
      input.addEventListener(event, (e) => e.stopPropagation());
    }
    return input;
  }

  function toggleLangPanel(wrapper, chip, video) {
    const existing = wrapper.querySelector(".uvt-panel");
    if (existing) { existing.remove(); return; }

    const panel = document.createElement("div");
    panel.className = "uvt-panel";
    Object.assign(panel.style, {
      position: "absolute",
      top: "34px",
      left: "0",
      display: "grid",
      gridTemplateColumns: "auto auto",
      gap: "6px",
      alignItems: "center",
      padding: "10px",
      background: "rgba(15, 15, 15, 0.92)",
      border: "1px solid rgba(255,255,255,0.25)",
      borderRadius: "10px",
      color: "#ddd",
      font: "12px -apple-system, system-ui, sans-serif",
      zIndex: "2147483647",
      whiteSpace: "nowrap",
    });

    const refresh = () => { chip.textContent = chipLabel(); };
    const rowSource = document.createElement("span");
    rowSource.textContent = "С какого:";
    const rowTarget = document.createElement("span");
    rowTarget.textContent = "На какой:";
    panel.appendChild(rowSource);
    panel.appendChild(makeSelect(prefs.source, true, (v) => { prefs.source = v; refresh(); }));
    panel.appendChild(rowTarget);
    panel.appendChild(makeSelect(prefs.target, false, (v) => { prefs.target = v; refresh(); }));

    const rowVoice = document.createElement("span");
    rowVoice.textContent = "Голос:";
    const voiceSelect = document.createElement("select");
    Object.assign(voiceSelect.style, {
      font: "12px -apple-system, system-ui, sans-serif",
      background: "#222", color: "#fff",
      border: "1px solid #555", borderRadius: "6px", padding: "2px 4px",
    });
    for (const [value, label] of [["auto", "авто (по голосу)"], ["male", "мужской"], ["female", "женский"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      if (value === prefs.voice) option.selected = true;
      voiceSelect.appendChild(option);
    }
    voiceSelect.addEventListener("change", () => { prefs.voice = voiceSelect.value; });
    voiceSelect.addEventListener("click", (e) => e.stopPropagation());
    panel.appendChild(rowVoice);
    panel.appendChild(voiceSelect);

    // Громкости: оригинал под репликами и переведённый голос — на лету
    const duckLabel = document.createElement("span");
    const refreshDuckLabel = () => {
      duckLabel.textContent = `Оригинал: ${Math.round(prefs.duck * 100)}%`;
    };
    refreshDuckLabel();
    duckLabel.title = "Громкость основной дорожки, пока звучит перевод";
    panel.appendChild(duckLabel);
    panel.appendChild(makeSlider(0, 0.6, 0.05, prefs.duck, (v) => {
      prefs.duck = v;
      refreshDuckLabel(); // приглушение подхватится на ближайшем тике
    }));

    const volLabel = document.createElement("span");
    const refreshVolLabel = () => {
      volLabel.textContent = `Перевод: ${Math.round(prefs.voiceVol * 100)}%`;
    };
    refreshVolLabel();
    volLabel.title = "Громкость переведённого голоса";
    panel.appendChild(volLabel);
    panel.appendChild(makeSlider(0.2, 1, 0.05, prefs.voiceVol, (v) => {
      prefs.voiceVol = v;
      refreshVolLabel();
      const s = video && state.get(video);
      if (s && s.audio) s.audio.volume = v; // сразу на играющем переводе
    }));

    const hint = document.createElement("div");
    hint.textContent = "применится к следующему нажатию UVT";
    hint.style.gridColumn = "1 / -1";
    hint.style.color = "#888";
    panel.appendChild(hint);

    panel.addEventListener("click", (e) => e.stopPropagation());
    wrapper.appendChild(panel);
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
      gap: "6px",
      alignItems: "flex-start",
      opacity: "1",
      transition: "opacity 0.25s ease",
    });

    const btn = document.createElement("div");
    setButton(btn, "UVT");
    Object.assign(btn.style, CHIP_STYLE);
    btn.title = "Перевести и заменить голос (локальный UVT-сервер)";

    const chip = document.createElement("div");
    chip.textContent = chipLabel();
    Object.assign(chip.style, CHIP_STYLE);
    chip.title = "Языки и голос перевода";

    const cancelBtn = document.createElement("div");
    cancelBtn.textContent = "✕";
    Object.assign(cancelBtn.style, CHIP_STYLE);
    cancelBtn.style.display = "none";
    cancelBtn.title = "Отменить подготовку перевода";

    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      const s = state.get(video);
      if (s.on) {
        detachAudio(video);
        setButton(btn, "UVT", "rgba(20, 20, 20, 0.75)");
      } else if (!s.busy) {
        s.busy = true;
        cancelBtn.style.display = "";
        translate(video, btn).finally(() => {
          s.busy = false;
          cancelBtn.style.display = "none";
        });
      }
    });

    cancelBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      cancelJob(video, btn);
    });

    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      event.preventDefault();
      toggleLangPanel(wrapper, chip, video);
    });

    wrapper.appendChild(btn);
    wrapper.appendChild(chip);
    wrapper.appendChild(cancelBtn);
    parent.appendChild(wrapper);
    state.set(video, { button: btn, chip, on: false, busy: false });
    setupAutoHide(video, wrapper);
  }

  function setupAutoHide(video, wrapper) {
    const container = video.parentElement;
    let timer = null;

    // Прячемся всегда, когда мышь замерла; исключение одно — открытая панель
    // языков (иначе она закроется под руками).
    const mustStay = () => !!wrapper.querySelector(".uvt-panel");
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
      wrapper.style.opacity = "0";
      wrapper.style.pointerEvents = "none";
    };
    const poke = () => {
      show();
      clearTimeout(timer);
      timer = setTimeout(hide, 2800);
    };

    for (const el of [container, video, wrapper]) {
      el.addEventListener("mousemove", poke);
      el.addEventListener("mouseenter", poke);
    }
    container.addEventListener("mouseleave", () => {
      clearTimeout(timer);
      timer = setTimeout(hide, 400);
    });
    video.addEventListener("pause", poke);
    video.addEventListener("play", poke);
    poke();
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
    if (s) s.cancelled = true; // остановить опрос задачи, если шёл
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
