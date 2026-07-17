// ==UserScript==
// @name         UVT — закадровый перевод видео
// @namespace    uvt
// @version      0.11.3
// @description  Пакетный закадровый перевод видео через локальный сервер uvt serve: готовит синхронную дорожку, не live-перевод
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

  async function api(path, options) {
    const response = await fetch(SERVER + path, options);
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(
        "сервер UVT: HTTP " + response.status + (detail ? " — " + detail.slice(0, 360) : "")
      );
    }
    return response.json();
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
    heading.textContent = "Не удалось подготовить перевод";
    heading.style.display = "block";
    panel.appendChild(heading);
    const body = document.createElement("div");
    body.textContent = String(error && error.message ? error.message : error);
    body.style.marginTop = "3px";
    panel.appendChild(body);
    const help = document.createElement("div");
    help.textContent = "Проверьте, что запущен uvt serve, ссылка доступна без DRM, а выбранные движки настроены.";
    help.style.color = "#f2c9c9";
    help.style.marginTop = "4px";
    panel.appendChild(help);

    const actions = document.createElement("div");
    Object.assign(actions.style, { display: "flex", gap: "6px", marginTop: "8px" });
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
    // Загрузка не входит в progress рендера: сервер сообщает её отдельно,
    // чтобы не рисовать вечные 0% при прямом потоке через ffmpeg.
    const pctBase = info.stage === "download" && Number.isFinite(stageProgress)
      ? stageProgress
      : (Number(info.progress) || 0);
    const pct = Math.round(pctBase * 100);
    let message;
    if (info.stage === "queue" || info.status === "queued") {
      const position = Number(info.queue_position);
      message = Number.isFinite(position) && position > 1
        ? `В очереди: перед вами ${position - 1}.`
        : "В очереди: задача следующая.";
    } else {
      message = `Этап: ${stage} (${pct}%).`;
    }
    const eta = Number(info.eta_seconds);
    const etaText = info.eta_is_estimate && Number.isFinite(eta)
      ? `Оценка до готовности ${formatDuration(eta)}.`
      : "";
    setButton(
      s.button,
      `UVT · ${info.stage === "queue" ? "очередь" : stage}`,
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

  function attachAudio(video, audioUrl, entries) {
    const s = state.get(video);
    const audio = new Audio(SERVER + audioUrl);
    audio.preload = "auto";
    s.audio = audio;
    s.windows = buildWindows(entries);
    s.ducker = createDucker(video);
    s.audioErrorHandler = () => {
      if (!s.on) return;
      detachAudio(video);
      setRetryButton(video);
      showError(video, new Error("готовая аудиодорожка не загрузилась с локального сервера"));
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

  function beginTranslation(video) {
    const s = state.get(video);
    if (!s || s.busy || s.on) return;
    clearError(video);
    // Новый (в том числе повторный) запуск не должен оставлять рядом открытые
    // ползунки настроек. Прогресс новой задачи отражается только в кнопке.
    closeLangPanel(s.wrapper, s.chip, false);
    s.busy = true;
    s.cancelled = false;
    s.jobId = null;
    s.cancelButton.hidden = false;
    s.cancelButton.disabled = false;
    renderJobStatus(video, { status: "queued", stage: "queue", progress: 0 });
    translate(video).finally(() => {
      const current = state.get(video);
      if (!current) return;
      current.busy = false;
      current.cancelButton.hidden = true;
      current.cancelButton.disabled = false;
      if (!current.on && !current.errorPanel) setIdleButton(video);
    });
  }

  async function translate(video) {
    const s = state.get(video);
    try {
      const candidates = await ensureCandidates(video);
      if (s.cancelled) return;
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
      if (s.cancelled) {
        if (job.id) await api("/job/" + job.id + "/cancel", { method: "POST" }).catch(() => {});
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
        const info = await api("/job/" + job.id);
        if (info.status === "done") {
          if (!info.audio_url) throw new Error("сервер отметил задачу готовой, но не отдал аудиодорожку");
          attachAudio(video, info.audio_url, info.entries);
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
    if (s.jobId) {
      try {
        await api("/job/" + s.jobId + "/cancel", { method: "POST" });
      } catch (_) { /* сервер мог уже завершить задачу */ }
    }
  }

  // --- выбор языков ---

  function chipLabel() {
    return prefs.source + " → " + prefs.target;
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
    input.style.width = "140px";
    input.addEventListener("input", () => onInput(parseFloat(input.value)));
    for (const event of ["click", "mousedown", "mousemove"]) {
      input.addEventListener(event, (e) => e.stopPropagation());
    }
    return input;
  }

  function closeLangPanel(wrapper, chip, focusChip) {
    const existing = wrapper.querySelector(".uvt-panel");
    if (existing) existing.remove();
    chip.removeAttribute("aria-controls");
    chip.setAttribute("aria-expanded", "false");
    if (focusChip) chip.focus();
  }

  function toggleLangPanel(wrapper, chip, video) {
    const existing = wrapper.querySelector(".uvt-panel");
    if (existing) {
      closeLangPanel(wrapper, chip, false);
      return;
    }

    const panel = document.createElement("section");
    panel.className = "uvt-panel";
    panel.id = nextControlId("settings");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Настройки пакетного перевода UVT");
    panel.tabIndex = -1;
    chip.setAttribute("aria-controls", panel.id);
    chip.setAttribute("aria-expanded", "true");
    Object.assign(panel.style, {
      position: "absolute",
      top: "38px",
      left: "50%",
      transform: "translateX(-50%)",
      display: "grid",
      gridTemplateColumns: "auto minmax(150px, auto)",
      gap: "7px",
      alignItems: "center",
      padding: "10px",
      background: "rgba(15, 15, 15, 0.96)",
      border: "1px solid rgba(255,255,255,0.32)",
      borderRadius: "10px",
      color: "#ddd",
      font: "12px -apple-system, system-ui, sans-serif",
      zIndex: "2147483647",
      whiteSpace: "normal",
      boxShadow: "0 8px 26px rgba(0,0,0,.38)",
    });

    const refresh = () => {
      chip.textContent = chipLabel();
      chip.setAttribute("aria-label", `Настройки пакетного перевода: ${chipLabel()}`);
    };
    const sourceId = nextControlId("source");
    const rowSource = document.createElement("label");
    rowSource.htmlFor = sourceId;
    rowSource.textContent = "С какого:";
    const targetId = nextControlId("target");
    const rowTarget = document.createElement("label");
    rowTarget.htmlFor = targetId;
    rowTarget.textContent = "На какой:";
    panel.appendChild(rowSource);
    panel.appendChild(makeSelect(prefs.source, true, (v) => { prefs.source = v; refresh(); }, sourceId));
    panel.appendChild(rowTarget);
    panel.appendChild(makeSelect(prefs.target, false, (v) => { prefs.target = v; refresh(); }, targetId));

    const voiceId = nextControlId("voice");
    const rowVoice = document.createElement("label");
    rowVoice.htmlFor = voiceId;
    rowVoice.textContent = "Голос:";
    const voiceSelect = document.createElement("select");
    voiceSelect.id = voiceId;
    Object.assign(voiceSelect.style, {
      font: "12px -apple-system, system-ui, sans-serif",
      background: "#222", color: "#fff",
      border: "1px solid #555", borderRadius: "6px", padding: "2px 4px",
    });
    for (const [value, label] of [["auto", "авто (по тону)"], ["male", "мужской"], ["female", "женский"]]) {
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

    // Громкости: оригинал под репликами и переведённый голос — на лету.
    const duckId = nextControlId("original-volume");
    const duckLabel = document.createElement("label");
    duckLabel.htmlFor = duckId;
    const refreshDuckLabel = () => {
      duckLabel.textContent = `Оригинал: ${Math.round(prefs.duck * 100)}%`;
    };
    refreshDuckLabel();
    duckLabel.title = "Громкость основной дорожки, пока звучит перевод";
    panel.appendChild(duckLabel);
    panel.appendChild(makeSlider(0, 0.6, 0.05, prefs.duck, (v) => {
      prefs.duck = v;
      refreshDuckLabel(); // приглушение подхватится на ближайшем тике
    }, duckId));

    const volumeId = nextControlId("translation-volume");
    const volLabel = document.createElement("label");
    volLabel.htmlFor = volumeId;
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
    }, volumeId));

    const hint = document.createElement("div");
    hint.textContent = "Язык и голос применятся к следующему batch-запуску. «Авто» — эвристика тона, не определение личности или спикера.";
    hint.style.gridColumn = "1 / -1";
    hint.style.color = "#aeb6c2";
    panel.appendChild(hint);

    const close = document.createElement("button");
    close.type = "button";
    setButton(close, "Закрыть настройки", "transparent", "Закрыть настройки пакетного перевода");
    Object.assign(close.style, CHIP_STYLE, { gridColumn: "1 / -1", padding: "3px 7px", justifySelf: "end" });
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
    panel.addEventListener("click", (e) => e.stopPropagation());
    wrapper.appendChild(panel);
    const firstControl = panel.querySelector("select, input, button");
    if (firstControl) firstControl.focus();
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

    const btn = document.createElement("button");
    btn.type = "button";
    Object.assign(btn.style, CHIP_STYLE);
    btn.setAttribute("aria-pressed", "false");

    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = chipLabel();
    Object.assign(chip.style, CHIP_STYLE);
    chip.title = "Языки, голос и громкости пакетного перевода";
    chip.setAttribute("aria-haspopup", "dialog");
    chip.setAttribute("aria-expanded", "false");
    chip.setAttribute("aria-label", `Настройки пакетного перевода: ${chipLabel()}`);

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
    wrapper.appendChild(chip);
    wrapper.appendChild(cancelBtn);
    parent.appendChild(wrapper);
    state.set(video, {
      wrapper,
      button: btn,
      chip,
      cancelButton: cancelBtn,
      on: false,
      busy: false,
      cancelled: false,
      jobId: null,
      errorPanel: null,
    });
    setIdleButton(video);
    setupAutoHide(video, wrapper);
  }

  function setupAutoHide(video, wrapper) {
    const container = video.parentElement;
    let timer = null;
    let keyboardFocus = false;

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
        wrapper.querySelector(".uvt-panel") ||
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
      // hiding so an invisible control cannot receive the next Space/Enter.
      const active = document.activeElement;
      if (active instanceof HTMLElement && wrapper.contains(active)) active.blur();
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
    wrapper.addEventListener("focusin", () => {
      keyboardFocus = lastInteractionWasKeyboard;
      poke();
    });
    wrapper.addEventListener("focusout", (event) => {
      if (event.relatedTarget && wrapper.contains(event.relatedTarget)) return;
      keyboardFocus = false;
      clearTimeout(timer);
      timer = setTimeout(hide, 300);
    });
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
