"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const terminal = new Set(["done", "error", "cancelled"]);
  const labels = {queued: "В очереди", running: "Переводим", awaiting_approval: "Нужно решение", done: "Готово", error: "Не получилось", cancelled: "Отменено"};
  const stages = {queue: "Ожидаем свободную студию", queued: "Ожидаем свободную студию", download: "Получаем видео", decode: "Подготавливаем звук", separate: "Отделяем голос от фона · Demucs", upload: "Загружаем файл", transcribe: "Распознаём речь", translate: "Переводим реплики", synthesize: "Создаём озвучку", mix: "Собираем результат", encode: "Сохраняем результат", export: "Сохраняем результат", done: "Перевод готов"};
  const stageOrder = ["download", "transcribe", "translate", "synthesize", "mix"];
  const heavyProfileNote = "Тяжёлый профиль local-natural: Demucs отделяет голос от фона, F5 клонирует голос. На M4 / 16 ГБ обработка может занимать в несколько раз больше времени, чем длится речь. Для скорости выберите «Сбалансированный».";
  const storage = {
    get(key) { try { return sessionStorage.getItem(key) || ""; } catch { return ""; } },
    set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* Private storage may be disabled. */ } },
    remove(key) { try { sessionStorage.removeItem(key); } catch { /* Optional persistence. */ } }
  };
  const state = {tab: "file", file: null, token: storage.get("uvt-dashboard-token"), meta: null, settings: null, connected: false, uploading: false, creating: false, upload: null, job: null, pollTimer: null, pollController: null, pollGeneration: 0, refreshing: false, refreshTimer: null, restoreAttempted: false};

  class ApiError extends Error { constructor(message, status) { super(message); this.status = status; } }
  function messageFor(data, fallback) {
    const message = typeof data === "string" ? data : data?.error || data?.detail || data?.message || fallback;
    return String(typeof message === "object" ? message.message || fallback : message).slice(0, 1600);
  }
  function showAuth() {
    state.connected = false;
    $("auth-panel").hidden = false;
    connection("Нужен токен доступа", "error");
    updateSubmit();
  }
  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.token) headers.set("X-UVT-Token", state.token);
    const controller = new AbortController();
    const abort = () => controller.abort(options.signal.reason);
    if (options.signal?.aborted) abort();
    else options.signal?.addEventListener("abort", abort, {once: true});
    const timeout = setTimeout(() => controller.abort(new ApiError("Студия долго не отвечает. Попробуйте обновить состояние.", 0)), 20000);
    let response;
    try { response = await fetch(path, {...options, headers, signal: controller.signal, cache: "no-store"}); }
    finally { clearTimeout(timeout); options.signal?.removeEventListener("abort", abort); }
    let data;
    const raw = await response.text();
    try { data = raw ? JSON.parse(raw) : {}; } catch { data = raw; }
    if (!response.ok) {
      if (response.status === 401) showAuth();
      throw new ApiError(messageFor(data, `Не удалось выполнить запрос (${response.status})`), response.status);
    }
    return data;
  }
  function post(path, body) { return api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); }
  function connection(text, kind = "") { $("connection-text").textContent = text; $("connection-dot").className = `status-dot ${kind}`; }
  function lang(code) {
    if (!code || code === "auto") return "Автоопределение";
    const languages = state.settings?.catalog?.all_source_languages || [];
    return languages.find((item) => item.id === code)?.label || ({ru: "Русский", uk: "Украинский", en: "Английский", de: "Немецкий", fr: "Французский", es: "Испанский"}[code]) || code;
  }
  function setOptions(element, items, preferred, emptyLabel) {
    const selected = preferred ?? element.value;
    element.replaceChildren();
    if (emptyLabel) element.add(new Option(emptyLabel, ""));
    for (const raw of items || []) {
      const item = typeof raw === "string" ? {id: raw, label: raw} : raw;
      const option = new Option(String(item.label || item.name || item.id), String(item.id));
      option.disabled = item.installed === false;
      element.add(option);
    }
    element.value = selected || "";
    if (element.selectedIndex < 0 && element.options.length) element.selectedIndex = [...element.options].findIndex((item) => !item.disabled);
  }
  function selectedProfile() { return (state.meta?.profiles || []).find((item) => item.id === $("profile-select").value); }
  function updateVoices(preferred) {
    const voices = selectedProfile()?.voices ?? state.meta?.voices ?? state.settings?.catalog?.voices ?? [];
    const target = $("target-lang").value;
    const profile = selectedProfile();
    const engine = profile?.engines?.tts || state.meta?.profile?.engines?.tts;
    const compatible = voices.filter((voice) => voice.installed !== false && (!voice.language && !voice.languages || (voice.languages || [voice.language]).includes(target)) && (!voice.engine || voice.engine === engine));
    setOptions($("voice-select"), compatible, preferred, "Автоматически");
  }
  function profileChanged() {
    const profile = selectedProfile();
    const catalog = state.settings?.catalog || {};
    const targets = profile?.target_languages;
    const targetOptions = catalog.target_languages || [{id: "ru", label: "Русский"}, {id: "uk", label: "Украинский"}];
    setOptions($("target-lang"), targetOptions.filter((item) => !targets?.length || targets.includes(item.id)), $("target-lang").value);
    const all = catalog.all_source_languages || catalog.source_languages || [{id: "auto", label: "Автоопределение"}, {id: "en", label: "Английский"}, {id: "ru", label: "Русский"}, {id: "uk", label: "Украинский"}];
    const supported = profile?.source_languages || [];
    setOptions($("source-lang"), all.filter((item) => !supported.length || item.id === "auto" || supported.includes(item.id)), $("source-lang").value || state.meta?.profile?.source_lang || "auto");
    const readiness = profile?.installed === false ? "Для этого режима ещё нужно подготовить модели." : profile?.stt_loaded ? "Модель распознавания уже загружена." : "При первом запуске может понадобиться время на загрузку модели в память.";
    $("profile-note").textContent = profile?.id === "local-natural"
      ? `${heavyProfileNote} ${readiness} Снимите «Выбрать настройки для этого видео», чтобы вернуться к настройкам сервера.`
      : profile?.id === "local-balanced" ? `Рекомендуется для M4 / 16 ГБ: баланс скорости, памяти и качества. ${readiness}` : readiness;
    updateVoices();
    updateSubmit();
  }
  function renderMetadata(meta, settings) {
    const first = !state.meta;
    state.meta = meta; state.settings = settings;
    const profile = meta.profile || {};
    const catalog = settings?.catalog || {};
    const effective = settings?.effective || {};
    const profileLabel = meta.profiles?.find((item) => item.id === profile.name)?.label || profile.name || "Текущий профиль";
    $("profile-title").textContent = profile.name === "local-natural" && !profileLabel.includes("медленно") ? `${profileLabel} · медленно` : profileLabel;
    $("device-note").textContent = profile.name === "local-natural" ? heavyProfileNote : profile.name === "local-balanced" ? "Рекомендуется для M4 / 16 ГБ: баланс скорости, памяти и качества. Одна тяжёлая задача за раз." : "Одна тяжёлая задача за раз помогает сохранить память для вашего Mac.";
    $("mode-tag").textContent = {local: "ЛОКАЛЬНО", cloud: "ОБЛАКО", mixed: "СМЕШАННЫЙ", unknown: "РЕЖИМ"}[profile.kind] || "РЕЖИМ";
    $("profile-description").textContent = profile.kind === "local" ? "Обработка на вашем устройстве" : profile.kind === "mixed" ? "Часть обработки использует внешние сервисы" : profile.kind === "cloud" ? "Обработка использует внешние сервисы" : "Уточните настройки обработки";
    $("privacy-note").textContent = profile.kind === "local" ? "Локальный режим: перевод на вашем устройстве." : "Текущий режим использует внешние сервисы.";
    $("language-summary").textContent = `${lang(profile.source_lang)} → ${lang(profile.target_lang)}`;
    const readiness = meta.model_readiness || {};
    $("readiness").textContent = {ready: "Готовы", warming: "Подготавливаются", loading: "Загружаются", missing: "Нужна подготовка", error: "Нужна проверка", "on-demand": "При запуске", idle: "При запуске", pending: "Подготавливаются"}[readiness.status] || "Проверяем";
    $("metadata-note").textContent = ["missing", "error"].includes(readiness.status) ? String(readiness.detail || "Проверьте модели в настройках студии.") : "";
    // Catalog updates must preserve a user's unsent overrides.
    setOptions($("profile-select"), meta.profiles || [], first ? effective.profile_id || profile.name : $("profile-select").value);
    setOptions($("target-lang"), catalog.target_languages || [{id: "ru", label: "Русский"}, {id: "uk", label: "Украинский"}], first ? effective.target_lang || profile.target_lang : $("target-lang").value);
    profileChanged();
    if (first) {
      $("source-lang").value = effective.source_lang || profile.source_lang || "auto";
      updateVoices(effective.voice_id || meta.defaults?.voice_id || "");
    }
  }
  function humanSize(bytes) { return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} ГБ` : `${(bytes / 1024 ** 2).toFixed(1)} МБ`; }
  function audioOnly(file) { return !!file && (file.type.startsWith("audio/") || /\.(mp3|wav|m4a|flac|aac|aiff|opus|oga)$/i.test(file.name)); }
  function setFile(file) {
    if (!file) return;
    if (state.uploading || state.creating) return;
    if (file.size > 2 * 1024 ** 3) { $("form-error").textContent = "Файл больше 2 ГБ. Выберите более короткое видео или уменьшите его размер."; return; }
    if (!file.size) { $("form-error").textContent = "Этот файл пуст. Выберите другой файл."; return; }
    state.file = file;
    if (audioOnly(file)) $("export-video").checked = false;
    $("file-title").textContent = file.name;
    $("file-description").textContent = `${humanSize(file.size)} · готов к загрузке`;
    $("form-error").textContent = "";
    updateSubmit();
  }
  function selectTab(tab, moveFocus = false) {
    state.tab = tab;
    for (const button of document.querySelectorAll("[data-tab]")) {
      const selected = button.dataset.tab === tab;
      button.setAttribute("aria-selected", String(selected)); button.tabIndex = selected ? 0 : -1;
      if (selected && moveFocus) button.focus();
    }
    for (const kind of ["file", "url", "live"]) $(`panel-${kind}`).hidden = tab !== kind;
    $("job-options").hidden = tab === "live";
    $("video-url").required = tab === "url";
    $("form-error").textContent = "";
    updateSubmit();
  }
  function updateSubmit() {
    const hasInput = state.tab === "file" ? !!state.file : !!$("video-url").value.trim();
    const ready = state.connected && hasInput && !state.uploading && !state.creating;
    $("start-job").disabled = !ready || ($("override-settings").checked && selectedProfile()?.installed === false);
    $("start-job").textContent = state.uploading ? "Загружаем…" : state.creating ? "Создаём задание…" : "Начать перевод →";
    $("file-input").disabled = state.uploading || state.creating;
    $("export-video").disabled = state.uploading || state.creating || audioOnly(state.file);
    $("video-url").disabled = state.uploading || state.creating;
    $("override-settings").disabled = state.uploading || state.creating;
    $("override-fields").disabled = !$("override-settings").checked || state.uploading || state.creating;
  }
  function requestOptions() {
    const override = $("override-settings").checked;
    const options = {settings_mode: override ? "override" : "server", workspace_video: true};
    if (override) {
      if ($("profile-select").value) options.profile_id = $("profile-select").value;
      options.source_lang = $("source-lang").value || "auto";
      options.target_lang = $("target-lang").value;
      options.voice_gender = state.meta?.defaults?.voice_gender || "auto";
      if ($("voice-select").value) options.voice_id = $("voice-select").value;
    }
    return options;
  }
  function stopPolling() {
    clearTimeout(state.pollTimer); state.pollTimer = null;
    state.pollGeneration += 1;
    state.pollController?.abort(); state.pollController = null;
  }
  function jobId(job) { return String(job?.id || job?.job_id || ""); }
  function titleFor(job) {
    const name = job?.source_name || job?.filename || job?.title || job?.source_title;
    if (name) return String(name).split(/[\\/]/).pop();
    if (job?.page_url) { try { return new URL(job.page_url).hostname; } catch { /* Use the generic title. */ } }
    return "Перевод видео";
  }
  function duration(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0));
    return value < 60 ? `${value} с` : `${Math.floor(value / 60)} мин${value % 60 ? ` ${value % 60} с` : ""}`;
  }
  function signedUrl(value) {
    try { const url = new URL(typeof value === "string" ? value : value.url, location.href); return url.origin === location.origin && ["http:", "https:"].includes(url.protocol) ? url.href : ""; } catch { return ""; }
  }
  function stopPlayback() {
    $("result-video").pause(); $("result-video").removeAttribute("src"); $("result-video").load();
    $("result-audio").pause(); $("result-audio").removeAttribute("src"); $("result-audio").load();
  }
  function syncTranslation(play = false) {
    const video = $("result-video"), audio = $("result-audio");
    if (!video.getAttribute("src") || !audio.getAttribute("src")) return;
    audio.playbackRate = video.playbackRate;
    if (Number.isFinite(video.currentTime) && Math.abs(audio.currentTime - video.currentTime) > 0.15) {
      try { audio.currentTime = video.currentTime; } catch { /* Wait for metadata. */ }
    }
    if (play && !video.paused && !video.seeking) {
      const pending = audio.play();
      pending?.catch((error) => { if (error?.name !== "AbortError" && !video.paused) $("video-error").textContent = "Браузер не запустил перевод. Нажмите паузу и воспроизведение ещё раз."; });
    }
  }
  function updateVolumes() {
    const original = Number($("original-volume").value) / 100;
    const translation = Number($("translation-volume").value) / 100;
    $("result-video").volume = original;
    $("result-audio").volume = translation;
    $("original-volume-value").value = `${Math.round(original * 100)}%`;
    $("translation-volume-value").value = `${Math.round(translation * 100)}%`;
  }
  function renderVideo(job) {
    const info = job.video || {};
    const url = signedUrl(info.url);
    const available = job.status === "done" && (info.can_prepare || url || info.has_video === false);
    $("video-result").hidden = !available;
    const busy = ["queued", "running"].includes(info.status);
    $("prepare-video").hidden = !!url || info.has_video === false;
    $("prepare-video").disabled = busy;
    $("prepare-video").textContent = busy ? "Готовим видео…" : "Подготовить видео";
    $("cancel-video").hidden = !busy;
    $("save-video-mix").hidden = !url;
    $("save-video-mix").disabled = busy;
    $("video-mixer").hidden = !url;
    $("result-video").hidden = !url;
    $("original-volume").disabled = info.independent_audio === false;
    $("video-note").textContent = info.detail || (info.has_video === false ? "У этого источника только звук." : url ? "Регулируйте оригинал и перевод отдельно. Для скачивания с новыми уровнями нажмите «Сохранить видео с этой громкостью»." : "Исходный видеоряд можно добавить к готовому переводу. Переводить повторно не нужно.");
    if (info.resolution) $("video-note").textContent += ` Исходное разрешение: ${info.resolution}.`;
    if (info.independent_audio === false) $("video-note").textContent += " В старой дорожке оригинал уже смешан с переводом; отдельно менять его громкость можно в новом задании.";
    if (url && $("result-video").getAttribute("src") !== url) {
      $("result-video").src = url;
      $("original-volume").value = info.independent_audio === false ? 0 : Math.round((info.original_volume ?? 0.15) * 100);
      $("translation-volume").value = Math.round((info.translation_volume ?? 1) * 100);
      updateVolumes();
    }
    $("result-audio").hidden = !!url;
    if (!url) $("result-audio").volume = 1;
  }
  async function prepareVideo() {
    const id = jobId(state.job);
    if (!id) return;
    $("video-error").textContent = "";
    $("prepare-video").disabled = true; $("save-video-mix").disabled = true;
    try {
      const job = await post(`/workspace/job/${encodeURIComponent(id)}/video`, {original_volume: Number($("original-volume").value) / 100, translation_volume: Number($("translation-volume").value) / 100});
      if (id !== jobId(state.job)) return;
      renderJob(job);
      stopPolling(); pollJob(id, state.pollGeneration);
    } catch (error) {
      if (id === jobId(state.job)) { $("video-error").textContent = error.message; $("prepare-video").disabled = false; $("save-video-mix").disabled = false; }
    }
  }
  $("prepare-video").addEventListener("click", prepareVideo);
  $("save-video-mix").addEventListener("click", prepareVideo);
  $("cancel-video").addEventListener("click", async () => {
    const id = jobId(state.job);
    try { await post(`/workspace/job/${encodeURIComponent(id)}/video/cancel`, {}); stopPolling(); pollJob(id, state.pollGeneration); }
    catch (error) { $("video-error").textContent = error.message; }
  });
  for (const name of ["original", "translation"]) $(`${name}-volume`).addEventListener("input", updateVolumes);
  $("result-video").addEventListener("play", () => syncTranslation(true));
  $("result-video").addEventListener("playing", () => syncTranslation(true));
  for (const name of ["pause", "ended", "waiting", "seeking"]) $("result-video").addEventListener(name, () => $("result-audio").pause());
  $("result-video").addEventListener("seeked", () => syncTranslation(true));
  $("result-video").addEventListener("volumechange", () => { $("original-volume").value = Math.round($("result-video").volume * 100); $("original-volume-value").value = `${$("original-volume").value}%`; });
  $("result-video").addEventListener("ratechange", () => syncTranslation());
  $("result-video").addEventListener("timeupdate", () => syncTranslation());
  $("result-audio").addEventListener("loadedmetadata", () => syncTranslation(true));
  $("result-video").addEventListener("error", () => { $("video-error").textContent = "Браузер не воспроизвёл видео. Скачайте MP4 и откройте во внешнем плеере."; });
  function renderDownloads(job) {
    const downloads = job.downloads || {};
    const available = Object.entries(downloads).filter(([kind, value]) => ["original", "translated", "m4a", "srt", "vtt", "txt", "json", "mkv"].includes(kind) && signedUrl(value)).sort(([a], [b]) => (["original", "translated"].includes(a) ? 0 : 1) - (["original", "translated"].includes(b) ? 0 : 1));
    $("job-results").hidden = job.status !== "done" || !available.length;
    if (job.status !== "done") return;
    const audio = signedUrl(downloads.m4a);
    $("result-audio").hidden = !audio;
    if (audio && $("result-audio").getAttribute("src") !== audio) $("result-audio").src = audio;
    $("download-links").replaceChildren();
    const names = {m4a: "Озвучка · M4A", srt: "Субтитры · SRT", vtt: "Субтитры · VTT", txt: "Текст перевода", json: "Данные · JSON", mkv: "Две дорожки · MKV", original: "Оригинальное видео", translated: "Видео с переводом · MP4"};
    for (const [kind, value] of available) {
      const link = document.createElement("a"); link.href = signedUrl(value); link.textContent = `${names[kind]} ↓`; link.download = "";
      $("download-links").append(link);
    }
    renderVideo(job);
  }
  function renderJob(job) {
    const previousStage = jobId(state.job) === jobId(job) ? state.job?.stage : "";
    state.job = job;
    const detail = String(job.detail || job.error || "");
    const stageProgress = job.stage_progress == null ? null : Number(job.stage_progress);
    const waitingForSource = job.status === "running" && job.stage === "download"
      && (stageProgress === null || !Number.isFinite(stageProgress) || stageProgress <= 0);
    const pageExtraction = /yt-dlp|страниц|сайт|плеер/i.test(detail);
    const failedStage = job.failed_stage || job.error_stage || "";
    const sourceFailure = job.status === "error" && (failedStage ? failedStage === "download"
      : previousStage === "download" || /yt-dlp|страниц|источник|медиапоток|поток.*(?:скач|недоступ)/i.test(detail));
    $("job-card").hidden = false;
    $("job-title").textContent = titleFor(job);
    $("job-status").textContent = labels[job.status] || "Получаем состояние";
    $("job-status").className = `job-status ${job.status || ""}`;
    $("job-detail").textContent = String(job.detail || (job.status === "done" ? "Перевод завершён. Можно прослушать и сохранить результат." : job.status === "error" ? job.error || "Не удалось завершить перевод. Вы можете повторить с другим источником." : job.status === "cancelled" ? "Задание остановлено." : stages[job.stage] || "Подготавливаем видео…"));
    $("progress-label").textContent = stages[job.stage] || labels[job.status] || "Обработка";
    if (!terminal.has(job.status) && job.profile_name === "local-natural" && ["separate", "transcribe", "translate", "synthesize", "mix"].includes(job.stage)) {
      $("job-detail").textContent += ` ${heavyProfileNote}`;
      $("progress-label").textContent += " · тяжёлый профиль";
    }
    if (waitingForSource) {
      $("job-status").textContent = "Ждём источник";
      $("progress-label").textContent = pageExtraction ? "Ищем доступный звук на странице" : "Ожидаем данные источника";
      $("job-detail").textContent = `${detail || "Получаем исходный звук."} Процент загрузки ещё не получен. Распознавание и перевод пока не начались.`;
    }
    $("source-recovery").hidden = !sourceFailure;
    $("job-progress").hidden = waitingForSource;
    const progress = Number(job.progress);
    if (Number.isFinite(progress) && job.progress !== null) {
      const fraction = job.status === "done" ? 1 : Math.min(Math.max(progress, 0), .99);
      $("job-progress").value = fraction; $("progress-percent").textContent = `${Math.round(fraction * 100)}%`;
    } else { $("job-progress").removeAttribute("value"); $("progress-percent").textContent = ""; }
    if (waitingForSource) $("progress-percent").textContent = "Ожидание";
    const stage = ["encode", "export", "done"].includes(job.stage) ? "mix" : ["decode", "separate"].includes(job.stage) ? "download" : job.stage;
    const current = stageOrder.indexOf(stage);
    // Keep the existing compact steps truthful while showing preparation as its own stage.
    $("job-steps").setAttribute("aria-label", ["decode", "separate"].includes(job.stage) ? stages[job.stage] : "Этапы перевода");
    for (const item of $("job-steps").children) {
      const index = stageOrder.indexOf(item.dataset.stage);
      item.className = job.status === "done" || index < current ? "passed" : index === current ? "current" : "";
    }
    const timing = job.timing || {};
    const position = job.queue_position ?? job.queue?.position;
    $("job-timing").textContent = job.status === "queued" ? (position ? `Место в очереди: ${position}` : "Ждём завершения другого перевода") : timing.elapsed_s != null || timing.elapsed_seconds != null ? `Прошло ${duration(timing.elapsed_s ?? timing.elapsed_seconds)}` : "";
    if (waitingForSource) {
      const stageElapsed = timing.stage_elapsed_seconds ?? timing.stage_elapsed_s;
      if (stageElapsed != null) $("job-timing").textContent = `Получение источника: ${duration(stageElapsed)}`;
      if (Number(stageElapsed) >= 45) $("job-detail").textContent += " Если ожидание затянулось, можно отменить задачу и попробовать UVT прямо в плеере Chrome или загрузить локальный файл.";
    }
    $("cancel-job").hidden = terminal.has(job.status);
    $("cancel-job").disabled = false;
    $("cancel-job").textContent = "Отменить";
    $("retry-job").hidden = true;
    $("job-approval").hidden = job.status !== "awaiting_approval";
    $("approval-detail").textContent = job.approval_cause || job.detail || "Просмотрите условия и разрешите или отклоните продолжение задания.";
    $("job-network").textContent = "";
    renderDownloads(job);
  }
  async function pollJob(id, generation) {
    if (generation !== state.pollGeneration) return;
    state.pollController = new AbortController();
    try {
      const payload = await api(`/job/${encodeURIComponent(id)}`, {signal: state.pollController.signal});
      if (generation !== state.pollGeneration) return;
      const job = payload.job || payload;
      renderJob(job);
      if (terminal.has(job.status) && !["queued", "running"].includes(job.video?.status)) { await refreshJobs(); return; }
      state.pollTimer = setTimeout(() => pollJob(id, generation), 1600);
    } catch (error) {
      if (generation !== state.pollGeneration || error.name === "AbortError") return;
      $("job-network").textContent = error.status === 404 ? "Это задание больше недоступно. Возможно, студия была перезапущена. Обновите список или создайте новый перевод." : error.status === 401 ? "Введите токен доступа, чтобы продолжить просмотр задания." : "Связь со студией прервалась. Задание может продолжаться — попробуем обновить состояние.";
      $("retry-job").hidden = false;
      if (![401, 404].includes(error.status)) state.pollTimer = setTimeout(() => pollJob(id, generation), 5000);
    }
  }
  function openJob(job, scroll = false) {
    const id = typeof job === "string" ? job : jobId(job);
    if (!id) return;
    stopPolling();
    stopPlayback();
    if (typeof job !== "string") renderJob(job);
    storage.set("uvt-workspace-job", id);
    pollJob(id, state.pollGeneration);
    if (scroll) $("job-card").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "nearest"});
  }
  async function refreshJobs() {
    const payload = await api("/workspace/jobs");
    const jobs = Array.isArray(payload.jobs) ? payload.jobs.slice(0, 30) : [];
    $("recent-list").replaceChildren();
    $("recent-empty").hidden = !!jobs.length;
    $("recent-empty").textContent = "Здесь появятся ваши переводы. Начните с файла или ссылки.";
    for (const job of jobs) {
      const button = document.createElement("button"); button.type = "button"; button.className = "recent-item";
      const icon = document.createElement("span"); icon.textContent = job.status === "done" ? "✓" : job.status === "error" ? "!" : "▷"; icon.setAttribute("aria-hidden", "true");
      const description = document.createElement("span"); description.className = "recent-description";
      const title = document.createElement("strong"); title.textContent = titleFor(job);
      const subtitle = document.createElement("small");
      const timestamp = job.created_at || job.timing?.created_at;
      subtitle.textContent = timestamp ? new Date(Number(timestamp) * 1000).toLocaleString("ru-RU", {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"}) : "Открыть перевод";
      description.append(title, subtitle);
      const status = document.createElement("span"); status.className = "recent-status"; status.textContent = labels[job.status] || "Открыть";
      button.append(icon, description, status); button.addEventListener("click", () => { if (!state.creating && !state.uploading) openJob(job, true); }); $("recent-list").append(button);
    }
    if (!state.restoreAttempted) {
      state.restoreAttempted = true;
      const requested = new URLSearchParams(location.search).get("job") || "";
      const linked = /^[a-f0-9]{12}$/.test(requested) ? requested : "";
      const saved = linked || storage.get("uvt-workspace-job");
      const found = jobs.find((job) => jobId(job) === saved);
      if (!state.job && (found || linked)) openJob(found || linked);
      else if (saved && !found) storage.remove("uvt-workspace-job");
    }
  }
  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true; $("refresh").disabled = true;
    clearTimeout(state.refreshTimer);
    try {
      const [meta, settings] = await Promise.all([api("/meta"), api("/settings")]);
      renderMetadata(meta, settings);
      state.connected = true; connection("Студия подключена", "ready");
      $("auth-panel").hidden = true; $("auth-error").textContent = "";
      if (state.token) storage.set("uvt-dashboard-token", state.token);
      await refreshJobs();
    } catch (error) {
      if (error.status !== 401) { state.connected = false; connection("Студия недоступна", "error"); }
      $("recent-empty").textContent = error.status === 401 ? "Подключитесь, чтобы увидеть свои переводы." : "Не удалось получить задания. Нажмите «Обновить», когда студия снова будет доступна.";
      if (!$("recent-list").children.length) $("recent-empty").hidden = false;
    } finally {
      state.refreshing = false; $("refresh").disabled = false; updateSubmit();
      state.refreshTimer = setTimeout(refresh, 45000);
    }
  }
  function uploadFile(options) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); state.upload = xhr;
      const body = new FormData(); body.append("file", state.file); body.append("options", JSON.stringify({...options, export_video: $("export-video").checked}));
      xhr.open("POST", "/workspace/upload");
      if (state.token) xhr.setRequestHeader("X-UVT-Token", state.token);
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) { $("job-progress").value = event.loaded / event.total; $("progress-percent").textContent = `${Math.round(event.loaded / event.total * 100)}%`; $("job-detail").textContent = `${humanSize(event.loaded)} из ${humanSize(event.total)}`; }
      });
      xhr.upload.addEventListener("load", () => { $("job-progress").removeAttribute("value"); $("progress-percent").textContent = ""; $("job-detail").textContent = "Файл передан. Проверяем видео и создаём задание…"; });
      xhr.addEventListener("load", () => {
        let payload; try { payload = JSON.parse(xhr.responseText); } catch { payload = {}; }
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else { if (xhr.status === 401) showAuth(); reject(new ApiError(messageFor(payload, "Не удалось загрузить файл."), xhr.status)); }
      });
      xhr.addEventListener("error", () => reject(new ApiError("Связь прервалась во время загрузки. Проверьте последние задания перед повторной отправкой.", 0)));
      xhr.addEventListener("abort", () => reject(new ApiError("Загрузка прервана. Если файл уже был передан, проверьте последние задания — перевод мог успеть запуститься.", 499)));
      xhr.send(body);
    });
  }
  $("source-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.uploading || state.creating || !state.connected) return;
    $("form-error").textContent = "";
    const options = requestOptions();
    let sourceUrl;
    if (state.tab === "url") {
      try { sourceUrl = new URL($("video-url").value.trim()); if (!["http:", "https:"].includes(sourceUrl.protocol)) throw new Error(); }
      catch { $("form-error").textContent = "Вставьте полную ссылку на видео, начинающуюся с https:// или http://."; return; }
    } else if (!state.file || state.tab !== "file") return;
    stopPolling();
    state.creating = true; state.uploading = state.tab === "file"; updateSubmit();
    stopPlayback();
    renderJob({title: state.uploading ? state.file.name : sourceUrl.hostname, status: "running", stage: state.uploading ? "upload" : "download", progress: null, detail: state.uploading ? "Передаём файл в вашу студию…" : "Создаём задание…"});
    $("cancel-job").hidden = !state.uploading;
    $("cancel-job").textContent = "Отменить загрузку";
    try {
      const payload = state.uploading ? await uploadFile(options) : await post("/dub", {...options, page_url: sourceUrl.href});
      const job = payload.job || payload;
      if (!jobId(job)) throw new ApiError("Студия не вернула номер задания. Обновите последние переводы перед повторной отправкой.", 0);
      openJob(job, true); refreshJobs().catch(() => {});
    } catch (error) {
      $("form-error").textContent = error.message;
      $("job-status").textContent = error.status === 499 ? "Загрузка отменена" : "Проверьте отправку";
      $("job-detail").textContent = error.message; $("cancel-job").hidden = true;
      $("job-progress").value = 0; $("progress-percent").textContent = "";
      refreshJobs().catch(() => {});
    } finally { state.creating = false; state.uploading = false; state.upload = null; updateSubmit(); }
  });
  $("recovery-file").addEventListener("click", () => {
    selectTab("file", true);
    $("source-form").scrollIntoView({block: "start", behavior: "instant"});
    $("file-input").click();
  });
  $("cancel-job").addEventListener("click", async () => {
    if (state.upload) { state.upload.abort(); return; }
    const id = jobId(state.job); if (!id) return;
    $("cancel-job").disabled = true; $("cancel-job").textContent = "Останавливаем…";
    try { await post(`/job/${encodeURIComponent(id)}/cancel`, {}); openJob(id); }
    catch (error) { $("job-network").textContent = `Не удалось подтвердить отмену. ${error.message}`; $("cancel-job").disabled = false; $("cancel-job").textContent = "Отменить"; }
  });
  async function approve(approved) {
    const id = jobId(state.job); if (!id) return;
    $("approve-job").disabled = true; $("reject-job").disabled = true;
    try { await post(`/job/${encodeURIComponent(id)}/approve`, {approved}); openJob(id); }
    catch (error) { $("job-network").textContent = error.message; }
    finally { $("approve-job").disabled = false; $("reject-job").disabled = false; }
  }
  $("approve-job").addEventListener("click", () => approve(true)); $("reject-job").addEventListener("click", () => approve(false));
  $("retry-job").addEventListener("click", () => { if (jobId(state.job)) openJob(jobId(state.job)); });
  $("refresh").addEventListener("click", refresh);
  $("refresh-jobs").addEventListener("click", () => refreshJobs().catch((error) => { $("recent-empty").hidden = false; $("recent-empty").textContent = error.status === 401 ? "Подключитесь, чтобы увидеть задания." : "Не удалось обновить список. Попробуйте ещё раз."; }));
  $("auth-form").addEventListener("submit", async (event) => {
    event.preventDefault(); state.token = $("auth-token").value.trim();
    $("auth-error").textContent = "";
    await refresh();
    if (!state.connected) $("auth-error").textContent = "Подключиться не удалось. Проверьте токен и доступность студии.";
    else { $("auth-token").value = ""; if (jobId(state.job) && !terminal.has(state.job.status)) openJob(jobId(state.job)); }
  });
  for (const tab of document.querySelectorAll("[data-tab]")) {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      const kinds = ["file", "url", "live"], index = kinds.indexOf(state.tab);
      const destination = event.key === "ArrowRight" ? (index + 1) % 3 : event.key === "ArrowLeft" ? (index + 2) % 3 : event.key === "Home" ? 0 : event.key === "End" ? 2 : -1;
      if (destination >= 0) { event.preventDefault(); selectTab(kinds[destination], true); }
    });
  }
  $("file-input").addEventListener("change", (event) => setFile(event.target.files[0]));
  $("video-url").addEventListener("input", updateSubmit);
  for (const type of ["dragenter", "dragover"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); $("drop-zone").classList.add("dragging"); });
  for (const type of ["dragleave", "drop"]) $("drop-zone").addEventListener(type, (event) => { event.preventDefault(); $("drop-zone").classList.remove("dragging"); if (type === "drop") { if (event.dataTransfer.files.length > 1) $("form-error").textContent = "Выберите один файл для одного перевода."; else setFile(event.dataTransfer.files[0]); } });
  $("override-settings").addEventListener("change", () => { $("override-fields").hidden = !$("override-settings").checked; $("options-summary").textContent = $("override-settings").checked ? "Для этого видео" : "По настройкам студии"; updateSubmit(); });
  $("profile-select").addEventListener("change", () => { $("voice-select").value = ""; profileChanged(); }); $("target-lang").addEventListener("change", () => updateVoices());
  $("copy-command").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("meeting-command").textContent); $("copy-command").textContent = "Скопировано"; setTimeout(() => { $("copy-command").textContent = "Копировать"; }, 2000); }
    catch { const selection = window.getSelection(), range = document.createRange(); range.selectNodeContents($("meeting-command")); selection.removeAllRanges(); selection.addRange(range); $("copy-command").textContent = "Выделено"; }
  });
  window.addEventListener("online", () => { refresh(); if (jobId(state.job) && !terminal.has(state.job.status) && !state.creating) openJob(jobId(state.job)); });
  window.addEventListener("pagehide", () => { stopPolling(); clearTimeout(state.refreshTimer); });
  refresh();
})();
