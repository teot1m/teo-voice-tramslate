"""Понятное desktop-окно UVT: отдельные Live и Batch сценарии.

Live-конвейер крутится в фоновом потоке со своим asyncio-циклом; события
приходят в Qt через queued-сигналы. Batch-дубляж тоже запускается отдельно,
но не притворяется «мгновенным» переводом: это подготовка готовой дорожки.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from urllib.parse import urlparse

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uvt import registry
from uvt.config import AppConfig, load_config
from uvt.events import TOPIC_STATUS, ServiceStatus, SubtitleEvent
from uvt.gui.overlay_window import OverlayWindow

log = logging.getLogger("uvt.gui")

_LANGS = ["auto", "ru", "en", "uk", "de", "fr", "es", "it", "pt", "ja", "zh", "ko"]
_STATE_COLORS = {
    "running": "#1f9d62",
    "done": "#64748b",
    "stopped": "#64748b",
    "error": "#dc2626",
}
_SERVICE_NAMES = {
    "capture": "Захват",
    "vad": "Речь",
    "stt": "Распознавание",
    "translate": "Перевод",
    "tts": "Озвучка",
    "output": "Вывод",
}
_VIRTUAL_MARKERS = (
    "blackhole", "vb-audio", "cable", "monitor", "loopback", "virtual", "voicemeeter",
)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
_CURATED_PROFILES = (
    ("Текущая конфигурация", None),
    ("Local Private — всё на этом устройстве", "local"),
    ("Mac Local Fast — Parakeet + NLLB + Piper", "local-fast"),
    ("Диалог на Mac — Parakeet + NLLB + Piper", "local-dialogue"),
    ("Субтитры звонка — Parakeet + TranslateGemma", "local-meeting"),
    ("Mac Local Balanced — Parakeet + TranslateGemma", "local-balanced"),
    ("Hy-MT2 — быстрый перевод + Piper", "local-hymt"),
    ("Hy-MT2 + MOSS — живые голоса на CPU", "local-moss"),
    ("Nemotron + Hy-MT2 + Piper", "local-nemotron"),
    ("Mac Local Quality — Whisper + TranslateGemma", "local-quality"),
    ("Free — без оплаты, Edge TTS через сеть", "free"),
    ("Cloud Fast — меньше задержка", "cloud-fast"),
    ("Cloud ElevenLabs — другой провайдер озвучки", "cloud-eleven"),
    ("Cloud Quality — для пакетного дубляжа", "cloud-quality"),
    ("Live — системный звук через виртуальный вход", "live"),
)


def _is_local_endpoint(value: object) -> bool:
    """Возвращает True для локального OpenAI-совместимого endpoint."""
    text = str(value or "").strip()
    if not text:
        return True
    parsed = urlparse(text if "://" in text else f"http://{text}")
    return (parsed.hostname or "").lower() in _LOCAL_HOSTS


def _is_virtual_device(name: str) -> bool:
    return any(marker in name.lower() for marker in _VIRTUAL_MARKERS)


def _privacy_summary(cfg: AppConfig) -> tuple[str, str]:
    """Коротко и честно объясняет, какие данные покидают устройство."""
    local: list[str] = []
    remote: list[str] = []

    if cfg.stt.engine in {"faster-whisper", "mlx-whisper", "parakeet-mlx", "dummy"}:
        local.append("распознавание")
    elif cfg.stt.engine == "openai-compatible" and _is_local_endpoint(cfg.stt.base_url):
        local.append("распознавание")
    else:
        remote.append("аудио для распознавания")

    if cfg.translation.engine in {
        "nllb-ct2",
        "translategemma-mlx",
        "none",
        "passthrough",
        "dummy",
    }:
        local.append("перевод")
    elif _is_local_endpoint(getattr(cfg.translation, "base_url", "")):
        local.append("перевод")
    else:
        remote.append("текст для перевода")

    if cfg.tts.engine in {"none", "dummy", "piper", "kokoro"}:
        local.append("озвучка")
    elif cfg.tts.engine == "edge":
        remote.append("текст для Microsoft Edge TTS")
    elif cfg.tts.engine == "elevenlabs":
        remote.append("текст для ElevenLabs TTS")
    else:
        remote.append("текст для озвучки")

    if not remote:
        return (
            "PRIVATE LOCAL",
            "Маршрут локальный: " + ", ".join(local) + ". Модели должны быть установлены заранее.",
        )
    if not local:
        return (
            "CLOUD",
            "В облако уходят: " + ", ".join(remote) + ". Проверьте политику выбранного провайдера.",
        )
    return (
        "HYBRID",
        "Локально: " + ", ".join(local) + ". В облако: " + ", ".join(remote) + ".",
    )


class PipelineWorker(threading.Thread):
    """Фоновый поток Live-конвейера с телеметрией, пригодной для GUI."""

    def __init__(
        self,
        cfg: AppConfig,
        on_subtitle: Callable[[SubtitleEvent], None],
        on_status: Callable[[ServiceStatus], None],
        on_metrics: Callable[[object], None],
        on_finished: Callable[[], None],
    ) -> None:
        super().__init__(daemon=True, name="uvt-live-pipeline")
        self.cfg = cfg
        self.on_subtitle = on_subtitle
        self.on_status = on_status
        self.on_metrics = on_metrics
        self.on_finished = on_finished
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception as exc:  # noqa: BLE001 — ошибка должна попасть в интерфейс
            log.exception("конвейер аварийно завершился")
            self.on_status(ServiceStatus("pipeline", "error", str(exc)))
        finally:
            self.on_finished()

    async def _amain(self) -> None:
        from uvt.app import Pipeline

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        pipeline = Pipeline(self.cfg, subtitle_sink=self.on_subtitle)
        status_queue = pipeline.bus.topic(TOPIC_STATUS).subscribe()

        async def watch_status() -> None:
            while True:
                self.on_status(await status_queue.get())

        async def tick_metrics() -> None:
            while True:
                await asyncio.sleep(1)
                counters_fn = getattr(pipeline.metrics, "counters_snapshot", None)
                if callable(counters_fn):
                    counters = dict(counters_fn())
                else:
                    # Старое ядро не публиковало counters_snapshot. Это только
                    # очередь шины, поэтому подпись GUI не выдаёт его за все drops.
                    counters = {
                        f"queue.{name}": topic.dropped
                        for name, topic in getattr(pipeline.bus, "_topics", {}).items()
                        if topic.dropped
                    }
                self.on_metrics({
                    "latency": pipeline.metrics.format_line(),
                    "counters": counters,
                })

        watchers = [
            asyncio.create_task(watch_status()),
            asyncio.create_task(tick_metrics()),
        ]
        try:
            if not await pipeline.start(stop_event=self._stop_event):
                return
            await self._stop_event.wait()
        finally:
            for task in watchers:
                task.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)
            await pipeline.stop()

    def request_stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)


class _BatchLogHandler(logging.Handler):
    """Преобразует реальные сообщения dub.py в короткие стадии интерфейса."""

    def __init__(self, callback: Callable[[ServiceStatus], None]) -> None:
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        lowered = text.lower()
        if "декодирую" in lowered:
            detail = "Читаю источник и готовлю аудио"
        elif "распозна" in lowered or "whisper" in lowered:
            detail = "Распознаю речь и таймкоды"
        elif "перевод" in lowered:
            detail = "Перевожу реплики с контекстом"
        elif "озвуч" in lowered or "синтез" in lowered:
            detail = "Синтезирую перевод"
        elif "собираю" in lowered or "ffmpeg" in lowered:
            detail = "Собираю синхронную дорожку"
        else:
            return
        self.callback(ServiceStatus("batch", "running", detail))


class BatchWorker(threading.Thread):
    """Готовит файл/ссылку через существующий batch API без подмены Live."""

    def __init__(
        self,
        cfg: AppConfig,
        source: str,
        output: str | None,
        duck_db: float,
        keep_original: bool,
        on_status: Callable[[ServiceStatus], None],
        on_finished: Callable[[object], None],
    ) -> None:
        super().__init__(daemon=True, name="uvt-batch-dub")
        self.cfg = cfg
        self.source = source
        self.output = output
        self.duck_db = duck_db
        self.keep_original = keep_original
        self.on_status = on_status
        self.on_finished = on_finished

    def run(self) -> None:
        handler = _BatchLogHandler(self.on_status)
        logger = logging.getLogger("uvt.dub")
        logger.addHandler(handler)
        self.on_status(ServiceStatus("batch", "running", "Ожидаю подготовку движков"))
        try:
            from uvt.dub import dub

            result = asyncio.run(
                dub(
                    self.cfg,
                    self.source,
                    output=self.output or None,
                    duck_db=self.duck_db,
                    keep_original=self.keep_original,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("пакетный дубляж завершился ошибкой")
            self.on_status(ServiceStatus("batch", "error", str(exc)))
            self.on_finished((None, str(exc)))
        else:
            self.on_status(ServiceStatus("batch", "done", "Готово"))
            self.on_finished((str(result), None))
        finally:
            logger.removeHandler(handler)


class MainWindow(QMainWindow):
    subtitleReceived = Signal(object)
    statusChanged = Signal(object)
    metricsUpdated = Signal(object)
    pipelineFinished = Signal()
    batchFinished = Signal(object)

    def __init__(self, cfg: AppConfig, profile_name: str | None = None) -> None:
        super().__init__()
        self.initial_cfg = cfg.model_copy(deep=True)
        self.base_cfg = cfg.model_copy(deep=True)
        self.profile_name = profile_name
        self.worker: PipelineWorker | None = None
        self.batch_worker: BatchWorker | None = None
        self.error_count = 0
        self.overlay = OverlayWindow(cfg.overlay)
        self.setWindowTitle("UVT — перевод речи")
        self._build_ui()
        self.subtitleReceived.connect(self._show_subtitle)
        self.statusChanged.connect(self._show_status)
        self.metricsUpdated.connect(self._show_metrics)
        self.pipelineFinished.connect(self._pipeline_finished)
        self.batchFinished.connect(self._batch_finished)
        self._select_initial_profile(profile_name)
        self._refresh_privacy()

    def _build_ui(self) -> None:
        registry.load_builtins()
        self.setStyleSheet(
            """
            QMainWindow { background: #f8fafc; }
            QGroupBox { font-weight: 650; border: 1px solid #dbe3ee; border-radius: 8px;
                        margin-top: 10px; padding: 12px 10px 10px 10px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QTabWidget::pane { border: 1px solid #dbe3ee; border-radius: 8px; background: #ffffff; }
            QTabBar::tab { padding: 9px 16px; background: #eaf0f7; border: 1px solid #dbe3ee; }
            QTabBar::tab:selected { background: #ffffff; color: #0f5bd7; font-weight: 650; }
            QPushButton#primary { background: #1463d9; color: white; font-weight: 700; border: 0;
                                  border-radius: 7px; padding: 8px 14px; }
            QPushButton#primary:hover { background: #0e4fad; }
            QLabel#hint { color: #475569; }
            QLabel#privacy { background: #eff6ff; color: #1e3a5f; padding: 8px; border-radius: 6px; }
            QLabel#error { background: #fef2f2; color: #991b1b; padding: 8px; border-radius: 6px; }
            """
        )

        root = QVBoxLayout()
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("UVT — перевод без ложных обещаний")
        title.setStyleSheet("font-size: 21px; font-weight: 700; color: #0f172a;")
        subtitle = QLabel(
            "Live перевод начинается после фразы; пакетный дубляж готовит синхронную дорожку заранее."
        )
        subtitle.setObjectName("hint")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        root.addWidget(self._build_profile_panel())
        root.addWidget(self._build_engine_panel())

        self.workflows = QTabWidget()
        self.workflows.addTab(self._build_live_page(), "Live перевод")
        self.workflows.addTab(self._build_batch_page(), "Пакетный дубляж")
        root.addWidget(self.workflows, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self.resize(960, 760)
        self.setMinimumSize(780, 620)

    def _build_profile_panel(self) -> QGroupBox:
        panel = QGroupBox("Профиль и приватность")
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        self.profile_combo = QComboBox()
        for label, value in _CURATED_PROFILES:
            self.profile_combo.addItem(label, value)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        row.addWidget(QLabel("Маршрут:"))
        row.addWidget(self.profile_combo, 1)
        layout.addLayout(row)

        self.privacy_title = QLabel("—")
        self.privacy_title.setStyleSheet("font-weight: 700; color: #0f5bd7;")
        self.privacy_detail = QLabel()
        self.privacy_detail.setObjectName("privacy")
        self.privacy_detail.setWordWrap(True)
        layout.addWidget(self.privacy_title)
        layout.addWidget(self.privacy_detail)
        return panel

    def _build_engine_panel(self) -> QGroupBox:
        panel = QGroupBox("Языки и движки (общие для обоих сценариев)")
        form = QFormLayout(panel)

        def engine_combo(kind: str, current: str) -> QComboBox:
            box = QComboBox()
            for name in registry.available(kind):
                box.addItem(name)
            box.setCurrentText(current)
            box.currentTextChanged.connect(lambda _value: self._refresh_privacy())
            return box

        self.source_lang = QComboBox()
        self.source_lang.addItems(_LANGS)
        self.target_lang = QComboBox()
        self.target_lang.addItems([lang for lang in _LANGS if lang != "auto"])
        self.live_mode_combo = QComboBox()
        self.live_mode_combo.addItem("Голос поверх оригинала", "voiceover")
        self.live_mode_combo.addItem("Только субтитры", "subtitles")
        self.stt_combo = engine_combo("stt", self.base_cfg.stt.engine)
        self.translate_combo = engine_combo("translation", self.base_cfg.translation.engine)
        self.tts_combo = engine_combo("tts", self.base_cfg.tts.engine)
        self.overlay_check = QCheckBox("Показывать оверлей субтитров")

        form.addRow("Язык оригинала:", self.source_lang)
        form.addRow("Перевод на:", self.target_lang)
        form.addRow("Live-вывод:", self.live_mode_combo)
        form.addRow("Распознавание (STT):", self.stt_combo)
        form.addRow("Перевод:", self.translate_combo)
        form.addRow("Озвучка (TTS):", self.tts_combo)
        form.addRow("", self.overlay_check)
        self._refresh_engine_controls()
        return panel

    def _build_live_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(12, 12, 12, 12)
        source_box = QGroupBox("Звук для Live")
        source_form = QFormLayout(source_box)

        self.source_combo = QComboBox()
        self.output_combo = QComboBox()
        self._populate_audio_devices()
        self.source_combo.currentIndexChanged.connect(self._refresh_source_guidance)
        source_form.addRow("Вход:", self.source_combo)
        self.source_guidance = QLabel()
        self.source_guidance.setObjectName("hint")
        self.source_guidance.setWordWrap(True)
        source_form.addRow("", self.source_guidance)
        source_form.addRow("Вывод перевода:", self.output_combo)

        self.overlay_hint = QLabel(
            "Для системного звука выберите виртуальный вход: BlackHole на macOS, "
            "WASAPI loopback/VB-Cable на Windows или monitor PipeWire/PulseAudio на Linux."
        )
        self.overlay_hint.setObjectName("hint")
        self.overlay_hint.setWordWrap(True)
        source_form.addRow("", self.overlay_hint)

        self.start_button = QPushButton("Запустить Live перевод")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(42)
        self.start_button.clicked.connect(self._toggle_live)
        source_form.addRow("", self.start_button)

        source_box.setMinimumWidth(350)
        root.addWidget(source_box, 2)

        runtime = QVBoxLayout()
        runtime.addWidget(self._build_status_panel())
        self.subtitles_view = QPlainTextEdit()
        self.subtitles_view.setReadOnly(True)
        self.subtitles_view.setPlaceholderText(
            "Здесь появятся оригинал и перевод. Ошибки стадий остаются выше, а не прячутся в tooltip."
        )
        runtime.addWidget(self.subtitles_view, 1)
        root.addLayout(runtime, 3)
        self._refresh_source_guidance()
        return page

    def _build_status_panel(self) -> QGroupBox:
        panel = QGroupBox("Live: состояние и синхронность")
        layout = QVBoxLayout(panel)
        grid = QGridLayout()
        self.status_labels: dict[str, QLabel] = {}
        for index, service in enumerate(_SERVICE_NAMES):
            label = QLabel(f"● {_SERVICE_NAMES[service]} — ожидание")
            label.setStyleSheet("color: #64748b;")
            label.setToolTip("ещё не запускалось")
            self.status_labels[service] = label
            grid.addWidget(label, index // 2, index % 2)
        layout.addLayout(grid)
        self.metrics_label = QLabel("Задержка p50: —")
        self.metrics_label.setWordWrap(True)
        self.drop_label = QLabel("Пропущено очередью: 0")
        self.drop_label.setWordWrap(True)
        self.error_label = QLabel("Ошибок стадий: 0")
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.drop_label)
        layout.addWidget(self.error_label)
        return panel

    def _build_batch_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(18, 18, 18, 18)
        intro = QLabel(
            "Пакетный дубляж сначала анализирует весь файл/ролик, а потом создаёт дорожку по таймкодам. "
            "Это точнее синхронизируется с видео, но не является Live-переводом."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hint")
        root.addWidget(intro)

        form_box = QGroupBox("Источник и результат")
        form = QFormLayout(form_box)
        source_row = QHBoxLayout()
        self.batch_source = QLineEdit()
        self.batch_source.setPlaceholderText("/путь/к/видео.mp4 или https://…")
        source_browse = QPushButton("Выбрать файл…")
        source_browse.clicked.connect(self._choose_batch_source)
        source_row.addWidget(self.batch_source, 1)
        source_row.addWidget(source_browse)
        self.batch_source_browse = source_browse

        output_row = QHBoxLayout()
        self.batch_output = QLineEdit()
        self.batch_output.setPlaceholderText("Необязательно: UVT выберет имя рядом с источником")
        output_browse = QPushButton("Куда сохранить…")
        output_browse.clicked.connect(self._choose_batch_output)
        output_row.addWidget(self.batch_output, 1)
        output_row.addWidget(output_browse)
        self.batch_output_browse = output_browse

        self.duck_spin = QDoubleSpinBox()
        self.duck_spin.setRange(-60.0, 0.0)
        self.duck_spin.setSingleStep(1.0)
        self.duck_spin.setValue(-12.0)
        self.duck_spin.setSuffix(" dB")
        self.keep_original_check = QCheckBox("Сохранить исходную дорожку в итоговом MKV")
        self.keep_original_check.setChecked(True)
        form.addRow("Файл или ссылка:", source_row)
        form.addRow("Результат:", output_row)
        form.addRow("Оригинал под переводом:", self.duck_spin)
        form.addRow("", self.keep_original_check)
        root.addWidget(form_box)

        self.batch_stage = QLabel("Готов к подготовке. Стадии появятся по фактическим сообщениям движка.")
        self.batch_stage.setObjectName("privacy")
        self.batch_stage.setWordWrap(True)
        self.batch_error = QLabel("")
        self.batch_error.setObjectName("error")
        self.batch_error.setWordWrap(True)
        self.batch_error.hide()
        root.addWidget(self.batch_stage)
        root.addWidget(self.batch_error)

        self.batch_start_button = QPushButton("Подготовить синхронный дубляж")
        self.batch_start_button.setObjectName("primary")
        self.batch_start_button.setMinimumHeight(42)
        self.batch_start_button.clicked.connect(self._start_batch)
        root.addWidget(self.batch_start_button)
        caution = QLabel("Отмена пакетной обработки пока не поддерживается: закройте приложение только если готовы прервать работу.")
        caution.setObjectName("hint")
        caution.setWordWrap(True)
        root.addWidget(caution)
        root.addStretch(1)
        return page

    def _populate_audio_devices(self) -> None:
        self.source_combo.addItem("Вход по умолчанию — часто это микрофон", None)
        self.output_combo.addItem("Выход по умолчанию", None)
        try:
            import sounddevice as sd

            for index, device in enumerate(sd.query_devices()):
                name = str(device["name"])
                marker = "🔁 " if _is_virtual_device(name) else ""
                if device["max_input_channels"] > 0:
                    self.source_combo.addItem(f"{marker}{name}", index)
                if device["max_output_channels"] > 0:
                    self.output_combo.addItem(f"{marker}{name}", index)
        except Exception as exc:  # noqa: BLE001
            log.warning("не удалось перечислить аудиоустройства: %s", exc)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        # Профили исторически разрешают имя устройства (например,
        # "BlackHole 2ch"), тогда как GUI хранит его индекс в userData.
        # Подбираем по отображаемому имени, чтобы `uvt gui -p live` не
        # незаметно откатывался на микрофон по умолчанию.
        if index < 0 and isinstance(value, str):
            wanted = value.strip().lower()
            for candidate in range(combo.count()):
                label = combo.itemText(candidate).removeprefix("🔁 ").strip().lower()
                if label == wanted or wanted in label:
                    index = candidate
                    break
        if index >= 0:
            combo.setCurrentIndex(index)

    def _select_initial_profile(self, profile_name: str | None) -> None:
        if profile_name:
            index = self.profile_combo.findData(profile_name)
            if index >= 0:
                self.profile_combo.blockSignals(True)
                self.profile_combo.setCurrentIndex(index)
                self.profile_combo.blockSignals(False)

    def _on_profile_changed(self) -> None:
        profile = self.profile_combo.currentData()
        try:
            cfg = self.initial_cfg.model_copy(deep=True) if profile is None else load_config(profile)
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            self.error_label.setText(f"Ошибок стадий: {self.error_count}. Не удалось загрузить профиль: {exc}")
            return
        self.base_cfg = cfg
        self.profile_name = profile
        self._recreate_overlay()
        self._refresh_engine_controls()
        self._refresh_privacy()

    def _recreate_overlay(self) -> None:
        old_overlay = self.overlay
        old_overlay.hide()
        old_overlay.close()
        self.overlay = OverlayWindow(self.base_cfg.overlay)

    def _refresh_engine_controls(self) -> None:
        self.source_lang.setCurrentText(self.base_cfg.source_lang)
        self.target_lang.setCurrentText(self.base_cfg.target_lang)
        self._set_combo_data(self.live_mode_combo, self.base_cfg.mode if self.base_cfg.mode in {"voiceover", "subtitles"} else "voiceover")
        self.stt_combo.setCurrentText(self.base_cfg.stt.engine)
        self.translate_combo.setCurrentText(self.base_cfg.translation.engine)
        self.tts_combo.setCurrentText(self.base_cfg.tts.engine)
        self.overlay_check.setChecked(self.base_cfg.overlay.enabled)
        # Эта функция вызывается и при сборке верхней панели — до создания
        # live-виджетов. Не трогаем их, пока соответствующая вкладка не готова.
        if hasattr(self, "source_combo"):
            self._set_combo_data(self.source_combo, self.base_cfg.capture.device)
            self._set_combo_data(self.output_combo, self.base_cfg.output.device)
            self._refresh_source_guidance()

    def _refresh_privacy(self) -> None:
        # Показываем выбор в ComboBox сразу, до запуска; backend и model_path
        # остаются из профиля, а выбранные в форме движки подставляем в копию.
        cfg = self._collect_cfg()
        title, detail = _privacy_summary(cfg)
        self.privacy_title.setText(title)
        self.privacy_detail.setText(detail)

    def _refresh_source_guidance(self) -> None:
        if not hasattr(self, "source_combo"):
            return
        selected = self.source_combo.currentText()
        if self.source_combo.currentData() is None:
            text = (
                "Выбран системный вход по умолчанию. На большинстве компьютеров это микрофон, "
                "а не звук браузера или приложения."
            )
        elif _is_virtual_device(selected):
            text = "Виртуальный/monitor-вход выбран — подходит для захвата системного звука."
        else:
            text = (
                "Это обычный вход, обычно микрофон. Для звука приложения выберите устройство с 🔁 "
                "или настройте виртуальный кабель."
            )
        self.source_guidance.setText(text)

    def _collect_cfg(self) -> AppConfig:
        cfg = self.base_cfg.model_copy(deep=True)
        if hasattr(self, "source_combo"):
            cfg.capture.device = self.source_combo.currentData()
            cfg.output.device = self.output_combo.currentData()
        if hasattr(self, "source_lang"):
            cfg.source_lang = self.source_lang.currentText()
            cfg.target_lang = self.target_lang.currentText()
            cfg.mode = self.live_mode_combo.currentData() or "voiceover"
            cfg.stt.engine = self.stt_combo.currentText()
            cfg.translation.engine = self.translate_combo.currentText()
            cfg.tts.engine = self.tts_combo.currentText()
            cfg.overlay.enabled = self.overlay_check.isChecked()
        return cfg

    def _reset_live_status(self) -> None:
        self.error_count = 0
        self.error_label.setText("Ошибок стадий: 0")
        self.metrics_label.setText("Задержка p50: —")
        self.drop_label.setText("Пропущено очередью: 0")
        for service, label in self.status_labels.items():
            label.setText(f"● {_SERVICE_NAMES[service]} — ожидание")
            label.setStyleSheet("color: #64748b;")
            label.setToolTip("ещё не запускалось")

    def _toggle_live(self) -> None:
        if self.worker is not None:
            self.start_button.setText("Останавливаю…")
            self.start_button.setEnabled(False)
            self.worker.request_stop()
            return
        if self.batch_worker is not None:
            self.error_label.setText("Сначала дождитесь завершения пакетного дубляжа.")
            return
        self._reset_live_status()
        cfg = self._collect_cfg()
        self.worker = PipelineWorker(
            cfg,
            on_subtitle=self.subtitleReceived.emit,
            on_status=self.statusChanged.emit,
            on_metrics=self.metricsUpdated.emit,
            on_finished=self.pipelineFinished.emit,
        )
        self.worker.start()
        self.start_button.setText("Остановить Live перевод")
        self.profile_combo.setEnabled(False)
        self.workflows.setTabEnabled(1, False)

    def _pipeline_finished(self) -> None:
        self.worker = None
        self.start_button.setEnabled(True)
        self.start_button.setText("Запустить Live перевод")
        self.profile_combo.setEnabled(self.batch_worker is None)
        self.workflows.setTabEnabled(1, self.batch_worker is None)
        self.overlay.hide()

    def _show_subtitle(self, event: SubtitleEvent) -> None:
        self.subtitles_view.appendPlainText(f"[{event.language}] {event.original}")
        self.subtitles_view.appendPlainText(f"[{event.target_lang}] {event.translated}\n")
        if self.overlay_check.isChecked():
            self.overlay.show_subtitle(event)

    def _show_status(self, status: ServiceStatus) -> None:
        if status.service == "batch":
            self.batch_stage.setText(status.detail or status.state)
            if status.state == "error":
                self.batch_error.setText(f"Пакетный дубляж: {status.detail or 'неизвестная ошибка'}")
                self.batch_error.show()
            return

        label = self.status_labels.get(status.service)
        display_name = _SERVICE_NAMES.get(status.service, "Конвейер")
        color = _STATE_COLORS.get(status.state, "#b45309")
        detail = status.detail.strip()
        if label is not None:
            label.setText(f"● {display_name} — {status.state}")
            label.setStyleSheet(f"color: {color};")
            label.setToolTip(detail or status.state)
        if status.state == "error":
            self.error_count += 1
            suffix = f": {detail}" if detail else ""
            self.error_label.setText(f"Ошибок стадий: {self.error_count}. {display_name}{suffix}")

    def _show_metrics(self, payload: object) -> None:
        if isinstance(payload, str):  # совместимость с прежним PipelineWorker
            line, counters = payload, {}
        elif isinstance(payload, dict):
            line = str(payload.get("latency") or "")
            raw_counters = payload.get("counters") or {}
            counters = raw_counters if isinstance(raw_counters, dict) else {}
        else:
            line, counters = "", {}
        self.metrics_label.setText(f"Задержка p50: {line or '—'}")
        # Metrics хранит один и тот же drop на трёх уровнях:
        # ``dropped``, ``dropped.tts`` и ``dropped.tts.superseded``. Для
        # пользователю понятного total берём верхний счётчик, а в расшифровке
        # показываем только стадии, не суммируя причины повторно.
        total = int(counters.get("dropped", 0) or 0)
        by_stage = {
            str(key): int(value)
            for key, value in counters.items()
            if str(key).startswith("dropped.")
            and str(key).count(".") == 1
            and isinstance(value, (int, float))
            and value
        }
        if not total and by_stage:
            # Совместимость с внешним Metrics, который умеет только стадии.
            total = sum(by_stage.values())
        if not total:
            self.drop_label.setText("Пропущено: 0")
            return
        breakdown = ", ".join(f"{key}: {value}" for key, value in by_stage.items())
        self.drop_label.setText(
            f"Пропущено: {total}" + (f" ({breakdown})" if breakdown else "")
        )

    def _choose_batch_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео или аудио",
            "",
            "Медиа (*.mp4 *.mkv *.mov *.webm *.mp3 *.wav *.m4a *.flac);;Все файлы (*)",
        )
        if path:
            self.batch_source.setText(path)

    def _choose_batch_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Куда сохранить дубляж",
            self.batch_output.text(),
            "Видео MKV (*.mkv);;Аудио WAV (*.wav);;Все файлы (*)",
        )
        if path:
            self.batch_output.setText(path)

    def _start_batch(self) -> None:
        if self.worker is not None:
            self.batch_error.setText("Сначала остановите Live перевод.")
            self.batch_error.show()
            return
        source = self.batch_source.text().strip()
        if not source:
            self.batch_error.setText("Выберите локальный файл или вставьте URL ролика.")
            self.batch_error.show()
            return
        self.batch_error.hide()
        self.batch_start_button.setEnabled(False)
        self.batch_source.setEnabled(False)
        self.batch_output.setEnabled(False)
        self.batch_source_browse.setEnabled(False)
        self.batch_output_browse.setEnabled(False)
        self.profile_combo.setEnabled(False)
        self.workflows.setTabEnabled(0, False)
        self.batch_stage.setText("Запускаю подготовку…")
        self.batch_worker = BatchWorker(
            self._collect_cfg(),
            source,
            self.batch_output.text().strip() or None,
            self.duck_spin.value(),
            self.keep_original_check.isChecked(),
            on_status=self.statusChanged.emit,
            on_finished=self.batchFinished.emit,
        )
        self.batch_worker.start()

    def _batch_finished(self, result: object) -> None:
        output, error = result if isinstance(result, tuple) else (None, "неизвестный результат")
        self.batch_worker = None
        self.batch_start_button.setEnabled(True)
        self.batch_source.setEnabled(True)
        self.batch_output.setEnabled(True)
        self.batch_source_browse.setEnabled(True)
        self.batch_output_browse.setEnabled(True)
        self.profile_combo.setEnabled(self.worker is None)
        self.workflows.setTabEnabled(0, self.worker is None)
        if error:
            self.batch_stage.setText("Пакетный дубляж не завершился")
            return
        self.batch_stage.setText(f"Готово: {output}")
        self.batch_output.setText(str(output))

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.join(timeout=2)
        self.overlay.close()
        super().closeEvent(event)


def run_gui(cfg: AppConfig, profile_name: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("UVT")
    window = MainWindow(cfg, profile_name=profile_name)
    window.show()
    return app.exec()
