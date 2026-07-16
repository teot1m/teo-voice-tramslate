"""Главное окно (ТЗ §17): источник, языки, движки, START и панель статусов.

Конвейер крутится в фоновом потоке со своим asyncio-циклом; в GUI события
приходят через Qt-сигналы (queued connection — потокобезопасно).
"""
from __future__ import annotations

import asyncio
import logging
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from uvt import registry
from uvt.config import AppConfig
from uvt.events import TOPIC_STATUS, ServiceStatus, SubtitleEvent
from uvt.gui.overlay_window import OverlayWindow

log = logging.getLogger("uvt.gui")

_LANGS = ["auto", "ru", "en", "uk", "de", "fr", "es", "it", "pt", "ja", "zh", "ko"]
_STATE_COLORS = {
    "running": "#2ecc71", "done": "#95a5a6", "stopped": "#95a5a6",
    "error": "#e74c3c",
}


class PipelineWorker(threading.Thread):
    """Фоновый поток с asyncio-циклом конвейера."""

    def __init__(self, cfg: AppConfig, on_subtitle, on_status, on_metrics) -> None:
        super().__init__(daemon=True, name="uvt-pipeline")
        self.cfg = cfg
        self.on_subtitle = on_subtitle
        self.on_status = on_status
        self.on_metrics = on_metrics
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception:  # noqa: BLE001
            log.exception("конвейер аварийно завершился")

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
                self.on_metrics(pipeline.metrics.format_line())

        watchers = [asyncio.create_task(watch_status()), asyncio.create_task(tick_metrics())]
        await pipeline.start()
        try:
            await self._stop_event.wait()
        finally:
            for task in watchers:
                task.cancel()
            await asyncio.gather(*watchers, return_exceptions=True)
            await pipeline.stop()

    def request_stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)


class MainWindow(QMainWindow):
    subtitleReceived = Signal(object)
    statusChanged = Signal(object)
    metricsUpdated = Signal(str)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.base_cfg = cfg
        self.worker: PipelineWorker | None = None
        self.overlay = OverlayWindow(cfg.overlay)
        self.setWindowTitle("UVT — Universal Voice Translator")
        self._build_ui()
        self.subtitleReceived.connect(self._show_subtitle)
        self.statusChanged.connect(self._show_status)
        self.metricsUpdated.connect(self._show_metrics)

    def _build_ui(self) -> None:
        registry.load_builtins()

        self.source_combo = QComboBox()
        self.source_combo.addItem("Системный звук (по умолчанию)", None)
        try:
            import sounddevice as sd

            for index, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    self.source_combo.addItem(f"{dev['name']}", index)
        except Exception:  # noqa: BLE001
            log.exception("не удалось перечислить устройства")

        def combo(kind: str, current: str) -> QComboBox:
            box = QComboBox()
            for name in registry.available(kind):
                box.addItem(name)
            box.setCurrentText(current)
            return box

        self.source_lang = QComboBox(); self.source_lang.addItems(_LANGS)
        self.source_lang.setCurrentText(self.base_cfg.source_lang)
        self.target_lang = QComboBox(); self.target_lang.addItems([l for l in _LANGS if l != "auto"])
        self.target_lang.setCurrentText(self.base_cfg.target_lang)
        self.mode_combo = QComboBox(); self.mode_combo.addItems(["voiceover", "subtitles", "replace", "dual"])
        self.mode_combo.setCurrentText(self.base_cfg.mode)
        self.stt_combo = combo("stt", self.base_cfg.stt.engine)
        self.translate_combo = combo("translation", self.base_cfg.translation.engine)
        self.tts_combo = combo("tts", self.base_cfg.tts.engine)
        self.overlay_check = QCheckBox("Оверлей субтитров")
        self.overlay_check.setChecked(self.base_cfg.overlay.enabled)

        form = QFormLayout()
        form.addRow("Источник:", self.source_combo)
        form.addRow("Язык:", self.source_lang)
        form.addRow("Перевод на:", self.target_lang)
        form.addRow("Режим:", self.mode_combo)
        form.addRow("STT:", self.stt_combo)
        form.addRow("Перевод:", self.translate_combo)
        form.addRow("TTS:", self.tts_combo)
        form.addRow("", self.overlay_check)

        self.start_button = QPushButton("START")
        self.start_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self._toggle)

        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(self.start_button)
        left.addStretch(1)

        # Правая панель: статусы сервисов и задержка (ТЗ §17)
        self.status_labels: dict[str, QLabel] = {}
        right = QVBoxLayout()
        right.addWidget(QLabel("Статус"))
        for service in ("capture", "vad", "stt", "translate", "tts", "output"):
            label = QLabel(f"● {service}")
            label.setStyleSheet("color: #95a5a6;")
            self.status_labels[service] = label
            right.addWidget(label)
        self.metrics_label = QLabel("Задержка: —")
        self.metrics_label.setWordWrap(True)
        right.addWidget(self.metrics_label)
        right.addStretch(1)

        self.subtitles_view = QPlainTextEdit()
        self.subtitles_view.setReadOnly(True)
        self.subtitles_view.setPlaceholderText("Здесь появятся распознанные и переведённые реплики…")

        top = QHBoxLayout()
        top.addLayout(left, 3)
        top.addLayout(right, 2)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.subtitles_view, 1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self.resize(720, 560)

    def _collect_cfg(self) -> AppConfig:
        cfg = self.base_cfg.model_copy(deep=True)
        cfg.capture.device = self.source_combo.currentData()
        cfg.source_lang = self.source_lang.currentText()
        cfg.target_lang = self.target_lang.currentText()
        cfg.mode = self.mode_combo.currentText()
        cfg.stt.engine = self.stt_combo.currentText()
        cfg.translation.engine = self.translate_combo.currentText()
        cfg.tts.engine = self.tts_combo.currentText()
        cfg.overlay.enabled = self.overlay_check.isChecked()
        return cfg

    def _toggle(self) -> None:
        if self.worker is None:
            cfg = self._collect_cfg()
            self.worker = PipelineWorker(
                cfg,
                on_subtitle=self.subtitleReceived.emit,
                on_status=self.statusChanged.emit,
                on_metrics=self.metricsUpdated.emit,
            )
            self.worker.start()
            self.start_button.setText("STOP")
        else:
            self.worker.request_stop()
            self.worker.join(timeout=15)
            self.worker = None
            self.start_button.setText("START")
            self.overlay.hide()
            for label in self.status_labels.values():
                label.setStyleSheet("color: #95a5a6;")

    def _show_subtitle(self, event: SubtitleEvent) -> None:
        self.subtitles_view.appendPlainText(f"[{event.language}] {event.original}")
        self.subtitles_view.appendPlainText(f"[{event.target_lang}] {event.translated}\n")
        if self.overlay_check.isChecked():
            self.overlay.show_subtitle(event)

    def _show_status(self, status: ServiceStatus) -> None:
        label = self.status_labels.get(status.service)
        if label is None:
            return
        color = _STATE_COLORS.get(status.state, "#f1c40f")
        label.setStyleSheet(f"color: {color};")
        tooltip = f"{status.state}: {status.detail}" if status.detail else status.state
        label.setToolTip(tooltip)

    def _show_metrics(self, line: str) -> None:
        self.metrics_label.setText(f"Задержка: {line or '—'}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.join(timeout=15)
        self.overlay.close()
        super().closeEvent(event)


def run_gui(cfg: AppConfig) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("UVT")
    window = MainWindow(cfg)
    window.show()
    return app.exec()
