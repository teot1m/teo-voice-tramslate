"""Ненавязчивый оверлей субтитров поверх приложений.

Оверлей не забирает фокус у видео/звонка, остаётся в границах экрана и можно
перетащить за любую область. Правый клик или двойной клик скрывает его до
следующей реплики.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from uvt.config import OverlayConfig
from uvt.events import SubtitleEvent


class OverlayWindow(QWidget):
    def __init__(self, cfg: OverlayConfig) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.cfg = cfg
        self.setWindowTitle("UVT — субтитры")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAccessibleName("UVT subtitles overlay")

        alpha = int(max(0.0, min(1.0, cfg.opacity)) * 255)
        frame = QFrame(self)
        # Дочерние labels не должны перехватывать drag-события у окна.
        frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        frame.setStyleSheet(
            f"QFrame {{ background: rgba(0, 0, 0, {alpha}); border-radius: 10px; }}"
        )

        self.original_label = QLabel("", frame)
        self.original_label.setObjectName("uvtOriginalSubtitle")
        self.original_label.setWordWrap(True)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setStyleSheet(
            f"color: #cbd5e1; font-size: {max(10, cfg.font_size - 5)}px; "
            "background: transparent;"
        )

        self.translated_label = QLabel("", frame)
        self.translated_label.setObjectName("uvtTranslatedSubtitle")
        self.translated_label.setWordWrap(True)
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translated_label.setStyleSheet(
            f"color: {cfg.color}; font-size: {max(10, cfg.font_size)}px; font-weight: 600; "
            "background: transparent;"
        )

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(20, 12, 20, 12)
        inner.setSpacing(4)
        if cfg.show in ("original", "both"):
            inner.addWidget(self.original_label)
        if cfg.show in ("translation", "both"):
            inner.addWidget(self.translated_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._drag_offset = None
        self._place_initially()

    def _available_geometry(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else self.geometry()

    def _overlay_width(self) -> int:
        area = self._available_geometry()
        available = max(240, area.width() - 32)
        return min(available, max(300, min(1080, int(area.width() * 0.72))))

    def _place_initially(self) -> None:
        area = self._available_geometry()
        width = self._overlay_width()
        self.resize(width, 10)
        x = area.x() + (area.width() - width) // 2
        y = area.y() + (80 if self.cfg.position == "top" else area.height() - 180)
        self.move(x, y)

    def _keep_on_screen(self) -> None:
        area = self._available_geometry()
        width = min(self._overlay_width(), max(1, area.width()))
        if self.width() != width:
            self.resize(width, self.height())
        max_x = area.right() - self.width() + 1
        max_y = area.bottom() - self.height() + 1
        self.move(
            min(max(self.x(), area.x()), max_x),
            min(max(self.y(), area.y()), max_y),
        )

    def show_subtitle(self, event: SubtitleEvent) -> None:
        self._hide_timer.stop()
        self.original_label.setText(event.original)
        self.translated_label.setText(event.translated)
        # Сначала фиксируем безопасную ширину, затем Qt вычисляет высоту
        # word-wrap. Так длинная фраза не уезжает за край экрана.
        self.resize(self._overlay_width(), 10)
        self.adjustSize()
        self.resize(self._overlay_width(), max(self.sizeHint().height(), self.minimumSizeHint().height()))
        self._keep_on_screen()
        self.show()
        self.raise_()

        # В live событие пока не содержит фактических TTS start/end. Используем
        # длительность исходной фразы и умеренный reading-rate fallback, а не
        # только число символов; верхняя граница не блокирует следующий текст.
        speech_s = max(0.0, event.end_ts - event.start_ts)
        reading_s = len(event.translated) / 16.0
        duration_ms = int(min(10_000, max(2_500, max(speech_s, reading_s) * 1_000)))
        self._hide_timer.start(duration_ms)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.hide()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._keep_on_screen()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.hide()
        event.accept()
