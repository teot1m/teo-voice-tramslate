"""Оверлей субтитров: полупрозрачное окно без рамки поверх всех приложений.

Перетаскивается мышью; прозрачность, цвет, размер и позиция — из конфига
(ТЗ §8). Скрывается сам, когда реплика «отговорила».
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
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        alpha = int(max(0.0, min(1.0, cfg.opacity)) * 255)
        frame = QFrame(self)
        frame.setStyleSheet(
            f"QFrame {{ background: rgba(0, 0, 0, {alpha}); border-radius: 10px; }}"
        )

        self.original_label = QLabel("", frame)
        self.original_label.setWordWrap(True)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setStyleSheet(
            f"color: #bbbbbb; font-size: {max(10, cfg.font_size - 5)}px; background: transparent;"
        )

        self.translated_label = QLabel("", frame)
        self.translated_label.setWordWrap(True)
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translated_label.setStyleSheet(
            f"color: {cfg.color}; font-size: {cfg.font_size}px; font-weight: 600; background: transparent;"
        )

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(16, 10, 16, 10)
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

        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.6)
        self.resize(width, 10)
        x = screen.x() + (screen.width() - width) // 2
        y = screen.y() + (80 if cfg.position == "top" else screen.height() - 180)
        self.move(x, y)

    def show_subtitle(self, event: SubtitleEvent) -> None:
        self.original_label.setText(event.original)
        self.translated_label.setText(event.translated)
        self.adjustSize()
        self.show()
        self.raise_()
        duration_ms = max(2500, 55 * len(event.translated))
        self._hide_timer.start(duration_ms)

    # Перетаскивание мышью
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
