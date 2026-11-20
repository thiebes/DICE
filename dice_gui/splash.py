"""
Splash screen for DICE GUI.

Shows loading progress while heavy modules are imported.
Uses a plain QWidget instead of QSplashScreen to avoid the ~1s
overhead that QSplashScreen.show() incurs on Windows.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QColor


class DiceSplashScreen(QWidget):
    """Splash screen with progress bar for DICE GUI."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 200)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
        )

        self._progress = 0
        self._message = "Starting..."

    def paintEvent(self, event):
        """Draw splash screen contents."""
        painter = QPainter(self)

        # Background
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        # Title
        painter.setPen(QColor("#ffffff"))
        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(self.rect().adjusted(0, 30, 0, 0), Qt.AlignmentFlag.AlignHCenter, "DICE")

        # Subtitle
        subtitle_font = QFont("Arial", 10)
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(self.rect().adjusted(0, 65, 0, 0), Qt.AlignmentFlag.AlignHCenter,
                        "Diffusion Insight Computation Engine")

        # Progress bar background
        bar_rect = self.rect().adjusted(40, 120, -40, -50)
        painter.fillRect(bar_rect, QColor("#333355"))

        # Progress bar fill
        if self._progress > 0:
            fill_width = int(bar_rect.width() * self._progress / 100)
            fill_rect = bar_rect.adjusted(0, 0, fill_width - bar_rect.width(), 0)
            painter.fillRect(fill_rect, QColor("#4a7c59"))

        # Progress text
        painter.setPen(QColor("#ffffff"))
        status_font = QFont("Arial", 9)
        painter.setFont(status_font)
        painter.drawText(self.rect().adjusted(0, 150, 0, 0), Qt.AlignmentFlag.AlignHCenter, self._message)

        painter.end()

    def set_progress(self, value: int, message: str = ""):
        """Update progress bar and message."""
        self._progress = min(100, max(0, value))
        if message:
            self._message = message
        self.repaint()
        # Process events to keep UI responsive
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
