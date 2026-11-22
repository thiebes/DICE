"""
Histogram of the estimates with the part near the nominal value highlighted.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPalette, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class ProximityHistogramWidget(QWidget):
    """
    Histogram of D estimate / nominal D that highlights the accuracy threshold.

    The bars show how the estimates of a simulation are distributed around
    the nominal value, at ratio 1. The part of each bar that lies within the
    threshold of the nominal value (from 1 - proximity to 1 + proximity) is
    drawn in the highlight color, so the highlighted share of the bars is the
    fraction of estimates within the threshold.

    The axis is centered on the nominal value and spans the estimates, so
    it does not move when the threshold changes. The rare estimates beyond
    the 99.5th percentile of the distance from 1 are counted in a note
    instead of stretching the axis.
    """

    MIN_WIDTH, MIN_HEIGHT = 320, 210
    LEFT_MARGIN, RIGHT_MARGIN = 14, 14
    TOP_MARGIN, BOTTOM_MARGIN = 22, 40
    MIN_BINS, MAX_BINS = 10, 60
    PLACEHOLDER = "Run a simulation or load results to see the distribution of the estimates."

    def __init__(self, proximity: float = 0.10, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._proximity = proximity
        self._ratios = np.array([], dtype=float)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Histogram of the estimates around the nominal value")
        self._describe()

    # ---- data ------------------------------------------------------------------

    def set_proximity(self, proximity: float) -> None:
        """Set the accuracy threshold (0.1 for within 10% of nominal) and redraw."""
        self._proximity = proximity
        self._describe()
        self.update()

    def set_ratios(self, ratios: Optional[Sequence[float]]) -> None:
        """Set the estimates as D estimate / nominal D; None or an empty sequence clears the histogram."""
        values = np.asarray([] if ratios is None else ratios, dtype=float)
        self._ratios = values[np.isfinite(values)]
        self._describe()
        self.update()

    def ratios(self) -> np.ndarray:
        """The finite ratios the histogram shows or counts."""
        return self._ratios

    def axis_range(self) -> Optional[Tuple[float, float]]:
        """The ratio at the left and right end of the axis; None without estimates."""
        if self._ratios.size == 0:
            return None
        distance = np.abs(self._ratios - 1.0)
        half_span = max(float(np.percentile(distance, 99.5)), 1e-9) * 1.08
        return 1.0 - half_span, 1.0 + half_span

    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Counts and bin edges of the ratios inside the axis range."""
        low, high = self.axis_range()
        inside = self._ratios[(self._ratios >= low) & (self._ratios <= high)]
        automatic = len(np.histogram_bin_edges(inside, bins='auto', range=(low, high))) - 1
        bin_count = int(min(max(automatic, self.MIN_BINS), self.MAX_BINS))
        return np.histogram(inside, bins=bin_count, range=(low, high))

    def count_outside_axis(self) -> int:
        """Number of estimates beyond the ends of the axis."""
        if self._ratios.size == 0:
            return 0
        low, high = self.axis_range()
        return int(np.count_nonzero((self._ratios < low) | (self._ratios > high)))

    # ---- geometry --------------------------------------------------------------

    def plot_rect(self) -> QRect:
        """The area the bars are drawn in, in widget pixels."""
        return QRect(
            self.LEFT_MARGIN, self.TOP_MARGIN,
            max(self.width() - self.LEFT_MARGIN - self.RIGHT_MARGIN, 1),
            max(self.height() - self.TOP_MARGIN - self.BOTTOM_MARGIN, 1),
        )

    def x_position(self, ratio: float) -> int:
        """Pixel column of a ratio on the axis."""
        low, high = self.axis_range()
        plot = self.plot_rect()
        return plot.left() + round((ratio - low) / (high - low) * plot.width())

    def band_rect(self) -> QRect:
        """The part of the plot area within the threshold of the nominal value."""
        plot = self.plot_rect()
        left = max(self.x_position(1.0 - self._proximity), plot.left())
        right = min(self.x_position(1.0 + self._proximity), plot.left() + plot.width())
        return QRect(left, plot.top(), max(right - left, 0), plot.height())

    def bar_rects(self) -> List[QRect]:
        """One rectangle for each bin that holds estimates, with its foot on the axis."""
        counts, edges = self.histogram()
        plot = self.plot_rect()
        peak = counts.max() if counts.size else 0
        if peak == 0:
            return []
        columns = [self.x_position(edge) for edge in edges]
        bottom = plot.top() + plot.height()
        bars = []
        for count, left, right in zip(counts, columns[:-1], columns[1:]):
            if count == 0:
                continue
            # One pixel between bars, when the bars are wide enough to spare it.
            width = right - left - (1 if right - left > 3 else 0)
            height = max(round(count / peak * plot.height()), 1)
            bars.append(QRect(left, bottom - height, max(width, 1), height))
        return bars

    # ---- drawing ---------------------------------------------------------------

    def bar_colors(self) -> Tuple[QColor, QColor]:
        """The color of the bars within the threshold, and of the rest of the bars.

        Both are opaque and stand out from the window background by at
        least 3:1 in either theme.
        """
        dark_mode = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        if dark_mode:
            return QColor(100, 181, 246), QColor(120, 120, 120)
        return QColor(13, 71, 161), QColor(140, 140, 140)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        group = QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled
        text_color = self.palette().color(group, QPalette.ColorRole.Text)
        font = self.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(text_color)

        if self._ratios.size == 0 or not self.isEnabled():
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.PLACEHOLDER)
            return

        highlight, muted = self.bar_colors()
        plot, band = self.plot_rect(), self.band_rect()
        bottom = plot.top() + plot.height()
        low, high = self.axis_range()

        # The threshold band behind the bars.
        tint = QColor(highlight)
        tint.setAlpha(34)
        painter.fillRect(band, tint)

        # Every bar in the muted color, then the part inside the band in the
        # highlight color. Without antialiasing, the bar edges are exact.
        painter.setPen(Qt.PenStyle.NoPen)
        bars = self.bar_rects()
        for bar in bars:
            painter.fillRect(bar, muted)
        painter.save()
        painter.setClipRect(band)
        for bar in bars:
            painter.fillRect(bar, highlight)
        painter.restore()

        # Limits of the threshold, where they fall on the axis.
        painter.setPen(QPen(text_color, 1, Qt.PenStyle.DashLine))
        for limit in (1.0 - self._proximity, 1.0 + self._proximity):
            if low <= limit <= high:
                column = self.x_position(limit)
                painter.drawLine(column, plot.top() - 4, column, bottom)

        # Label of the band, above the plot and centered on the nominal value.
        painter.setPen(text_color)
        center = self.x_position(1.0)
        percent = int(round(self._proximity * 100))
        painter.drawText(QRect(center - 60, 2, 120, self.TOP_MARGIN - 6),
                         Qt.AlignmentFlag.AlignCenter, f"within ±{percent}%")

        outside = self.count_outside_axis()
        if outside:
            painter.drawText(QRect(plot.left(), 2, plot.width(), self.TOP_MARGIN - 6),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{outside} beyond this range")

        # Axis, ticks at the ends and at the nominal value, and the axis title.
        painter.setPen(QPen(text_color, 1))
        painter.drawLine(plot.left(), bottom, plot.left() + plot.width(), bottom)
        ticks = (
            (plot.left(), f"{low:.3g}", Qt.AlignmentFlag.AlignLeft),
            (center, "1", Qt.AlignmentFlag.AlignHCenter),
            (plot.left() + plot.width(), f"{high:.3g}", Qt.AlignmentFlag.AlignRight),
        )
        for column, label, alignment in ticks:
            painter.drawLine(column, bottom, column, bottom + 4)
            left = {Qt.AlignmentFlag.AlignLeft: column, Qt.AlignmentFlag.AlignHCenter: column - 40,
                    Qt.AlignmentFlag.AlignRight: column - 80}[alignment]
            painter.drawText(QRect(left, bottom + 5, 80, 14), alignment | Qt.AlignmentFlag.AlignTop, label)
        painter.drawText(QRect(plot.left(), bottom + 20, plot.width(), 16),
                         Qt.AlignmentFlag.AlignCenter, "D estimate / nominal D")

    def _describe(self) -> None:
        """Keep the accessible description in step with what the histogram shows."""
        percent = int(round(self._proximity * 100))
        if self._ratios.size == 0:
            self.setAccessibleDescription("No estimates to show.")
            return
        within = np.count_nonzero(np.abs(self._ratios - 1.0) <= self._proximity)
        self.setAccessibleDescription(
            f"Histogram of {self._ratios.size} estimates of the diffusion coefficient, as ratios to the "
            f"nominal value. {within} of them are within ±{percent}% of the nominal value and are highlighted."
        )
