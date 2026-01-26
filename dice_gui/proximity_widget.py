"""
Custom widget for visualizing proximity level as a target graphic.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QPalette


class ProximityTargetWidget(QWidget):
    """
    Widget that displays proximity level as a bullseye target.

    The outer circle represents 100% of the nominal value (the true value).
    The inner highlighted ring shows the proximity tolerance band.
    For example, proximity=0.1 shows a ring at 10% of the radius from center.
    """

    def __init__(self, proximity: float = 0.10, parent=None):
        super().__init__(parent)
        self._proximity = proximity
        self.setMinimumSize(150, 230)
        self.setMaximumSize(200, 280)

    def set_proximity(self, proximity: float):
        """Update the proximity value and redraw."""
        self._proximity = proximity
        self.update()

    def paintEvent(self, event):
        """Draw the target graphic."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get widget dimensions
        width = self.width()
        height = self.height()

        # Use width for the target size since we need space below for the scale
        target_size = min(width, 150)

        # Calculate center and radius - position target in upper portion
        center_x = width / 2
        center_y = target_size / 2 + 10  # Position near top with margin
        outer_radius = target_size / 2 - 10  # Leave margin for border

        # Get colors from palette
        palette = self.palette()

        # Detect dark mode
        window_color = palette.color(QPalette.ColorRole.Window)
        is_dark_mode = window_color.lightness() < 128

        if is_dark_mode:
            target_color = QColor(100, 181, 246)  # Light blue for dark mode
            ring_color = QColor(129, 199, 132, 100)  # Semi-transparent green
            grid_color = QColor(160, 160, 160, 80)  # Light gray
        else:
            target_color = QColor(13, 71, 161)  # Dark blue for light mode
            ring_color = QColor(27, 94, 32, 100)  # Semi-transparent dark green
            grid_color = QColor(97, 97, 97, 80)  # Dark gray

        # Draw outer circle (100% - the bullseye)
        pen = QPen(target_color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        outer_rect = QRectF(
            center_x - outer_radius,
            center_y - outer_radius,
            outer_radius * 2,
            outer_radius * 2
        )
        painter.drawEllipse(outer_rect)

        # Draw center point (nominal value)
        painter.setBrush(target_color)
        painter.drawEllipse(
            QRectF(center_x - 3, center_y - 3, 6, 6)
        )

        # Draw proximity ring
        proximity_radius = outer_radius * self._proximity

        pen = QPen(ring_color, 3)
        painter.setPen(pen)
        painter.setBrush(ring_color)

        # Draw the proximity tolerance band
        proximity_rect = QRectF(
            center_x - proximity_radius,
            center_y - proximity_radius,
            proximity_radius * 2,
            proximity_radius * 2
        )
        painter.drawEllipse(proximity_rect)

        # Draw reference rings at 0.25, 0.5, 0.75 for scale
        pen = QPen(grid_color, 1, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for fraction in [0.25, 0.5, 0.75]:
            ref_radius = outer_radius * fraction
            ref_rect = QRectF(
                center_x - ref_radius,
                center_y - ref_radius,
                ref_radius * 2,
                ref_radius * 2
            )
            painter.drawEllipse(ref_rect)

        # Draw center crosshair
        pen = QPen(grid_color, 1)
        painter.setPen(pen)
        crosshair_size = 8
        painter.drawLine(
            int(center_x - crosshair_size), int(center_y),
            int(center_x + crosshair_size), int(center_y)
        )
        painter.drawLine(
            int(center_x), int(center_y - crosshair_size),
            int(center_x), int(center_y + crosshair_size)
        )

        # Draw scale below the target
        scale_top = center_y + outer_radius + 20
        scale_left = center_x - outer_radius
        scale_right = center_x + outer_radius
        scale_width = outer_radius * 2

        # Draw horizontal axis line
        pen = QPen(palette.color(QPalette.ColorRole.Text), 2)
        painter.setPen(pen)
        painter.drawLine(
            int(scale_left), int(scale_top),
            int(scale_right), int(scale_top)
        )

        # Draw tick marks and labels (ticks extend downward only)
        text_color = palette.color(QPalette.ColorRole.Text)
        pen = QPen(text_color, 1)
        painter.setPen(pen)
        font = self.font()
        font.setPointSize(8)
        painter.setFont(font)

        tick_length = 5

        # Left tick (0) - pointing upward
        painter.drawLine(int(scale_left), int(scale_top - tick_length), int(scale_left), int(scale_top))
        painter.drawText(QRectF(scale_left - 15, scale_top + 3, 30, 15),
                        Qt.AlignmentFlag.AlignCenter, "0")

        # Center tick (D_nom) - pointing upward
        painter.drawLine(int(center_x), int(scale_top - tick_length), int(center_x), int(scale_top))
        painter.drawText(QRectF(center_x - 20, scale_top + 3, 40, 15),
                        Qt.AlignmentFlag.AlignCenter, "D_nom")

        # Right tick (2×D_nom) - pointing upward
        painter.drawLine(int(scale_right), int(scale_top - tick_length), int(scale_right), int(scale_top))
        painter.drawText(QRectF(scale_right - 25, scale_top + 3, 50, 15),
                        Qt.AlignmentFlag.AlignCenter, "2×D_nom")

        # Draw proximity tolerance bar
        # The bar spans from (1-proximity)*D_nom to (1+proximity)*D_nom
        # On the scale from 0 to 2×D_nom, D_nom is at 0.5 (center)
        # So (1-proximity)*D_nom is at position 0.5*(1-proximity)
        # And (1+proximity)*D_nom is at position 0.5*(1+proximity)

        bar_left_fraction = 0.5 * (1 - self._proximity)
        bar_right_fraction = 0.5 * (1 + self._proximity)

        bar_left = scale_left + scale_width * bar_left_fraction
        bar_right = scale_left + scale_width * bar_right_fraction
        bar_height = 8
        bar_top = scale_top - 15

        # Draw the tolerance bar
        painter.setBrush(ring_color)
        pen = QPen(ring_color, 2)
        painter.setPen(pen)
        painter.drawRect(QRectF(bar_left, bar_top, bar_right - bar_left, bar_height))

        # Draw percentage label below scale
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(text_color)
        percentage_text = f"±{self._proximity * 100:.1f}%"
        text_rect = QRectF(0, scale_top + 25, width, 20)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, percentage_text)
