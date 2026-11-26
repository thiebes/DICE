"""
Global Units Dialog

Modal dialog for selecting global length and time units.
"""

from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QDialogButtonBox,
)

from dice_gui.tabs.base import LENGTH_UNITS_DISPLAY, TIME_UNITS_DISPLAY


class GlobalUnitsDialog(QDialog):
    """Dialog for selecting global length and time units."""

    def __init__(self, parent=None, current_length="micrometer",
                 current_time="nanosecond"):
        super().__init__(parent)
        self.setWindowTitle("Global Units")
        self.setMinimumWidth(300)

        layout = QFormLayout(self)

        self.length_combo = QComboBox()
        self.length_combo.addItems(LENGTH_UNITS_DISPLAY)
        self.length_combo.setCurrentText(current_length)
        layout.addRow("Length Unit:", self.length_combo)

        self.time_combo = QComboBox()
        self.time_combo.addItems(TIME_UNITS_DISPLAY)
        self.time_combo.setCurrentText(current_time)
        layout.addRow("Time Unit:", self.time_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def selected_length_unit(self) -> str:
        return self.length_combo.currentText()

    def selected_time_unit(self) -> str:
        return self.time_combo.currentText()
