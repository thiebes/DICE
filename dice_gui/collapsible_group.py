"""Collapsible group box widget for progressive disclosure."""

from PyQt6.QtWidgets import (
    QGroupBox, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSizePolicy, QFrame
)
from PyQt6.QtCore import (
    QPropertyAnimation, QEasingCurve, QSettings,
    pyqtProperty, QParallelAnimationGroup, Qt
)
from PyQt6.QtGui import QIcon


class CollapsibleGroupBox(QGroupBox):
    """
    QGroupBox that can be collapsed/expanded.

    Collapse state is persisted via QSettings when settings_key is provided.
    """

    def __init__(self, title: str, settings_key: str = None,
                 initially_collapsed: bool = False, parent=None):
        super().__init__(title, parent)
        self._settings_key = settings_key
        self._content_widget = None
        self._collapsed = initially_collapsed
        self._animation_duration = 150

        # Load saved state from QSettings if available
        if settings_key:
            settings = QSettings("DICE", "DICE_GUI")
            saved_state = settings.value(f"collapsed/{settings_key}")
            if saved_state is not None:
                self._collapsed = saved_state == "true" or saved_state is True

        self._setup_ui()

    def _setup_ui(self):
        """Setup the collapsible UI structure."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Header with toggle button
        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        self._toggle_button = QPushButton()
        self._toggle_button.setFlat(True)
        self._toggle_button.setFixedSize(20, 20)
        self._toggle_button.clicked.connect(self.toggle_collapsed)
        self._update_toggle_icon()

        header_layout.addWidget(self._toggle_button)
        header_layout.addStretch()

        self._main_layout.addWidget(self._header)

        # Content container
        self._content_container = QFrame()
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(12, 8, 12, 8)
        self._main_layout.addWidget(self._content_container)

        # Apply initial state
        if self._collapsed:
            self._content_container.setMaximumHeight(0)
            self._content_container.setVisible(False)

    def _update_toggle_icon(self):
        """Update the toggle button icon based on state."""
        if self._collapsed:
            self._toggle_button.setText("▶")
        else:
            self._toggle_button.setText("▼")

    def set_content_widget(self, widget: QWidget):
        """Set the collapsible content widget."""
        if self._content_widget:
            self._content_layout.removeWidget(self._content_widget)

        self._content_widget = widget
        self._content_layout.addWidget(widget)

    def set_content_layout(self, layout):
        """Set the content layout directly."""
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Create a widget to hold the layout
        content = QWidget()
        content.setLayout(layout)
        self._content_widget = content
        self._content_layout.addWidget(content)

    def toggle_collapsed(self):
        """Toggle collapse state with animation."""
        self._collapsed = not self._collapsed
        self._animate_collapse()
        self._save_state()
        self._update_toggle_icon()

    def _animate_collapse(self):
        """Animate height change."""
        if self._content_container is None:
            return

        if self._collapsed:
            # Collapse
            self._content_container.setVisible(True)
            start_height = self._content_container.sizeHint().height()
            end_height = 0
        else:
            # Expand
            self._content_container.setVisible(True)
            self._content_container.setMaximumHeight(16777215)  # Reset max
            start_height = 0
            end_height = self._content_container.sizeHint().height()

        animation = QPropertyAnimation(self._content_container, b"maximumHeight")
        animation.setDuration(self._animation_duration)
        animation.setStartValue(start_height)
        animation.setEndValue(end_height)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        if self._collapsed:
            animation.finished.connect(
                lambda: self._content_container.setVisible(False)
            )

        animation.finished.connect(lambda: setattr(self, '_current_animation', None))
        animation.start()
        self._current_animation = animation  # Keep reference

    def _save_state(self):
        """Persist collapse state to QSettings."""
        if self._settings_key:
            settings = QSettings("DICE", "DICE_GUI")
            settings.setValue(f"collapsed/{self._settings_key}",
                            "true" if self._collapsed else "false")

    def is_collapsed(self) -> bool:
        """Return current collapsed state."""
        return self._collapsed

    def set_collapsed(self, collapsed: bool, animate: bool = True):
        """Set collapsed state."""
        if collapsed != self._collapsed:
            if animate:
                self.toggle_collapsed()
            else:
                self._collapsed = collapsed
                self._content_container.setVisible(not collapsed)
                if collapsed:
                    self._content_container.setMaximumHeight(0)
                else:
                    self._content_container.setMaximumHeight(16777215)
                self._update_toggle_icon()
                self._save_state()
