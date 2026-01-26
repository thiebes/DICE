"""
Real-time validation management for DICE GUI.
"""

from typing import Dict, Callable, Optional, Set
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit

from dice_gui.validators import ValidationResult


class FieldValidation:
    """Tracks validation state for a single field."""

    def __init__(
        self,
        field_id: str,
        widget: QWidget,
        validator: Callable[[str], ValidationResult],
        required: bool = True,
        condition_group: Optional[str] = None
    ):
        self.field_id = field_id
        self.widget = widget
        self.validator = validator
        self.required = required
        self.condition_group = condition_group
        self.last_result: ValidationResult = ValidationResult(True)
        self.is_active = True


class ValidationManager(QObject):
    """
    Manages real-time validation state across all form fields.

    Responsibilities:
    - Track validation state for all registered fields
    - Handle conditional field groups (radio button selections)
    - Debounce validation on text input
    - Emit signals when overall validity changes
    - Manage Run button enabled state
    """

    validity_changed = pyqtSignal(bool)
    field_validated = pyqtSignal(str, bool, str)  # field_id, is_valid, error_message

    DEBOUNCE_MS = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: Dict[str, FieldValidation] = {}
        self._condition_groups: Dict[str, Set[str]] = {}
        self._active_condition: Dict[str, str] = {}
        self._debounce_timers: Dict[str, QTimer] = {}
        self._is_valid = True

    def register_field(
        self,
        field_id: str,
        widget: QWidget,
        validator: Callable[[str], ValidationResult],
        required: bool = True,
        condition_group: Optional[str] = None,
        condition_category: Optional[str] = None
    ) -> None:
        """Register a field for validation tracking."""
        field = FieldValidation(field_id, widget, validator, required, condition_group)
        self._fields[field_id] = field

        if condition_group and condition_category:
            if condition_category not in self._condition_groups:
                self._condition_groups[condition_category] = set()
            self._condition_groups[condition_category].add(field_id)

        self._connect_widget_signals(field_id, widget)

    def _connect_widget_signals(self, field_id: str, widget: QWidget) -> None:
        """Connect widget signals for real-time validation."""
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda: self._on_text_changed(field_id))
        elif isinstance(widget, QTextEdit):
            widget.textChanged.connect(lambda: self._on_text_changed(field_id))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda: self._validate_field(field_id))

    def _on_text_changed(self, field_id: str) -> None:
        """Handle text change with debouncing."""
        if field_id not in self._debounce_timers:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._validate_field(field_id))
            self._debounce_timers[field_id] = timer

        self._debounce_timers[field_id].stop()
        self._debounce_timers[field_id].start(self.DEBOUNCE_MS)

    def _validate_field(self, field_id: str) -> None:
        """Validate a single field and update state."""
        field = self._fields.get(field_id)
        if not field or not field.is_active:
            return

        value = self._get_widget_value(field.widget)
        result = field.validator(value)
        field.last_result = result

        self.field_validated.emit(field_id, result.is_valid, result.error_message)
        self._check_overall_validity()

    def _get_widget_value(self, widget: QWidget) -> str:
        """Extract current value from widget as string."""
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QTextEdit):
            return widget.toPlainText()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return str(widget.value())
        return ""

    def set_condition_active(self, category: str, active_group: str) -> None:
        """Set which condition group is active for a category."""
        self._active_condition[category] = active_group

        if category in self._condition_groups:
            for field_id in self._condition_groups[category]:
                field = self._fields.get(field_id)
                if field and field.condition_group:
                    field.is_active = (field.condition_group == active_group)
                    if not field.is_active:
                        field.last_result = ValidationResult(True)
                        self.field_validated.emit(field_id, True, "")

        self._check_overall_validity()

    def _check_overall_validity(self) -> None:
        """Check if all active required fields are valid."""
        all_valid = True

        for field in self._fields.values():
            if field.is_active and field.required:
                if not field.last_result.is_valid:
                    all_valid = False
                    break

        if all_valid != self._is_valid:
            self._is_valid = all_valid
            self.validity_changed.emit(all_valid)

    def validate_all(self) -> tuple:
        """Validate all active fields and return overall result."""
        for field in self._fields.values():
            if field.is_active:
                self._validate_field(field.field_id)

        errors = []
        for field in self._fields.values():
            if field.is_active and not field.last_result.is_valid:
                errors.append(field.last_result.error_message)

        if errors:
            return False, errors[0]
        return True, ""

    def is_valid(self) -> bool:
        """Return current overall validity state."""
        return self._is_valid


def apply_validation_style(widget: QWidget, is_valid: bool) -> None:
    """Apply or remove validation error styling to a widget."""
    widget.setProperty("validation-state", "valid" if is_valid else "invalid")
    widget.style().unpolish(widget)
    widget.style().polish(widget)
