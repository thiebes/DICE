"""
GUI tests for gui.main_window module.
"""
import pytest
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt
from gui.main_window import ValidatedLineEdit, MainWindow


@pytest.mark.gui
class TestValidatedLineEdit:
    """Tests for ValidatedLineEdit widget."""

    def test_no_styling_on_init(self, qtbot):
        """Test that no styling is applied on initialization."""
        validator = QDoubleValidator()
        widget = ValidatedLineEdit(validator=validator)
        qtbot.addWidget(widget)

        # Set initial text programmatically
        widget.setText("5.0")

        # Should NOT have colored border styling since user hasn't interacted
        style = widget.styleSheet()
        assert "border" not in style or style == ""

    def test_styling_after_interaction(self, qtbot):
        """Test that styling is applied after user interaction."""
        validator = QDoubleValidator()
        widget = ValidatedLineEdit(validator=validator)
        qtbot.addWidget(widget)

        # Simulate user interaction by focusing and editing
        widget.setFocus()
        qtbot.keyClicks(widget, "5.0")

        # Trigger editingFinished to mark as interacted
        widget.editingFinished.emit()

        # Now change the text to trigger validation styling
        widget.clear()
        qtbot.keyClicks(widget, "10.0")

        # Should have green border for valid input
        style = widget.styleSheet()
        assert "border" in style
        assert "green" in style

    def test_invalid_input_styling(self, qtbot):
        """Test styling for intermediate input after interaction.

        Note: QDoubleValidator returns Intermediate (not Invalid) for text like
        'abc' because Qt considers it potentially correctable. Orange indicates
        Intermediate state.
        """
        validator = QDoubleValidator()
        widget = ValidatedLineEdit(validator=validator)
        qtbot.addWidget(widget)

        # Simulate user interaction by typing valid text first, then clearing
        widget.setFocus()
        qtbot.keyClicks(widget, "1.0")
        widget.editingFinished.emit()

        # Clear and enter non-numeric text
        widget.clear()
        qtbot.keyClicks(widget, "abc")

        # Should have orange border for intermediate input (Qt considers 'abc' intermediate, not invalid)
        style = widget.styleSheet()
        assert "border" in style
        assert "orange" in style


@pytest.mark.gui
class TestMainWindow:
    """Tests for MainWindow."""

    def test_main_window_creation(self, qtbot):
        """Test that MainWindow can be created."""
        window = MainWindow()
        qtbot.addWidget(window)

        assert window.windowTitle() == "DICE Simulation"
        assert window.run_button is not None
        assert window.progress_bar is not None

    def test_validation_spatial_width_positive(self, qtbot):
        """Test validation requires positive spatial width."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set invalid spatial width
        window.spatial_width_edit.setText("0")

        is_valid, errors = window.validate_parameters()

        assert not is_valid
        assert any("Spatial width must be greater than 0" in err for err in errors)

    def test_validation_spatial_width_negative(self, qtbot):
        """Test validation rejects negative spatial width."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set negative spatial width
        window.spatial_width_edit.setText("-5")

        is_valid, errors = window.validate_parameters()

        assert not is_valid
        assert any("Spatial width must be greater than 0" in err for err in errors)

    def test_validation_proximity_range(self, qtbot):
        """Test validation requires proximity level between 0 and 1."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set all required fields to valid values first
        window.spatial_width_edit.setText("10.0")
        window.pixel_width_spinbox.setValue(100)
        window.radio_time_range.setChecked(True)
        window.time_range_start_edit.setText("0")
        window.time_range_stop_edit.setText("5")
        window.time_range_steps_edit.setText("6")

        # Set invalid proximity (> 1)
        window.proximity_level_edit.setText("1.5")

        is_valid, errors = window.validate_parameters()

        assert not is_valid
        assert any("Proximity level must be between 0.0 and 1.0" in err for err in errors)

    def test_validation_proximity_negative(self, qtbot):
        """Test validation rejects negative proximity."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set all required fields to valid values first
        window.spatial_width_edit.setText("10.0")
        window.pixel_width_spinbox.setValue(100)
        window.radio_time_range.setChecked(True)
        window.time_range_start_edit.setText("0")
        window.time_range_stop_edit.setText("5")
        window.time_range_steps_edit.setText("6")

        # Set negative proximity
        window.proximity_level_edit.setText("-0.1")

        is_valid, errors = window.validate_parameters()

        assert not is_valid
        assert any("Proximity level must be between 0.0 and 1.0" in err for err in errors)

    def test_validation_time_range_ordering(self, qtbot):
        """Test validation requires time start < stop."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set all required fields to valid values
        window.spatial_width_edit.setText("10.0")
        window.pixel_width_spinbox.setValue(100)
        window.proximity_level_edit.setText("0.1")

        # Set time range with start >= stop
        window.radio_time_range.setChecked(True)
        window.time_range_start_edit.setText("5")
        window.time_range_stop_edit.setText("0")
        window.time_range_steps_edit.setText("6")

        is_valid, errors = window.validate_parameters()

        assert not is_valid
        assert any("Time range start must be less than stop" in err for err in errors)

    def test_validation_all_valid(self, qtbot):
        """Test validation passes with all valid parameters."""
        window = MainWindow()
        qtbot.addWidget(window)

        # Set all fields to valid values
        window.spatial_width_edit.setText("10.0")
        window.pixel_width_spinbox.setValue(100)
        window.proximity_level_edit.setText("0.1")
        window.radio_time_range.setChecked(True)
        window.time_range_start_edit.setText("0")
        window.time_range_stop_edit.setText("5")
        window.time_range_steps_edit.setText("6")

        is_valid, errors = window.validate_parameters()

        assert is_valid
        assert len(errors) == 0
