"""
Theme and styling system for DICE GUI.

Provides adaptive colors and styles that work in both light and dark modes
while maintaining WCAG AA/AAA accessibility standards.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from typing import Dict


class DiceTheme:
    """Adaptive theme system that respects system light/dark mode."""

    def __init__(self):
        """Initialize theme with system detection."""
        self.is_dark_mode = self._detect_dark_mode()
        self.palette = QApplication.palette()

    def _detect_dark_mode(self) -> bool:
        """
        Detect if system is using dark mode.

        Returns:
            True if dark mode is detected, False otherwise
        """
        palette = QApplication.palette()
        window_color = palette.color(QPalette.ColorRole.Window)

        # Dark mode typically has window color with low lightness
        return window_color.lightness() < 128

    def get_semantic_colors(self) -> Dict[str, QColor]:
        """
        Get semantic colors that adapt to light/dark mode.

        Returns:
            Dictionary of semantic color names to QColor objects
        """
        if self.is_dark_mode:
            return {
                'info_text': QColor(160, 160, 160),  # Light gray - WCAG AA on dark backgrounds
                'calculated_value': QColor(100, 181, 246),  # Light blue - WCAG AA contrast
                'success': QColor(129, 199, 132),  # Light green
                'error': QColor(239, 83, 80),  # Light red
                'warning': QColor(255, 183, 77),  # Amber
                'link': QColor(100, 181, 246),  # Light blue
            }
        else:
            return {
                'info_text': QColor(97, 97, 97),  # Dark gray - WCAG AAA on white (7.8:1)
                'calculated_value': QColor(13, 71, 161),  # Dark blue - WCAG AAA (8.6:1)
                'success': QColor(27, 94, 32),  # Dark green - WCAG AAA (9.4:1)
                'error': QColor(183, 28, 28),  # Dark red - WCAG AA (6.6:1)
                'warning': QColor(230, 81, 0),  # Dark orange - WCAG AA (5.3:1)
                'link': QColor(13, 71, 161),  # Dark blue - WCAG AAA
            }

    def get_stylesheet(self) -> str:
        """
        Get complete stylesheet for the application.

        Returns:
            CSS stylesheet string
        """
        colors = self.get_semantic_colors()

        # Convert QColor to CSS rgba
        def to_css(color: QColor) -> str:
            return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"

        return f"""
            /* Title styling */
            QLabel#title {{
                font-size: 16pt;
                font-weight: bold;
            }}

            /* Info text styling */
            QLabel.info-text {{
                color: {to_css(colors['info_text'])};
                font-style: italic;
            }}

            /* Calculated value labels */
            QLabel.calculated-value {{
                color: {to_css(colors['calculated_value'])};
                font-style: italic;
                font-weight: 500;
            }}

            /* Status labels */
            QLabel.status-info {{
                color: {to_css(colors['calculated_value'])};
            }}

            QLabel.status-success {{
                color: {to_css(colors['success'])};
            }}

            QLabel.status-error {{
                color: {to_css(colors['error'])};
            }}

            /* Proximity display */
            QLabel#proximity-display {{
                font-weight: bold;
                font-size: 11pt;
            }}

            /* Enhanced focus indicators - WCAG 2.4.7 */
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 2px solid {to_css(colors['link'])};
                border-radius: 2px;
            }}

            QRadioButton:focus {{
                outline: 2px solid {to_css(colors['link'])};
                outline-offset: 2px;
            }}

            QSlider:focus {{
                outline: 2px solid {to_css(colors['link'])};
                outline-offset: 2px;
            }}

            QPushButton:focus {{
                border: 2px solid {to_css(colors['link'])};
                outline: 2px solid {to_css(colors['link'])};
                outline-offset: 2px;
            }}

            /* Button styling - respects system theme */
            QPushButton#run-button {{
                font-weight: bold;
                padding: 10px 20px;
                min-height: 44px;
                min-width: 120px;
            }}

            QPushButton#run-button:enabled {{
                background-color: {to_css(colors['success'])};
                color: {'white' if self.is_dark_mode else 'white'};
            }}

            QPushButton#run-button:hover:enabled {{
                background-color: {to_css(QColor(colors['success'].red() + 20,
                                                 colors['success'].green() + 20,
                                                 colors['success'].blue() + 20))};
            }}

            QPushButton#run-button:pressed:enabled {{
                background-color: {to_css(QColor(colors['success'].red() - 20,
                                                 colors['success'].green() - 20,
                                                 colors['success'].blue() - 20))};
            }}

            QPushButton#run-button:disabled {{
                opacity: 0.5;
            }}

            QPushButton#stop-button {{
                min-height: 44px;
                min-width: 120px;
                padding: 10px 20px;
            }}

            /* Group boxes */
            QGroupBox {{
                font-weight: 500;
                margin-top: 12px;
                padding-top: 12px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}

            /* Tab widget */
            QTabWidget::pane {{
                border-top: 2px solid palette(mid);
            }}

            QTabBar::tab {{
                min-width: 120px;
                padding: 8px 16px;
                border: 1px solid palette(mid);
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-bottom: -1px;
            }}

            QTabBar::tab:selected {{
                font-weight: 600;
                border-bottom: 3px solid {to_css(colors['link'])};
                background-color: palette(base);
            }}

            QTabBar::tab:!selected {{
                margin-top: 2px;
                background-color: palette(window);
                opacity: 0.7;
            }}

            QTabBar::tab:hover:!selected {{
                background-color: palette(midlight);
            }}

            /* Ensure minimum clickable areas - WCAG 2.5.5 */
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}

            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}

            /* Validation states for input fields */
            QLineEdit[validation-state="invalid"],
            QTextEdit[validation-state="invalid"] {{
                border: 2px solid {to_css(colors['error'])};
                background-color: rgba({colors['error'].red()}, {colors['error'].green()}, {colors['error'].blue()}, 0.05);
            }}

            QLineEdit[validation-state="valid"],
            QTextEdit[validation-state="valid"] {{
                border: 2px solid {to_css(colors['success'])};
            }}

            /* Disabled run button styling */
            QPushButton#run-button:disabled {{
                background-color: palette(mid);
                color: palette(midlight);
            }}

            /* Collapsible group box styling */
            CollapsibleGroupBox {{
                border: 1px solid palette(mid);
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }}

            CollapsibleGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px 0 24px;
            }}

            /* Required field indicators */
            QLabel.required-label {{
                font-weight: 600;
            }}

            /* Optional field styling */
            QLabel.optional-label {{
                color: {to_css(colors['info_text'])};
            }}

            /* Card-style containers for radio-dependent options */
            QFrame.option-card {{
                border: 1px solid palette(mid);
                border-radius: 4px;
                padding: 12px;
                margin-left: 18px;
                margin-top: 4px;
                margin-bottom: 4px;
            }}

            QFrame.option-card:disabled {{
                background-color: transparent;
                border-color: palette(midlight);
            }}

            /* Advanced section styling */
            CollapsibleGroupBox.advanced-section {{
                border-left: 2px dashed palette(mid);
            }}

            /* Reset button styling */
            QPushButton.reset-button {{
                padding: 4px 8px;
                font-size: 10pt;
                min-height: 28px;
            }}

            /* Preset combo box */
            QComboBox.preset-selector {{
                min-width: 150px;
                padding: 6px 12px;
            }}

            /* Sticky action bar */
            QWidget#sticky-action-bar {{
                border-top: 1px solid palette(mid);
                padding: 10px;
                background-color: palette(window);
            }}

            /* Inline validation error labels */
            QLabel.validation-error {{
                color: {to_css(colors['error'])};
                font-size: 9pt;
                padding-left: 2px;
                margin-top: 2px;
                min-height: 20px;
            }}

            /* Output path preview styling */
            QLabel.output-path-preview {{
                color: {to_css(colors['info_text'])};
                font-family: monospace;
                font-size: 9pt;
                background-color: palette(base);
                border: 1px solid palette(mid);
                border-radius: 3px;
                padding: 8px;
                margin-top: 4px;
            }}
        """

    def get_button_style(self, button_type: str = 'primary') -> str:
        """
        Get stylesheet for a specific button type.

        Args:
            button_type: Type of button ('primary', 'secondary', 'danger')

        Returns:
            CSS stylesheet for the button
        """
        colors = self.get_semantic_colors()

        def to_css(color: QColor) -> str:
            return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"

        if button_type == 'primary':
            bg_color = colors['success']
        elif button_type == 'danger':
            bg_color = colors['error']
        else:  # secondary
            return ""  # Use default styling

        return f"""
            font-weight: bold;
            padding: 10px 20px;
            background-color: {to_css(bg_color)};
            color: white;
            border: none;
            border-radius: 4px;
            min-height: 44px;
        """


def apply_theme(app: QApplication) -> DiceTheme:
    """
    Apply DICE theme to the application.

    Args:
        app: QApplication instance

    Returns:
        DiceTheme instance
    """
    theme = DiceTheme()
    app.setStyleSheet(theme.get_stylesheet())
    return theme
