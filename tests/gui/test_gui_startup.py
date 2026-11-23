"""
The step of the startup that replaces the splash screen with the main window.

Qt ends the application when its last visible window closes. The splash
screen is a plain top-level widget, so closing it before the main window is
shown ended the application at some launches, without an error.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt

from dice_gui.__main__ import show_main_window
from dice_gui.splash import DiceSplashScreen


class RecordingWidget:
    """Stand-in for a window: records the calls made on it in a shared list."""

    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls

    def show(self) -> None:
        self.calls.append(f"{self.name}.show")

    def close(self) -> None:
        self.calls.append(f"{self.name}.close")


class TestSplashToMainWindow:

    def test_main_window_is_shown_before_the_splash_closes(self):
        calls: list[str] = []

        show_main_window(RecordingWidget("window", calls), RecordingWidget("splash", calls))

        assert calls == ["window.show", "splash.close"]

    def test_closing_the_splash_does_not_end_the_application(self, qtbot):
        """With no other window shown, a closing splash screen must not count as the last window."""
        splash = DiceSplashScreen()
        qtbot.addWidget(splash)

        assert not splash.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose)

    def test_splash_is_replaced_by_the_main_window(self, qtbot, qapp, window):
        splash = DiceSplashScreen()
        qtbot.addWidget(splash)
        splash.show()

        with qtbot.assertNotEmitted(qapp.lastWindowClosed, wait=200):
            show_main_window(window, splash)

        assert window.isVisible()
        assert not splash.isVisible()
