"""
Tests for dice_gui/standalone.py, the entry script of the standalone build.
"""

import sys

import pytest

from dice_gui import standalone


class TestEntryPoint:

    @pytest.fixture
    def calls(self, monkeypatch) -> list:
        """Record freeze_support and the GUI start instead of running them."""
        import multiprocessing
        import dice_gui.dice_gui

        calls = []
        monkeypatch.setattr(multiprocessing, 'freeze_support', lambda: calls.append('freeze_support'))
        monkeypatch.setattr(dice_gui.dice_gui, 'main', lambda: calls.append('gui'))
        return calls

    def test_without_the_option_the_gui_starts(self, monkeypatch, calls):
        monkeypatch.setattr(sys, 'argv', ['DICE'])
        standalone.main()
        assert calls == ['freeze_support', 'gui']

    def test_a_worker_process_ends_in_freeze_support(self, monkeypatch, calls):
        import multiprocessing

        def run_worker():
            calls.append('freeze_support')
            raise SystemExit(0)

        monkeypatch.setattr(multiprocessing, 'freeze_support', run_worker)
        monkeypatch.setattr(sys, 'argv', ['DICE', '--multiprocessing-fork', 'parent_pid=1', 'pipe_handle=2'])

        with pytest.raises(SystemExit):
            standalone.main()
        assert calls == ['freeze_support']
