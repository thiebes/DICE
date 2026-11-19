"""
Command-line interface package for DICE.

This package provides the command-line interface for DICE, including argument
parsing, validation, and the main entry point for command-line execution.
"""

import sys
import types

# Lazy imports (PEP 562). Importing the package must not import dice.cli.main:
# `python -m dice.cli.main` imports the package first, and runpy warns when the
# module it is about to run as __main__ is already in sys.modules.
_LAZY_IMPORTS = {
    'main': '.main',
    'create_parser': '.arguments',
    'parse_arguments': '.arguments',
    'validate_arguments': '.arguments',
    'show_parameter_summary': '.arguments',
    'get_help_text': '.arguments',
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _CliPackage(types.ModuleType):
    """Keeps `dice.cli.main` resolving to the entry function, never the submodule.

    When the submodule dice.cli.main is first imported, the import system binds
    it to the package attribute of the same name, which would make
    `from dice.cli import main` return the module. That binding is skipped, so
    the name falls through to __getattr__ and resolves to the function.
    """

    def __setattr__(self, name, value):
        if name == 'main' and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CliPackage

__all__ = [
    'main',
    'create_parser',
    'parse_arguments',
    'validate_arguments',
    'show_parameter_summary',
    'get_help_text',
]
