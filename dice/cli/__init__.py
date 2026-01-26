"""
Command-line interface package for DICE.

This package provides the command-line interface for DICE, including argument
parsing, validation, and the main entry point for command-line execution.
"""

from .main import main
from .arguments import (
    create_parser,
    parse_arguments,
    validate_arguments,
    show_parameter_summary,
    get_help_text,
)

__all__ = [
    'main',
    'create_parser',
    'parse_arguments', 
    'validate_arguments',
    'show_parameter_summary',
    'get_help_text',
]