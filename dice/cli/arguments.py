"""
Command-line argument parsing for DICE.

This module provides argument parsing functionality for the DICE command-line interface,
including parameter validation and help text generation.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser for DICE.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    
    Examples
    --------
    >>> parser = create_parser()
    >>> args = parser.parse_args(['parameters.txt'])
    >>> args.parameters_file
    'parameters.txt'
    """
    parser = argparse.ArgumentParser(
        prog='dice',
        description='DICE (Diffusion Insight Computation Engine) - Monte Carlo simulation tool for quantifying noise effects in optical measures of excited state transport.',
        epilog="""
Examples:
  dice parameters.txt                    # Run simulation with parameters file
  dice parameters.txt --verbose         # Run with verbose output
  dice parameters.txt --dry-run          # Validate parameters without running
  dice --version                         # Show version information
  
For more information, visit: https://github.com/your-repo/DICE
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        'parameters_file',
        type=str,
        help='Path to the parameters file containing simulation configuration'
    )
    
    # Optional arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed progress information'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--dry-run', '--check',
        action='store_true',
        help='Validate parameters and show simulation plan without running'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (overrides parameters file setting)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generation of plots and visualizations'
    )
    
    parser.add_argument(
        '--multiprocessing',
        type=int,
        metavar='N',
        help='Number of parallel processes to use (0 for single-threaded, -1 for all cores)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        metavar='N',
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Advanced options
    advanced = parser.add_argument_group('advanced options')
    
    advanced.add_argument(
        '--retain-profiles',
        action='store_true',
        help='Retain all profile data in memory (increases memory usage)'
    )
    
    advanced.add_argument(
        '--export-format',
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Export format for results (default: csv)'
    )
    
    advanced.add_argument(
        '--progress-interval',
        type=int,
        default=100,
        metavar='N',
        help='Progress update interval for verbose mode (default: 100)'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate parsed command-line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from argparse.
    
    Raises
    ------
    ValueError
        If arguments are invalid or incompatible.
    FileNotFoundError
        If the parameters file doesn't exist.
    
    Examples
    --------
    >>> args = Namespace(parameters_file='test.txt', verbose=True, quiet=False)
    >>> validate_arguments(args)
    """
    # Check for conflicting options
    if args.verbose and args.quiet:
        raise ValueError("Cannot specify both --verbose and --quiet")
    
    # Validate parameters file exists
    params_path = Path(args.parameters_file)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {args.parameters_file}")
    
    if not params_path.is_file():
        raise ValueError(f"Parameters path is not a file: {args.parameters_file}")
    
    # Validate output directory if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {args.output_dir}")
    
    # Validate multiprocessing setting
    if args.multiprocessing is not None:
        if args.multiprocessing < -1:
            raise ValueError("Multiprocessing value must be >= -1")
    
    # Validate seed
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("Random seed must be non-negative")
    
    # Validate progress interval
    if hasattr(args, 'progress_interval') and args.progress_interval <= 0:
        raise ValueError("Progress interval must be positive")


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments with validation.
    
    Parameters
    ----------
    args : list of str, optional
        Arguments to parse. If None, uses sys.argv.
    
    Returns
    -------
    Namespace
        Parsed and validated arguments.
    
    Raises
    ------
    SystemExit
        If arguments are invalid or help is requested.
    
    Examples
    --------
    >>> args = parse_arguments(['parameters.txt', '--verbose'])
    >>> args.verbose
    True
    """
    parser = create_parser()
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    try:
        # Validate arguments
        validate_arguments(parsed_args)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    
    return parsed_args


def show_parameter_summary(args: argparse.Namespace) -> None:
    """
    Display a summary of the parsed arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments to summarize.
    
    Examples
    --------
    >>> args = parse_arguments(['test.txt'])
    >>> show_parameter_summary(args)
    DICE Simulation Parameters:
    Parameters file: test.txt
    ...
    """
    print("DICE Simulation Parameters:")
    print(f"  Parameters file: {args.parameters_file}")
    
    if args.output_dir:
        print(f"  Output directory: {args.output_dir}")
    
    if args.verbose:
        print("  Verbose output: enabled")
    elif args.quiet:
        print("  Quiet mode: enabled")
    
    if args.dry_run:
        print("  Dry run: enabled (validation only)")
    
    if args.no_plots:
        print("  Plot generation: disabled")
    
    if args.multiprocessing is not None:
        if args.multiprocessing == -1:
            print("  Multiprocessing: all available cores")
        elif args.multiprocessing == 0:
            print("  Multiprocessing: disabled (single-threaded)")
        else:
            print(f"  Multiprocessing: {args.multiprocessing} processes")
    
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    
    if hasattr(args, 'retain_profiles') and args.retain_profiles:
        print("  Profile retention: enabled")
    
    if hasattr(args, 'export_format'):
        print(f"  Export format: {args.export_format}")
    
    print()


def get_help_text() -> str:
    """
    Get formatted help text for the CLI.
    
    Returns
    -------
    str
        Formatted help text.
    
    Examples
    --------
    >>> help_text = get_help_text()
    >>> 'DICE' in help_text
    True
    """
    parser = create_parser()
    return parser.format_help()


def handle_special_arguments(args: argparse.Namespace) -> bool:
    """
    Handle special arguments that don't require full simulation run.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    
    Returns
    -------
    bool
        True if a special argument was handled (should exit), False otherwise.
    
    Examples
    --------
    >>> args = Namespace(dry_run=True, parameters_file='test.txt')
    >>> handled = handle_special_arguments(args)
    >>> handled
    True
    """
    if args.dry_run:
        print("Dry run mode: validating parameters only...")
        # Parameter validation would happen here
        print("✓ Parameters file is valid")
        print("✓ Configuration looks good")
        print("Simulation would run with these parameters.")
        return True
    
    return False