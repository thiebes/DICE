"""
Tests for dice.cli.arguments module.
"""

import pytest
import tempfile
import os
from pathlib import Path
import argparse
from dice.cli.arguments import (
    create_parser,
    validate_arguments,
    parse_arguments,
    show_parameter_summary,
    get_help_text,
    handle_special_arguments,
)


class TestCreateParser:
    """Test argument parser creation."""
    
    def test_parser_creation(self):
        """Test that parser is created correctly."""
        parser = create_parser()
        
        assert parser.prog == 'dice'
        assert 'DICE' in parser.description
        assert 'parameters_file' in [action.dest for action in parser._actions]
    
    def test_parser_help(self):
        """Test that parser generates help text."""
        parser = create_parser()
        help_text = parser.format_help()
        
        assert 'DICE' in help_text
        assert 'parameters_file' in help_text
        assert '--verbose' in help_text
        assert '--quiet' in help_text


class TestValidateArguments:
    """Test argument validation."""
    
    def test_valid_arguments(self):
        """Test validation with valid arguments."""
        # Create a temporary parameters file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = argparse.Namespace(
                parameters_file=temp_filename,
                verbose=False,
                quiet=False,
                output_dir=None,
                multiprocessing=None,
                seed=None
            )
            
            # Should not raise any exceptions
            validate_arguments(args)
        
        finally:
            os.unlink(temp_filename)
    
    def test_conflicting_verbose_quiet(self):
        """Test validation with conflicting verbose and quiet options."""
        args = argparse.Namespace(
            parameters_file='dummy.txt',
            verbose=True,
            quiet=True,
            output_dir=None,
            multiprocessing=None,
            seed=None
        )
        
        with pytest.raises(ValueError, match="Cannot specify both --verbose and --quiet"):
            validate_arguments(args)
    
    def test_missing_parameters_file(self):
        """Test validation with missing parameters file."""
        args = argparse.Namespace(
            parameters_file='nonexistent_file.txt',
            verbose=False,
            quiet=False,
            output_dir=None,
            multiprocessing=None,
            seed=None
        )
        
        with pytest.raises(FileNotFoundError, match="Parameters file not found"):
            validate_arguments(args)
    
    def test_invalid_output_directory(self):
        """Test validation with invalid output directory."""
        # Create a temporary file (not directory)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Create temp params file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as params_file:
            params_file.write("{'test': 'data'}")
            params_filename = params_file.name
        
        try:
            args = argparse.Namespace(
                parameters_file=params_filename,
                verbose=False,
                quiet=False,
                output_dir=temp_filename,  # This is a file, not directory
                multiprocessing=None,
                seed=None
            )
            
            with pytest.raises(ValueError, match="Output path exists but is not a directory"):
                validate_arguments(args)
        
        finally:
            os.unlink(temp_filename)
            os.unlink(params_filename)
    
    def test_invalid_multiprocessing(self):
        """Test validation with invalid multiprocessing value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = argparse.Namespace(
                parameters_file=temp_filename,
                verbose=False,
                quiet=False,
                output_dir=None,
                multiprocessing=-2,  # Invalid value
                seed=None
            )
            
            with pytest.raises(ValueError, match="Multiprocessing value must be >= -1"):
                validate_arguments(args)
        
        finally:
            os.unlink(temp_filename)
    
    def test_invalid_seed(self):
        """Test validation with invalid seed value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = argparse.Namespace(
                parameters_file=temp_filename,
                verbose=False,
                quiet=False,
                output_dir=None,
                multiprocessing=None,
                seed=-1  # Invalid value
            )
            
            with pytest.raises(ValueError, match="Random seed must be non-negative"):
                validate_arguments(args)
        
        finally:
            os.unlink(temp_filename)


class TestParseArguments:
    """Test argument parsing with validation."""
    
    def test_minimal_arguments(self):
        """Test parsing with minimal arguments."""
        # Create temporary parameters file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = parse_arguments([temp_filename])
            
            assert args.parameters_file == temp_filename
            assert not args.verbose
            assert not args.quiet
        
        finally:
            os.unlink(temp_filename)
    
    def test_all_arguments(self):
        """Test parsing with all arguments."""
        # Create temporary parameters file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = parse_arguments([
                temp_filename,
                '--verbose',
                '--output-dir', '/tmp/test',
                '--multiprocessing', '4',
                '--seed', '42',
                '--dry-run',
                '--no-plots',
                '--retain-profiles'
            ])
            
            assert args.parameters_file == temp_filename
            assert args.verbose
            assert not args.quiet
            assert args.output_dir == '/tmp/test'
            assert args.multiprocessing == 4
            assert args.seed == 42
            assert args.dry_run
            assert args.no_plots
            assert args.retain_profiles
        
        finally:
            os.unlink(temp_filename)
    
    def test_parse_error_handling(self):
        """Test parsing error handling."""
        # This should raise SystemExit due to missing file
        with pytest.raises(SystemExit):
            parse_arguments(['nonexistent_file.txt'])


class TestShowParameterSummary:
    """Test parameter summary display."""
    
    def test_basic_summary(self):
        """Test basic parameter summary (just check it doesn't crash)."""
        args = argparse.Namespace(
            parameters_file='test.txt',
            verbose=False,
            quiet=False,
            output_dir=None,
            multiprocessing=None,
            seed=None,
            dry_run=False,
            no_plots=False
        )
        
        # This should not raise any exceptions
        show_parameter_summary(args)
    
    def test_verbose_summary(self):
        """Test summary with all options enabled."""
        args = argparse.Namespace(
            parameters_file='test.txt',
            verbose=True,
            quiet=False,
            output_dir='/tmp/output',
            multiprocessing=8,
            seed=123,
            dry_run=True,
            no_plots=True,
            retain_profiles=True,
            export_format='json'
        )
        
        # This should not raise any exceptions
        show_parameter_summary(args)


class TestGetHelpText:
    """Test help text generation."""
    
    def test_help_text(self):
        """Test that help text is generated correctly."""
        help_text = get_help_text()
        
        assert isinstance(help_text, str)
        assert 'DICE' in help_text
        assert 'parameters_file' in help_text
        assert '--verbose' in help_text


class TestHandleSpecialArguments:
    """Test special argument handling."""
    
    def test_dry_run_handling(self):
        """Test dry run argument handling."""
        args = argparse.Namespace(
            dry_run=True,
            parameters_file='test.txt'
        )
        
        # Should return True (handled)
        result = handle_special_arguments(args)
        assert result is True
    
    def test_normal_arguments(self):
        """Test normal arguments (no special handling)."""
        args = argparse.Namespace(
            dry_run=False,
            parameters_file='test.txt'
        )
        
        # Should return False (not handled)
        result = handle_special_arguments(args)
        assert result is False


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_directory_as_parameters_file(self):
        """Test validation when parameters file is a directory."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            args = argparse.Namespace(
                parameters_file=temp_dir,
                verbose=False,
                quiet=False,
                output_dir=None,
                multiprocessing=None,
                seed=None
            )
            
            with pytest.raises(ValueError, match="Parameters path is not a file"):
                validate_arguments(args)
        
        finally:
            os.rmdir(temp_dir)
    
    def test_zero_multiprocessing(self):
        """Test validation with zero multiprocessing (should be valid)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("{'test': 'data'}")
            temp_filename = temp_file.name
        
        try:
            args = argparse.Namespace(
                parameters_file=temp_filename,
                verbose=False,
                quiet=False,
                output_dir=None,
                multiprocessing=0,  # Should be valid
                seed=None
            )
            
            # Should not raise any exceptions
            validate_arguments(args)
        
        finally:
            os.unlink(temp_filename)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])