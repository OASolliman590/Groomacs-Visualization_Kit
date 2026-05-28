"""
MD CLI Entry Point
==================

Unified command-line interface supporting:
1. Interactive mode (--interactive)
2. YAML config mode (--config file.yaml)
3. Command-line arguments mode
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..md.config import MDConfig, SystemConfig, PlotConfig
from ..md.analyzer import MDAnalyzer
from ..config.yaml_loader import load_yaml_config, generate_yaml_template
from ..utils.helpers import parse_amino_acid_range
from .interactive import interactive_prompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_config_from_args(args) -> MDConfig:
    """
    Build MDConfig from command-line arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        MDConfig object
    """
    # Parse systems
    systems = []
    
    if args.systems:
        for system_str in args.systems:
            # Format: "name:pattern:is_apo"
            # Example: "LigandA:HOLO_{}" or "APO:APO_{}:apo"
            parts = system_str.split(':')
            name = parts[0]
            dir_pattern = parts[1] if len(parts) > 1 else name
            is_apo = len(parts) > 2 and parts[2].lower() == 'apo'
            
            systems.append(SystemConfig(
                name=name,
                dir_pattern=dir_pattern,
                is_apo=is_apo,
                replicates=args.replicates
            ))
    
    # Parse amino acids
    amino_acids = None
    if args.amino_acids:
        amino_acids = parse_amino_acid_range(args.amino_acids)
    
    # Create plot config
    plot_config = PlotConfig(
        template=args.template,
        style=args.style
    )
    
    # Create config
    config = MDConfig(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output),
        protein_name=args.protein,
        systems=systems,
        plot_config=plot_config,
        amino_acids=amino_acids
    )
    
    return config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GROMACS MD Trajectory Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  gromacs-md --interactive
  
  # YAML config mode
  gromacs-md --config my_analysis.yaml
  
  # Generate YAML template
  gromacs-md --generate-template
  
  # Command-line arguments mode
  gromacs-md --protein ProteinX --base-dir ./data --systems LigandA:HOLO_{} APO:APO_{}:apo --replicates 3
  
For more information: https://github.com/yourusername/gromacs-analysis-toolkit
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive terminal prompts'
    )
    mode_group.add_argument(
        '--config', '-c',
        type=str,
        help='YAML configuration file'
    )
    mode_group.add_argument(
        '--generate-template', '-g',
        action='store_true',
        help='Generate a template YAML configuration file'
    )
    
    # Python API mode / Command-line arguments
    parser.add_argument(
        '--protein',
        type=str,
        help='Protein name'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory containing data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--systems',
        nargs='+',
        help='Systems to analyze (format: name:pattern:is_apo). Example: LigandA:HOLO_{} APO:APO_{}:apo'
    )
    parser.add_argument(
        '--replicates', '-r',
        type=int,
        default=1,
        help='Number of replicates (default: 1)'
    )
    parser.add_argument(
        '--amino-acids',
        type=str,
        help='Amino acid range for RMSF (e.g., "814-1166" or "814-936,994-1168")'
    )
    parser.add_argument(
        '--style',
        choices=['simple', 'enhanced', 'publication', 'overview', 'comparative'],
        default='simple',
        help='Visualization style (default: simple)'
    )
    parser.add_argument(
        '--template',
        choices=['plotly_white', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'],
        default='plotly_white',
        help='Plot template (default: plotly_white)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (DEBUG level logging)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle modes
    try:
        # Mode 1: Generate template
        if args.generate_template:
            logger.info("Generating YAML template...")
            generate_yaml_template()
            return 0
        
        # Mode 2: Interactive
        elif args.interactive:
            logger.info("Starting interactive mode...")
            config = interactive_prompt()
        
        # Mode 3: YAML config
        elif args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_yaml_config(Path(args.config))
        
        # Mode 4: Command-line arguments
        elif args.protein and args.base_dir and args.systems:
            logger.info("Building configuration from command-line arguments...")
            config = build_config_from_args(args)
        
        else:
            # No valid mode selected - show help
            parser.print_help()
            print("\n⚠ Error: Must specify one of: --interactive, --config, --generate-template, or provide --protein, --base-dir, and --systems")
            return 1
        
        # Run analysis (for modes 2, 3, 4)
        if not args.generate_template:
            logger.info("Initializing analyzer...")
            analyzer = MDAnalyzer(config)
            
            logger.info("Running analysis...")
            results = analyzer.run_analysis()
            
            if results.get('success'):
                print("\n" + "="*70)
                print(" " * 25 + "✓ ANALYSIS COMPLETE")
                print("="*70)
                print(f"\nResults saved to: {results['output_dir']}")
                print(f"Plots created: {len(results['plots_created'])}")
                print(f"  - {', '.join(results['plots_created'][:5])}")
                if len(results['plots_created']) > 5:
                    print(f"  - ... and {len(results['plots_created']) - 5} more")
                print(f"\nStatistics: {results.get('statistics_file', 'N/A')}")
                print(f"Summary: {results.get('summary_file', 'N/A')}")
                print("\n" + "="*70 + "\n")
                return 0
            else:
                print(f"\n⚠ Analysis failed: {results.get('error', 'Unknown error')}")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=args.verbose)
        print(f"\n⚠ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


