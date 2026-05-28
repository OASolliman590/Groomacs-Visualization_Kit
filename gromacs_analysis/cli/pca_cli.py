"""
PCA CLI Module
==============

Command-line interface for PCA analysis.
"""

import argparse
import logging
import sys
from typing import Optional
from pathlib import Path

from ..pca import PCAAnalyzer, PCAConfig
from ..pca.yaml_loader import load_yaml_config, generate_yaml_template
from ..md.config import PlotConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def interactive_prompt(
    default_base_dir: Optional[Path] = None,
    default_output_dir: Optional[Path] = None,
    default_protein_name: Optional[str] = None,
) -> PCAConfig:
    """
    Interactive prompt for PCA configuration.
    
    Returns:
        PCAConfig object
    """
    print("\n" + "="*60)
    print("PCA Analysis - Interactive Configuration")
    print("="*60 + "\n")
    
    # Base directory
    base_prompt = "Enter PCA base directory (e.g., /path/to/2-PCA)"
    if default_base_dir:
        base_prompt += f" [{default_base_dir}]"
    base_dir_str = input(f"{base_prompt}: ").strip()
    if not base_dir_str and default_base_dir:
        base_dir = Path(default_base_dir)
    elif base_dir_str:
        base_dir = Path(base_dir_str)
    else:
        print("Error: Base directory is required")
        sys.exit(1)
    
    if not base_dir.exists():
        print(f"Warning: Directory does not exist: {base_dir}")
        create = input("Create directory? (y/n): ").strip().lower()
        if create != 'y':
            sys.exit(1)
        base_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directory
    if default_output_dir:
        output_prompt = f"Enter output directory [{default_output_dir}]: "
    else:
        output_prompt = "Enter output directory (press Enter for base_dir/PCA_Vis): "
    output_dir_str = input(output_prompt).strip()
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path(default_output_dir) if default_output_dir else base_dir / 'PCA_Vis'
    
    # Protein name
    protein_default = default_protein_name or "Protein"
    protein_name = input(f"Enter protein name (default: {protein_default}): ").strip() or protein_default
    
    # Number of PCs
    n_pcs_str = input("Enter number of PCs to analyze (default: 10): ").strip()
    try:
        n_pcs = int(n_pcs_str) if n_pcs_str else 10
    except ValueError:
        n_pcs = 10
    
    # Plot config
    print("\nPlot Configuration:")
    template = input("Template (default: plotly_white): ").strip() or "plotly_white"
    font_family = input("Font family (default: Times New Roman): ").strip() or "Times New Roman"
    font_size_str = input("Font size (default: 24): ").strip()
    try:
        font_size = int(font_size_str) if font_size_str else 24
    except ValueError:
        font_size = 24
    
    plot_config = PlotConfig(
        template=template,
        font_family=font_family,
        font_size=font_size
    )
    
    config = PCAConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        protein_name=protein_name,
        n_pcs=n_pcs,
        plot_config=plot_config
    )
    
    return config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PCA Analysis for GROMACS trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--generate-template',
        action='store_true',
        help='Generate a template YAML configuration file'
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        help='Base directory containing PCA files (quick mode)'
    )
    
    parser.add_argument(
        '--protein-name',
        type=str,
        help='Protein name (quick mode)'
    )

    parser.add_argument(
        '--run-gromacs',
        action='store_true',
        help='Run GROMACS PCA generation before plotting (requires input files)'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        help='Simulation input directory for --run-gromacs'
    )
    
    args = parser.parse_args()
    
    # Generate template
    if args.generate_template:
        generate_yaml_template(Path("pca_config_template.yaml"))
        return
    
    # Load configuration
    if args.interactive:
        config = interactive_prompt()
    elif args.config:
        if not args.config.exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        config = load_yaml_config(args.config)
    elif args.base_dir:
        # Quick mode
        base_dir = args.base_dir
        protein_name = args.protein_name or "Protein"
        
        config = PCAConfig(
            base_dir=base_dir,
            output_dir=base_dir / 'PCA_Vis',
            protein_name=protein_name
        )
    else:
        parser.print_help()
        sys.exit(1)

    if args.run_gromacs:
        config.run_gromacs.enabled = True
    if args.input_dir:
        config.run_gromacs.input_dir = args.input_dir
    
    # Run analysis
    try:
        analyzer = PCAAnalyzer(config)
        results = analyzer.run_analysis()
        
        if results['success']:
            logger.info("\n✓ Analysis completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n✗ Analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
