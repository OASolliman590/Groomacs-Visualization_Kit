"""
MMPBSA CLI Entry Point
======================

Command-line interface for MMPBSA analysis.
Supports YAML config, interactive prompts, and CLI arguments.
"""

import argparse
import logging
import sys
from typing import Optional
from pathlib import Path

from ..mmpbsa import MMPBSAAnalyzer, MMPBSAConfig, MMPBSASystemConfig
from ..mmpbsa.yaml_loader import load_yaml_config, generate_yaml_template
from ..md.config import PlotConfig
import yaml

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def interactive_prompt(
    default_base_dir: Optional[Path] = None,
    default_output_dir: Optional[Path] = None,
    default_protein_name: Optional[str] = None,
    default_ligand_name: Optional[str] = None,
    save_yaml: bool = True,
    save_path: Optional[Path] = None,
) -> MMPBSAConfig:
    """
    Interactive prompt for MMPBSA configuration.
    
    Returns:
        MMPBSAConfig object
    """
    print("\n" + "="*70)
    print("  MMPBSA Analysis - Interactive Configuration")
    print("="*70 + "\n")
    
    # Basic info
    protein_default = default_protein_name or "ProteinX"
    ligand_default = default_ligand_name or "LigandA"
    protein_name = input(f"Enter protein name [{protein_default}]: ").strip() or protein_default
    ligand_name = input(f"Enter ligand name [{ligand_default}]: ").strip() or ligand_default
    
    base_prompt = "Enter base data directory"
    if default_base_dir:
        base_prompt += f" [{default_base_dir}]"
    base_dir = input(f"{base_prompt}: ").strip()
    if not base_dir and default_base_dir:
        base_dir = str(default_base_dir)
    if not base_dir:
        print("Error: Base directory is required")
        sys.exit(1)
    
    output_default = default_output_dir or "./output_mmpbsa"
    output_dir = input(f"Enter output directory [{output_default}]: ").strip() or str(output_default)
    
    # System configuration
    print("\n--- System Configuration ---")
    system_name = input(f"Enter system name (ligand) [{ligand_name}]: ").strip() or ligand_name
    dir_pattern = input("Enter directory pattern (use {} for replicate number) [3-Results_Holo/r/HOLO_{}]: ").strip()
    if not dir_pattern:
        dir_pattern = "3-Results_Holo/r/HOLO_{}"
    
    replicates_input = input("Number of replicates [3]: ").strip()
    replicates = int(replicates_input) if replicates_input else 3
    
    results_pattern = input("Results file pattern [FINAL_RESULTS_MMPBSA*.dat]: ").strip() or "FINAL_RESULTS_MMPBSA*.dat"
    decomp_pattern = input("Decomposition file pattern [FINAL_DECOMP_MMPBSA*.dat]: ").strip() or "FINAL_DECOMP_MMPBSA*.dat"
    
    # File format
    file_format = input("File format (csv/dat/auto) [auto]: ").strip() or "auto"
    
    # Create system
    system = MMPBSASystemConfig(
        name=system_name,
        dir_pattern=dir_pattern,
        replicates=replicates,
        results_file_pattern=results_pattern,
        decomp_file_pattern=decomp_pattern
    )
    
    # Plot config (use defaults)
    plot_config = PlotConfig(
        template='ggplot2',
        style='publication',
        font_family='Times New Roman',
        font_size=24
    )
    
    # Create config
    config = MMPBSAConfig(
        base_dir=Path(base_dir),
        output_dir=Path(output_dir),
        protein_name=protein_name,
        ligand_name=ligand_name,
        systems=[system],
        file_format=file_format,
        plot_config=plot_config
    )
    
    # Ask to save
    if save_yaml:
        save = input("\nSave this configuration to YAML? (y/n) [y]: ").strip().lower()
        if save != 'n':
            yaml_path = save_path or Path(f"{ligand_name.lower()}_mmpbsa_config.yaml")
            save_config_to_yaml(config, yaml_path)
            print(f"✓ Configuration saved to: {yaml_path}")
    
    return config


def save_config_to_yaml(config: MMPBSAConfig, output_file: Path):
    """Save MMPBSA config to YAML file."""
    import yaml
    
    data = {
        'protein_name': config.protein_name,
        'ligand_name': config.ligand_name,
        'base_dir': str(config.base_dir),
        'output_dir': str(config.output_dir),
        'systems': [
            {
                'name': s.name,
                'dir_pattern': s.dir_pattern,
                'replicates': s.replicates,
                'results_file_pattern': s.results_file_pattern,
                'decomp_file_pattern': s.decomp_file_pattern
            }
            for s in config.systems
        ],
        'file_format': config.file_format,
        'plot_config': {
            'template': config.plot_config.template,
            'style': config.plot_config.style,
            'width': config.plot_config.width,
            'height': config.plot_config.height,
            'scale': config.plot_config.scale,
            'font_family': config.plot_config.font_family,
            'font_size': config.plot_config.font_size,
            'save_formats': config.plot_config.save_formats
        }
    }
    
    if config.amino_acids:
        data['amino_acids'] = {'custom': config.amino_acids}
    
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GROMACS MMPBSA Analysis Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate YAML template
  gromacs-mmpbsa --generate-template
  
  # Use YAML configuration
  gromacs-mmpbsa --config my_config.yaml
  
  # Interactive mode
  gromacs-mmpbsa --interactive
  
  # CLI arguments
  gromacs-mmpbsa --protein ProteinX --ligand LigandA --base-dir ./data --replicates 3
        """
    )
    
    # Configuration modes
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode (step-by-step prompts)'
    )
    config_group.add_argument(
        '--generate-template', '-g',
        action='store_true',
        help='Generate a YAML configuration template'
    )
    
    # CLI arguments (for direct specification)
    parser.add_argument('--protein', help='Protein name')
    parser.add_argument('--ligand', help='Ligand name')
    parser.add_argument('--base-dir', type=Path, help='Base data directory')
    parser.add_argument('--output-dir', type=Path, help='Output directory')
    parser.add_argument('--replicates', type=int, default=3, help='Number of replicates')
    parser.add_argument('--dir-pattern', help='Directory pattern (use {} for replicate)')
    parser.add_argument('--file-format', choices=['csv', 'dat', 'auto'], default='auto', help='File format')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate template
    if args.generate_template:
        template_path = Path("mmpbsa_config_template.yaml")
        generate_yaml_template(template_path)
        print(f"\n✓ Template created: {template_path}")
        return
    
    # Load configuration
    if args.config:
        # YAML mode
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        config = load_yaml_config(args.config)
        
    elif args.interactive:
        # Interactive mode
        config = interactive_prompt()
        
    elif args.protein and args.ligand and args.base_dir:
        # CLI arguments mode
        system = MMPBSASystemConfig(
            name=args.ligand,
            dir_pattern=args.dir_pattern or "HOLO_{}",
            replicates=args.replicates,
            results_file_pattern="FINAL_RESULTS_MMPBSA*.dat",
            decomp_file_pattern="FINAL_DECOMP_MMPBSA*.dat"
        )
        
        config = MMPBSAConfig(
            base_dir=args.base_dir,
            output_dir=args.output_dir or args.base_dir / "mmpbsa_output",
            protein_name=args.protein,
            ligand_name=args.ligand,
            systems=[system],
            file_format=args.file_format,
            plot_config=PlotConfig(
                template='ggplot2',
                style='publication',
                font_family='Times New Roman',
                font_size=24
            )
        )
        
    else:
        parser.print_help()
        print("\nError: Must specify --config, --interactive, or provide CLI arguments")
        sys.exit(1)
    
    # Run analysis
    print("\n" + "="*70)
    print("  Starting MMPBSA Analysis")
    print("="*70 + "\n")
    
    analyzer = MMPBSAAnalyzer(config)
    results = analyzer.run_analysis()
    
    if results.get('success'):
        print("\n" + "="*70)
        print("  ✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📊 Results:")
        print(f"  Output directory: {results['output_dir']}")
        print(f"  Plots created: {len(results['plots_created'])}")
        print(f"  Systems analyzed: {results['n_systems']}")
        if results.get('statistics_file'):
            print(f"  Statistics: {results['statistics_file']}")
        if results.get('summary_file'):
            print(f"  Summary: {results['summary_file']}")
        print("="*70 + "\n")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
