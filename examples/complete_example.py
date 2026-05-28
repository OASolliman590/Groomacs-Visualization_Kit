#!/usr/bin/env python3
"""
Complete Example: GROMACS MD Analysis Toolkit
==============================================

Demonstrates all three configuration modes and key features.
"""

from pathlib import Path
from gromacs_analysis import (
    MDAnalyzer,
    MDConfig,
    SystemConfig,
    AnalysisMetric,
    PlotConfig,
    generate_yaml_template,
    load_yaml_config
)


def example_1_python_api():
    """
    Example 1: Using Python API directly
    
    Best for: Jupyter notebooks, scripting, programmatic analysis
    """
    print("\n" + "="*70)
    print("Example 1: Python API")
    print("="*70)
    
    # Define systems
    systems = [
        SystemConfig(
            name='LigandA',
            dir_pattern='HOLO_LigandA_{}',  # {} will be replaced with 1, 2, 3
            is_apo=False,
            replicates=3
        ),
        SystemConfig(
            name='APO',
            dir_pattern='APO_{}',
            is_apo=True,
            replicates=3
        )
    ]
    
    # Define custom metrics (optional - defaults available)
    metrics = [
        AnalysisMetric(
            name='rmsd_prot',
            file_pattern='RMSD_protein*.dat,RMSD_apo*.dat',
            title='Protein RMSD',
            ylabel='RMSD (Å)',
            is_holo_only=False
        ),
        AnalysisMetric(
            name='comcom',
            file_pattern='comcom*.dat',
            title='COM-COM Distance',
            ylabel='Distance (Å)',
            is_holo_only=True
        )
    ]
    
    # Plot configuration
    plot_config = PlotConfig(
        template='plotly_white',
        style='publication',  # simple, enhanced, publication, overview, comparative
        width=1400,
        height=700,
        scale=2,
        font_family='Arial',
        save_formats=['html', 'svg']
    )
    
    # Create configuration
    config = MDConfig(
        base_dir=Path('./data'),
        output_dir=Path('./output/example1'),
        protein_name='ProteinX',
        systems=systems,
        metrics=metrics,
        plot_config=plot_config,
        amino_acids=list(range(814, 1167))  # For RMSF
    )
    
    # Run analysis
    analyzer = MDAnalyzer(config)
    results = analyzer.run_analysis()
    
    print(f"\n✓ Analysis complete!")
    print(f"  Output: {results['output_dir']}")
    print(f"  Plots created: {len(results['plots_created'])}")
    
    return results


def example_2_yaml_config():
    """
    Example 2: Using YAML configuration
    
    Best for: Reproducible analysis, sharing configurations, batch processing
    """
    print("\n" + "="*70)
    print("Example 2: YAML Configuration")
    print("="*70)
    
    # Generate template (first time only)
    template_path = Path('my_analysis_config.yaml')
    if not template_path.exists():
        generate_yaml_template(template_path)
        print(f"\n✓ Template created: {template_path}")
        print("  Edit this file with your settings, then run again.")
        return None
    
    # Load configuration from YAML
    config = load_yaml_config(template_path)
    
    # Run analysis
    analyzer = MDAnalyzer(config)
    results = analyzer.run_analysis()
    
    print(f"\n✓ Analysis complete from YAML!")
    print(f"  Output: {results['output_dir']}")
    
    return results


def example_3_all_styles():
    """
    Example 3: Demonstrate all 5 visualization styles
    
    Shows the differences between visualization styles
    """
    print("\n" + "="*70)
    print("Example 3: All 5 Visualization Styles")
    print("="*70)
    
    # Base configuration
    base_config = {
        'base_dir': Path('./data'),
        'protein_name': 'ProteinX',
        'systems': [
            SystemConfig(name='LigandA', dir_pattern='HOLO_{}', replicates=3),
            SystemConfig(name='APO', dir_pattern='APO_{}', is_apo=True, replicates=3)
        ]
    }
    
    styles = ['simple', 'enhanced', 'publication', 'overview', 'comparative']
    
    for style in styles:
        print(f"\nCreating {style} plots...")
        
        plot_config = PlotConfig(style=style)
        
        config = MDConfig(
            **base_config,
            output_dir=Path(f'./output/style_{style}'),
            plot_config=plot_config
        )
        
        analyzer = MDAnalyzer(config)
        results = analyzer.run_analysis()
        
        print(f"  ✓ {style.title()} style complete: {results['plots_dir']}")


def example_4_custom_analysis():
    """
    Example 4: Custom analysis with specific metrics
    
    Shows how to create custom plots and access data programmatically
    """
    print("\n" + "="*70)
    print("Example 4: Custom Analysis")
    print("="*70)
    
    config = MDConfig(
        base_dir=Path('./data'),
        output_dir=Path('./output/custom'),
        protein_name='ProteinX',
        systems=[
            SystemConfig(name='LigandA', dir_pattern='HOLO_{}', replicates=3),
            SystemConfig(name='APO', dir_pattern='APO_{}', is_apo=True, replicates=3)
        ],
        plot_config=PlotConfig(style='comparative')
    )
    
    analyzer = MDAnalyzer(config)
    
    # Get data without running full analysis
    data = analyzer.get_data()
    
    # Create custom plot for specific metric
    if 'rmsd_prot' in data.get('LigandA', {}):
        fig = analyzer.create_custom_plot('rmsd_prot', systems=['LigandA', 'APO'])
        fig.write_html('./output/custom/rmsd_custom.html')
        print("\n✓ Custom RMSD plot created")
    
    # Access raw data for further processing
    for system_name, system_data in data.items():
        print(f"\n{system_name}:")
        for metric_name, metric_data in system_data.items():
            stats = metric_data.get('stats', {})
            print(f"  {metric_name}: mean={stats.get('mean', 0):.3f}, std={stats.get('std', 0):.3f}")


def example_5_single_trajectory():
    """
    Example 5: Single trajectory (no replicates)
    
    Shows handling of single trajectory data
    """
    print("\n" + "="*70)
    print("Example 5: Single Trajectory Analysis")
    print("="*70)
    
    config = MDConfig(
        base_dir=Path('./data/single'),
        output_dir=Path('./output/single'),
        protein_name='ProteinX',
        systems=[
            SystemConfig(
                name='K1',
                dir_pattern='HOLO_K1',  # No {} - single trajectory
                is_apo=False,
                replicates=1  # Single trajectory
            )
        ],
        plot_config=PlotConfig(style='enhanced')
    )
    
    analyzer = MDAnalyzer(config)
    results = analyzer.run_analysis()
    
    print(f"\n✓ Single trajectory analysis complete!")
    print(f"  Output: {results['output_dir']}")


def example_6_amino_acid_ranges():
    """
    Example 6: RMSF with custom amino acid numbering
    
    Shows different ways to specify amino acid ranges
    """
    print("\n" + "="*70)
    print("Example 6: RMSF with Custom AA Numbering")
    print("="*70)
    
    # Method 1: Simple range
    amino_acids_1 = list(range(814, 1167))
    
    # Method 2: Non-contiguous ranges
    amino_acids_2 = list(range(814, 937)) + list(range(994, 1169))
    
    # Method 3: Custom list
    amino_acids_3 = [814, 815, 820, 825, 994, 1000, 1010]
    
    for i, aa_list in enumerate([amino_acids_1, amino_acids_2, amino_acids_3], 1):
        config = MDConfig(
            base_dir=Path('./data'),
            output_dir=Path(f'./output/rmsf_{i}'),
            protein_name='ProteinX',
            systems=[SystemConfig(name='LigandA', dir_pattern='HOLO_{}', replicates=3)],
            amino_acids=aa_list,
            plot_config=PlotConfig(style='simple')
        )
        
        print(f"\nMethod {i}: {len(aa_list)} residues")
        # analyzer = MDAnalyzer(config)
        # results = analyzer.run_analysis()


def main():
    """
    Main function - runs all examples
    
    Comment out examples you don't want to run
    """
    print("\n" + "="*70)
    print(" " * 15 + "GROMACS MD Analysis Toolkit")
    print(" " * 20 + "Complete Examples")
    print("="*70)
    
    # Run examples (comment out as needed)
    # example_1_python_api()
    # example_2_yaml_config()
    # example_3_all_styles()
    # example_4_custom_analysis()
    # example_5_single_trajectory()
    # example_6_amino_acid_ranges()
    
    print("\n" + "="*70)
    print(" " * 15 + "All Examples Complete!")
    print("="*70 + "\n")
    print("To run a specific example, uncomment it in main()")
    print("\nUsage:")
    print("  - Edit main() to enable examples")
    print("  - Adjust data paths to match your setup")
    print("  - Run: python complete_example.py")
    print("\nOr use the CLI:")
    print("  - Interactive: gromacs-md --interactive")
    print("  - YAML: gromacs-md --config my_config.yaml")
    print("  - CLI args: gromacs-md --protein X --base-dir Y --systems ...")


if __name__ == '__main__':
    main()

