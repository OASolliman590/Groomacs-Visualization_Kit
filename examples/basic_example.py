#!/usr/bin/env python3
"""
Basic Example - MD Trajectory Analysis

This example shows how to use the consolidated GROMACS Analysis Toolkit
to analyze MD trajectories with replicates.
"""

from pathlib import Path
from gromacs_analysis import MDAnalyzer
from gromacs_analysis.md import MDConfig, SystemConfig

# Method 1: Load from JSON config
# ================================
config = MDConfig.from_json('example_config.json')
analyzer = MDAnalyzer(config)
results = analyzer.run_analysis()

print(f"Analysis complete!")
print(f"Plots created: {len(results['plots_created'])}")
print(f"Output directory: {results['output_directory']}")


# Method 2: Create config in Python
# ==================================
config = MDConfig(
    base_dir=Path('./data'),
    output_dir=Path('./output'),
    protein_name='ProteinX',
    systems=[
        SystemConfig(
            name='LigandA',
            dir_pattern='14-ProteinX_HOLO_{}',
            is_apo=False,
            replicates=3
        ),
        SystemConfig(
            name='APO',
            dir_pattern='14-ProteinX_APO_{}',
            is_apo=True,
            replicates=3
        )
    ],
    residue_range=(814, 1166)
)

analyzer = MDAnalyzer(config)
results = analyzer.run_analysis()


# Method 3: Using CLI (from command line)
# ========================================
# gromacs-md --config example_config.json --output ./output
#
# Or:
# gromacs-md \
#   --base-dir ./data \
#   --protein ProteinX \
#   --systems LigandA:14-ProteinX_HOLO_{} APO:14-ProteinX_APO_{}:apo \
#   --replicates 3 \
#   --output ./output


