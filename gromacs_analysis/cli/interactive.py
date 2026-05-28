"""
Interactive Terminal Prompt System
==================================

Terminal-based configuration builder with auto-detection.
"""

import logging
from pathlib import Path
from typing import List, Optional
import sys

from ..md.config import MDConfig, SystemConfig, PlotConfig
from ..utils.helpers import (
    detect_replicate_structure,
    parse_amino_acid_range,
    suggest_dir_pattern,
    detect_apo_system
)

logger = logging.getLogger(__name__)


def interactive_prompt(
    default_base_dir: Optional[Path] = None,
    default_output_dir: Optional[Path] = None,
    default_protein_name: Optional[str] = None,
    save_yaml: bool = True,
    save_path: Optional[Path] = None
) -> MDConfig:
    """
    Interactive terminal prompts to build configuration.
    
    Returns:
        MDConfig object
    """
    print("\n" + "="*70)
    print(" " * 15 + "GROMACS MD ANALYSIS TOOLKIT")
    print(" " * 20 + "Interactive Configuration")
    print("="*70 + "\n")
    
    # Step 1: Protein name
    protein_prompt = "Enter protein name"
    if default_protein_name:
        protein_prompt += f" [{default_protein_name}]"
    protein_name = input(f"{protein_prompt}: ").strip() or (default_protein_name or "")
    while not protein_name:
        print("  ⚠ Protein name is required!")
        protein_name = input("Enter protein name: ").strip()
    
    # Step 2: Base directory
    while True:
        base_prompt = "Enter base data directory"
        if default_base_dir:
            base_prompt += f" [{default_base_dir}]"
        base_dir_str = input(f"\n{base_prompt}: ").strip()
        if not base_dir_str and default_base_dir:
            base_dir = Path(default_base_dir).expanduser()
        elif base_dir_str:
            base_dir = Path(base_dir_str).expanduser()
        else:
            print("  ⚠ Base directory is required!")
            continue
        
        if base_dir.exists():
            print(f"  ✓ Directory found: {base_dir}")
            break
        else:
            print(f"  ⚠ Directory not found: {base_dir}")
            retry = input("  Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("\n  Exiting...")
                sys.exit(0)
    
    # Step 3: Auto-detect replicate structure
    print("\n" + "-"*70)
    print("Detecting data structure...")
    print("-"*70)
    
    detection = detect_replicate_structure(base_dir)
    
    has_replicates = False
    num_replicates = 1
    
    if detection and detection['has_replicates']:
        print(f"  ✓ Detected replicate structure:")
        print(f"    Pattern: {detection['pattern']}")
        print(f"    Replicates: {detection['num_replicates']}")
        
        confirm = input(f"\n  Use detected structure? (y/n) [y]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            has_replicates = True
            num_replicates = detection['num_replicates']
        else:
            has_replicates = input("  Do you have replicate data? (y/n): ").strip().lower() == 'y'
            if has_replicates:
                num_replicates = int(input("  Number of replicates: ").strip())
    else:
        print("  ℹ No replicate structure detected")
        has_replicates = input("\n  Do you have replicate data? (y/n) [n]: ").strip().lower() == 'y'
        if has_replicates:
            num_replicates = int(input("  Number of replicates: ").strip())
    
    # Step 4: Ligand names
    print("\n" + "-"*70)
    print("System Configuration")
    print("-"*70)
    
    ligands_input = input("\nEnter ligand name(s) (comma-separated, or leave empty for APO only): ").strip()
    
    systems = []
    
    if ligands_input:
        ligand_names = [l.strip() for l in ligands_input.split(',')]
        
        for ligand in ligand_names:
            print(f"\n  Configuring system: {ligand}")
            
            # Suggest directory pattern
            suggested = suggest_dir_pattern(base_dir, ligand)
            if suggested:
                print(f"    Suggested pattern: {suggested}")
                use_suggested = input(f"    Use suggested pattern? (y/n) [y]: ").strip().lower()
                if use_suggested in ['', 'y', 'yes']:
                    if '{}' in suggested:
                        dir_pattern = suggested
                    else:
                        dir_pattern = suggested if not has_replicates else f"{suggested}_{{}}"
                else:
                    dir_pattern = input(f"    Directory pattern (use {{}} for replicate number): ").strip()
            else:
                if has_replicates:
                    dir_pattern = input(f"    Directory pattern (use {{}} for replicate number): ").strip()
                else:
                    dir_pattern = input(f"    Directory name: ").strip()
            
            systems.append(SystemConfig(
                name=ligand,
                dir_pattern=dir_pattern,
                is_apo=False,
                replicates=num_replicates
            ))
    
    # Step 5: APO system
    print("\n" + "-"*70)
    print("APO System Configuration")
    print("-"*70)
    
    # Auto-detect APO
    detected_apo = detect_apo_system(base_dir)
    
    has_apo = False
    if detected_apo:
        print(f"  ✓ Detected APO system: {detected_apo}")
        has_apo = input("  Include APO system? (y/n) [y]: ").strip().lower() in ['', 'y', 'yes']
        apo_pattern = detected_apo
    else:
        has_apo = input("\n  Include APO system? (y/n) [n]: ").strip().lower() == 'y'
        if has_apo:
            if has_replicates:
                apo_pattern = input("  APO directory pattern (use {} for replicate number): ").strip()
            else:
                apo_pattern = input("  APO directory name: ").strip()
    
    if has_apo:
        systems.append(SystemConfig(
            name='APO',
            dir_pattern=apo_pattern,
            is_apo=True,
            replicates=num_replicates
        ))
    
    # Step 6: Amino acid range for RMSF
    print("\n" + "-"*70)
    print("RMSF Configuration")
    print("-"*70)
    
    aa_input = input("\nAmino acid range for RMSF (e.g., '814-1166' or '814-936,994-1168')\n  [press Enter to skip]: ").strip()
    
    amino_acids = None
    if aa_input:
        try:
            amino_acids = parse_amino_acid_range(aa_input)
            print(f"  ✓ Parsed {len(amino_acids)} residues")
        except ValueError as e:
            print(f"  ⚠ Error parsing range: {e}")
            print(f"  Continuing without RMSF amino acid numbering...")
    
    # Step 7: Visualization style
    print("\n" + "-"*70)
    print("Visualization Configuration")
    print("-"*70)
    
    print("\nAvailable styles:")
    print("  1. simple       - Clean lines with error bands (default)")
    print("  2. enhanced     - With mean lines and annotations")
    print("  3. publication  - High-quality for papers")
    print("  4. overview     - Multi-panel (2x2 grid)")
    print("  5. comparative  - With statistical comparison")
    
    style_input = input("\nSelect style (1-5) [1]: ").strip()
    style_map = {'1': 'simple', '2': 'enhanced', '3': 'publication', '4': 'overview', '5': 'comparative', '': 'simple'}
    style = style_map.get(style_input, 'simple')
    
    print("\nAvailable templates:")
    print("  1. plotly_white (default)")
    print("  2. ggplot2")
    print("  3. seaborn")
    print("  4. simple_white")
    
    template_input = input("\nSelect template (1-4) [1]: ").strip()
    template_map = {'1': 'plotly_white', '2': 'ggplot2', '3': 'seaborn', '4': 'simple_white', '': 'plotly_white'}
    template = template_map.get(template_input, 'plotly_white')
    
    # Step 8: Output directory
    default_output = default_output_dir if default_output_dir else Path("./output")
    output_dir_str = input(f"\nOutput directory [{default_output}]: ").strip()
    output_dir = Path(output_dir_str) if output_dir_str else Path(default_output)
    
    # Step 9: Summary and confirmation
    print("\n" + "="*70)
    print(" " * 25 + "CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nProtein:          {protein_name}")
    print(f"Base Directory:   {base_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Replicates:       {num_replicates}")
    print(f"\nSystems ({len(systems)}):")
    for sys in systems:
        sys_type = "APO" if sys.is_apo else "HOLO"
        print(f"  - {sys.name} ({sys_type}): {sys.dir_pattern}")
    print(f"\nVisualization:")
    print(f"  Style:    {style}")
    print(f"  Template: {template}")
    if amino_acids:
        print(f"\nRMSF Residues: {len(amino_acids)} residues")
    
    print("\n" + "="*70)
    
    proceed = input("\nProceed with analysis? (y/n) [y]: ").strip().lower()
    if proceed not in ['', 'y', 'yes']:
        print("\n  Analysis cancelled.")
        sys.exit(0)
    
    # Create plot config
    plot_config = PlotConfig(
        template=template,
        style=style
    )
    
    # Create MDConfig
    config = MDConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        protein_name=protein_name,
        systems=systems,
        plot_config=plot_config,
        amino_acids=amino_acids
    )
    
    # Offer to save config
    print("\n" + "-"*70)
    if save_yaml:
        save_config = input("Save this configuration to YAML file? (y/n) [y]: ").strip().lower()
        if save_config in ['', 'y', 'yes']:
            from ..config.yaml_loader import save_config_to_yaml
            config_file = save_path or Path(f"{protein_name.lower()}_analysis_config.yaml")
            save_config_to_yaml(config, config_file)
            print(f"  ✓ Configuration saved to: {config_file}")
            print(f"    You can run this again with: gromacs-md --config {config_file}")
    
    print("\n" + "="*70)
    print(" " * 20 + "Starting Analysis...")
    print("="*70 + "\n")
    
    return config

