"""
Helper Functions
================

Utility functions for detection, parsing, and validation.
"""

import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os

logger = logging.getLogger(__name__)


def detect_replicate_structure(base_dir: Path) -> Optional[Dict[str, int]]:
    """
    Auto-detect if data is organized in replicate folders.
    
    Looks for patterns like:
    - System_1, System_2, System_3
    - 1/, 2/, 3/
    - System_R1, System_R2, System_R3
    
    Args:
        base_dir: Base directory to check
        
    Returns:
        Dict with detection results:
        {
            'has_replicates': bool,
            'num_replicates': int,
            'pattern': str,  # e.g., '{}_1', '{}_R1', '{}1'
            'detected_systems': list of system names
        }
        or None if no clear pattern detected
    """
    if not base_dir.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return None
    
    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        logger.info("No subdirectories found")
        return None
    
    # Pattern 1: *_1, *_2, *_3, etc.
    pattern1_matches = []
    for d in subdirs:
        match = re.match(r'(.+)_(\d+)$', d.name)
        if match:
            base_name, num = match.groups()
            pattern1_matches.append((base_name, int(num)))
    
    if pattern1_matches:
        # Group by base name
        base_names = {}
        for base_name, num in pattern1_matches:
            if base_name not in base_names:
                base_names[base_name] = []
            base_names[base_name].append(num)
        
        # Check for consecutive numbering
        for base_name, nums in base_names.items():
            nums.sort()
            if nums == list(range(1, len(nums) + 1)):
                logger.info(f"Detected replicate pattern: {base_name}_N (N=1-{len(nums)})")
                return {
                    'has_replicates': True,
                    'num_replicates': len(nums),
                    'pattern': f'{base_name}_{{}}',
                    'detected_systems': [base_name]
                }
    
    # Pattern 2: Just numbers (1/, 2/, 3/)
    number_dirs = [d for d in subdirs if d.name.isdigit()]
    if len(number_dirs) >= 2:
        nums = sorted([int(d.name) for d in number_dirs])
        if nums == list(range(1, len(nums) + 1)):
            logger.info(f"Detected replicate pattern: N/ (N=1-{len(nums)})")
            return {
                'has_replicates': True,
                'num_replicates': len(nums),
                'pattern': '{}',
                'detected_systems': ['']
            }
    
    # Pattern 3: *_R1, *_R2, *_R3, etc.
    pattern3_matches = []
    for d in subdirs:
        match = re.match(r'(.+)_[Rr](\d+)$', d.name)
        if match:
            base_name, num = match.groups()
            pattern3_matches.append((base_name, int(num)))
    
    if pattern3_matches:
        base_names = {}
        for base_name, num in pattern3_matches:
            if base_name not in base_names:
                base_names[base_name] = []
            base_names[base_name].append(num)
        
        for base_name, nums in base_names.items():
            nums.sort()
            if nums == list(range(1, len(nums) + 1)):
                logger.info(f"Detected replicate pattern: {base_name}_RN (N=1-{len(nums)})")
                return {
                    'has_replicates': True,
                    'num_replicates': len(nums),
                    'pattern': f'{base_name}_R{{}}',
                    'detected_systems': [base_name]
                }
    
    logger.info("No clear replicate pattern detected")
    return None


def parse_amino_acid_range(range_str: str) -> List[int]:
    """
    Parse amino acid range strings into list of residue numbers.
    
    Supports formats:
    - "814-1166" -> [814, 815, ..., 1166]
    - "814-936,994-1168" -> [814, ..., 936, 994, ..., 1168]
    - "814,815,816,994,995" -> [814, 815, 816, 994, 995]
    
    Args:
        range_str: String describing residue range
        
    Returns:
        List of residue numbers
        
    Raises:
        ValueError: If format is invalid
    """
    if not range_str or not range_str.strip():
        return []
    
    residues = []
    
    # Split by comma for multiple ranges
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        
        if '-' in part:
            # Range format: start-end
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                
                residues.extend(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format: {part}. Error: {e}")
        else:
            # Single number
            try:
                residues.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid residue number: {part}")
    
    # Remove duplicates and sort
    residues = sorted(list(set(residues)))
    
    logger.info(f"Parsed {len(residues)} residues from range: {range_str}")
    
    return residues


def validate_data_directory(base_dir: Path, expected_patterns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that expected directories/files exist.
    
    Args:
        base_dir: Base directory to check
        expected_patterns: List of expected directory patterns
        
    Returns:
        Tuple of (is_valid, missing_items)
    """
    if not base_dir.exists():
        return False, [f"Base directory does not exist: {base_dir}"]
    
    missing = []
    
    for pattern in expected_patterns:
        # Check if directory exists
        expected_path = base_dir / pattern
        if not expected_path.exists():
            missing.append(pattern)
    
    is_valid = len(missing) == 0
    
    if is_valid:
        logger.info(f"Data directory validation passed: {base_dir}")
    else:
        logger.warning(f"Data directory validation failed. Missing: {missing}")
    
    return is_valid, missing


def create_summary_report(results: Dict, output_file: Path):
    """
    Generate a text summary report.
    
    Args:
        results: Results dictionary from MDAnalyzer
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GROMACS MD ANALYSIS - SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Output Directory: {results['output_dir']}\n")
            f.write(f"Plots Directory: {results['plots_dir']}\n\n")
            
            f.write(f"Systems Analyzed: {results['n_systems']}\n")
            f.write(f"Metrics Processed: {results['n_metrics']}\n")
            f.write(f"Plots Created: {len(results['plots_created'])}\n\n")
            
            f.write("Created Plots:\n")
            for plot in results['plots_created']:
                f.write(f"  - {plot}\n")
            f.write("\n")
            
            if results.get('statistics_file'):
                f.write(f"Statistics File: {results['statistics_file']}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Analysis completed successfully!\n")
            f.write("="*60 + "\n")
        
        logger.info(f"Summary report saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Could not create summary report: {e}")


def suggest_dir_pattern(base_dir: Path, system_name: str) -> Optional[str]:
    """
    Suggest a directory pattern based on existing directories.
    
    Args:
        base_dir: Base directory
        system_name: System name to look for
        
    Returns:
        Suggested pattern or None
    """
    if not base_dir.exists():
        return None
    
    subdirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    # Look for directories containing system name
    matching = [d for d in subdirs if system_name.lower() in d.lower()]
    
    if not matching:
        return None
    
    # Try to extract pattern
    for dir_name in matching:
        # Check if ends with _1, _2, etc.
        match = re.match(r'(.+)_(\d+)$', dir_name)
        if match:
            base, num = match.groups()
            return f"{base}_{{}}"
        
        # Check if ends with R1, R2, etc.
        match = re.match(r'(.+)_[Rr](\d+)$', dir_name)
        if match:
            base, num = match.groups()
            return f"{base}_R{{}}"
    
    # If no pattern, return as-is
    return matching[0]


def count_dat_files(directory: Path) -> int:
    """
    Count .dat files in a directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        Number of .dat files
    """
    if not directory.exists():
        return 0
    
    return len(list(directory.glob('*.dat')))


def detect_apo_system(base_dir: Path) -> Optional[str]:
    """
    Auto-detect APO system from directory names.
    
    Args:
        base_dir: Base directory to check
        
    Returns:
        APO directory pattern if found, None otherwise
    """
    if not base_dir.exists():
        return None
    
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    # Look for directories with 'apo' in the name
    apo_dirs = [d for d in subdirs if 'apo' in d.name.lower()]
    
    if not apo_dirs:
        return None
    
    # Check if it follows replicate pattern
    apo_names = [d.name for d in apo_dirs]
    
    # Pattern: APO_1, APO_2, etc.
    if any('_' in name and name.split('_')[-1].isdigit() for name in apo_names):
        base = apo_names[0].rsplit('_', 1)[0]
        return f"{base}_{{}}"
    
    # Pattern: APO (single)
    if len(apo_dirs) == 1:
        return apo_dirs[0].name
    
    return None


