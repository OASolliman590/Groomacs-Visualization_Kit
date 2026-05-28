"""
MD Data Processor Module
========================

Consolidated data processing combining features from:
- holo_md_analysis.py (single trajectory)
- holo_apo_vis.py (replicate averaging)
- mds_data_parser_replicates.py (multi-ligand)

Handles both single and replicate trajectory data with flexible file finding.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import MDConfig, SystemConfig, AnalysisMetric

logger = logging.getLogger(__name__)


class MDDataProcessor:
    """
    Processes MD trajectory data files with support for both single and replicate modes.
    
    Features:
    - Flexible file finding (pattern matching)
    - Multiple data formats (single/two column)
    - Replicate averaging with error bands
    - Single trajectory mode
    - Outlier detection and removal
    - Statistical analysis
    - Ligand name extraction from filenames
    
    Attributes:
        config: MDConfig object with analysis settings
    """
    
    def __init__(self, config: MDConfig):
        """
        Initialize data processor with configuration.
        
        Args:
            config: MDConfig object
        """
        self.config = config
        logger.info("MDDataProcessor initialized")
    
    # ==================== FILE FINDING ====================
    
    def find_data_files(self) -> Dict[str, Dict[str, Dict]]:
        """
        Find all data files for all systems and metrics.
        
        Returns:
            Nested dict: {system_name: {metric_name: {replicate_info}}}
            
        Example:
            {
                'LigandA': {
                    'rmsd_prot': {
                        'files': [Path1, Path2, Path3],  # If replicates
                        'replicate': 3
                    },
                    'rmsd_lig': {
                        'files': [Path1],
                        'replicate': 1
                    }
                }
            }
        """
        all_found_files = {}
        
        for system in self.config.systems:
            logger.info(f"Finding files for system: {system.name}")
            system_files = {}
            
            for metric in self.config.metrics:
                # Skip HOLO-only metrics for APO systems
                if system.is_apo and metric.is_holo_only:
                    logger.debug(f"Skipping {metric.name} for APO system")
                    continue
                
                found = self._find_metric_files(system, metric)
                if found['files']:
                    system_files[metric.name] = found
                    logger.info(f"  {metric.name}: Found {len(found['files'])} file(s)")
                else:
                    logger.warning(f"  {metric.name}: No files found")
            
            all_found_files[system.name] = system_files
        
        return all_found_files
    
    def _find_metric_files(self, system: SystemConfig, metric: AnalysisMetric) -> Dict:
        """
        Find files for a specific metric and system with enhanced pattern fallback.
        
        Handles both:
        - Single trajectory: base_dir/system_name/RMSD_lig.dat
        - Replicates: base_dir/system_name_1/RMSD_lig.dat, system_name_2/..., etc.
        
        Args:
            system: SystemConfig object
            metric: AnalysisMetric object
            
        Returns:
            Dict with 'files' (list of Paths) and 'replicates' (int)
        """
        found_files = []
        
        if system.dir_names:
            # Explicit replicate directories
            for dir_name in system.dir_names:
                system_dir = self.config.base_dir / dir_name
                if not system_dir.exists():
                    logger.warning(f"    Replicate directory not found: {system_dir}")
                    continue

                patterns = [p.strip() for p in metric.file_pattern.split(',')]
                file_path = self._find_file_with_pattern_fallback(system_dir, patterns)
                if file_path:
                    found_files.append(file_path)
                    logger.debug(f"    Replicate {dir_name}: {file_path}")

        elif system.replicates == 1:
            # Single trajectory mode
            # Look in: base_dir/system_dir_pattern/metric_file
            system_dir = self.config.base_dir / system.dir_pattern.format('')
            
            # Handle multiple patterns (comma-separated)
            patterns = [p.strip() for p in metric.file_pattern.split(',')]
            
            # Try each pattern with fallback
            file_path = self._find_file_with_pattern_fallback(system_dir, patterns)
            if file_path:
                found_files.append(file_path)
                logger.debug(f"    Found: {file_path}")
        
        else:
            # Replicate mode
            # Look in: base_dir/system_dir_pattern_1/, system_dir_pattern_2/, etc.
            for rep in range(1, system.replicates + 1):
                system_dir = self.config.base_dir / system.dir_pattern.format(rep)
                
                if not system_dir.exists():
                    logger.warning(f"    Replicate directory not found: {system_dir}")
                    continue
                
                # Handle multiple patterns
                patterns = [p.strip() for p in metric.file_pattern.split(',')]
                
                # Try to find file with pattern fallback
                file_path = self._find_file_with_pattern_fallback(system_dir, patterns)
                if file_path:
                    found_files.append(file_path)
                    logger.debug(f"    Replicate {rep}: {file_path}")
        
        return {
            'files': found_files,
            'replicates': len(found_files)
        }
    
    def _find_file_with_pattern_fallback(self, directory: Path, patterns: List[str]) -> Optional[Path]:
        """
        Find a file in directory that matches any of the given patterns.
        Enhanced with pattern fallback from holo_apo_vis.py.
        
        Args:
            directory: Directory to search in
            patterns: List of glob patterns to try
            
        Returns:
            First matching file path or None
        """
        if not directory.exists():
            return None
        
        # Try each pattern
        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # For RMSD ligand pattern (RMSD_.dat), exclude files with 'apo' or 'complex' in name
                    filename_lower = file_path.name.lower()
                    pattern_lower = pattern.lower()
                    is_rmsd = 'rmsd' in pattern_lower
                    is_specific_rmsd = any(token in pattern_lower for token in ['apo', 'complex', 'protein', 'lig', 'ligand'])
                    if is_rmsd and not is_specific_rmsd:
                        if any(token in filename_lower for token in ['apo', 'complex', 'protein']):
                            continue
                    return file_path
        
        return None
    
    # ==================== FILE PARSING ====================
    
    def parse_data_file(self, file_path: Path, data_format: str = 'two_column') -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a single data file.
        
        Handles multiple formats:
        - 'single_column': Just values (one per line)
        - 'two_column': Time and value (space-separated)
        - Auto-skips comments (@, #)
        
        Args:
            file_path: Path to data file
            data_format: 'single_column' or 'two_column'
            
        Returns:
            Tuple of (time_array, value_array)
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return self._parse_content(content, data_format, filename=file_path.name)
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return np.array([]), np.array([])
    
    def _parse_content(self, content: str, data_format: str, filename: str = '') -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse file content based on format with advanced cleaning.
        
        Args:
            content: File content as string
            data_format: Format specification
            filename: Filename for metric-specific handling
            
        Returns:
            Tuple of (time_array, value_array)
        """
        if data_format == 'two_column':
            parsed = self._parse_two_column(content)
            if parsed is not None:
                return parsed

        # Use advanced cleaning for better handling
        cleaned_data = self._advanced_cleaning(content, filename, data_format)

        # Fallback for single-column style values.
        value_data = np.array(cleaned_data)
        time_data = np.arange(len(cleaned_data))

        return time_data, value_data

    def _parse_two_column(self, content: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Parse conventional two-column trajectory data preserving the x-axis."""
        time_values: List[float] = []
        metric_values: List[float] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(('@', '#', ';')):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
            except ValueError:
                continue

            time_values.append(x_val)
            metric_values.append(y_val)

        if not metric_values:
            return None

        return np.array(time_values), np.array(metric_values)
    
    def _advanced_cleaning(self, data: str, filename: str, data_format: str) -> List[float]:
        """
        Enhanced cleaning from mds_data_parser with metric-specific logic.
        
        Handles special cases for different metrics:
        - rog: Skip first 2 values (GROMACS header)
        - bonds/RMSF/comcom: Extract every 2nd value if space-separated
        
        Args:
            data: Raw file content
            filename: Filename for detecting metric type
            data_format: Format specification
            
        Returns:
            List of cleaned float values
        """
        filename_lower = filename.lower()
        
        # Check if data has spaces (two-column format)
        if ' ' in data and data_format == 'two_column':
            # Replace newlines with spaces and split
            data_list = data.strip().replace('\n', ' ').split(' ')
            
            # Filter out empty strings and comments
            data_list = [d for d in data_list if d and not d.startswith(('@', '#', ';'))]
            
            final_data = []
            
            # Special handling for bonds, RMSF, comcom (extract 2nd column)
            if any(keyword in filename_lower for keyword in ['bond', 'rmsf', 'comcom']):
                for idx, val in enumerate(data_list):
                    if idx % 2 == 1:  # Take every second value (column 2)
                        try:
                            final_data.append(float(val))
                        except ValueError:
                            continue
            
            # Special handling for rog (skip first 2 values)
            elif 'rog' in filename_lower:
                for val in data_list[2:]:  # Skip first 2 values
                    try:
                        final_data.append(float(val))
                    except ValueError:
                        continue
            
            else:
                # Default: extract second column for two-column format
                for idx, val in enumerate(data_list):
                    if idx % 2 == 1:
                        try:
                            final_data.append(float(val))
                        except ValueError:
                            continue
            
            return final_data
        
        else:
            # Single column or no spaces - direct parsing
            lines = data.strip().split('\n')
            final_data = []
            
            for line in lines:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith(('@', '#', ';')):
                    continue
                
                parts = line.split()
                
                # Try to extract numeric value
                if data_format == 'single_column' and len(parts) >= 1:
                    try:
                        final_data.append(float(parts[0]))
                    except ValueError:
                        continue
                elif data_format == 'two_column' and len(parts) >= 2:
                    try:
                        final_data.append(float(parts[1]))
                    except ValueError:
                        continue
                elif len(parts) == 1:
                    try:
                        final_data.append(float(parts[0]))
                    except ValueError:
                        continue
            
            return final_data
    
    # ==================== DATA PROCESSING ====================
    
    def load_and_process_metric(self, system: SystemConfig, metric: AnalysisMetric, 
                                 file_info: Dict) -> Optional[Dict]:
        """
        Load and process data for a specific metric.
        
        Handles both single trajectory and replicates automatically.
        
        Args:
            system: SystemConfig object
            metric: AnalysisMetric object
            file_info: Dict from find_data_files with 'files' and 'replicates'
            
        Returns:
            Dict with processed data:
            - For single: {'time': array, 'values': array, 'mode': 'single', 'stats': dict}
            - For replicates: {'time': array, 'mean': array, 'upper': array, 
                               'lower': array, 'mode': 'replicate', 'stats': dict}
        """
        files = file_info['files']
        num_replicates = file_info['replicates']
        
        if not files:
            logger.warning(f"No files to process for {system.name}/{metric.name}")
            return None
        
        if num_replicates == 1:
            # Single trajectory mode
            return self._process_single_trajectory(files[0], metric)
        else:
            # Replicate mode
            return self._process_replicates(files, metric)
    
    def _process_single_trajectory(self, file_path: Path, metric: AnalysisMetric) -> Dict:
        """
        Process a single trajectory file.
        
        Args:
            file_path: Path to data file
            metric: AnalysisMetric object
            
        Returns:
            Dict with time, values, mode, and stats
        """
        time_data, value_data = self.parse_data_file(file_path, metric.data_format)

        if len(value_data) == 0:
            logger.warning(f"No valid data in {file_path}")
            return None

        time_data, time_metadata = self._convert_time_axis(time_data, metric)
        
        # Clean data and apply optional smoothing
        time_clean, value_clean = self._remove_outliers(
            time_data,
            value_data,
            threshold=self.config.outlier_threshold,
        )
        value_clean = self._apply_smoothing(value_clean)
        
        # Calculate statistics
        stats = self._calculate_statistics(value_clean)
        
        return {
            'time': time_clean,
            'values': value_clean,
            'mode': 'single',
            'stats': stats,
            'n_points': len(value_clean),
            'time_metadata': time_metadata,
        }
    
    def _process_replicates(self, file_paths: List[Path], metric: AnalysisMetric) -> Dict:
        """
        Process multiple replicate files and calculate average with error bands.
        
        Args:
            file_paths: List of Paths to replicate files
            metric: AnalysisMetric object
            
        Returns:
            Dict with time, mean, upper, lower bounds, mode, and stats
        """
        replicate_data = []
        replicate_times = []
        
        # Load all replicates
        time_metadata = None
        for file_path in file_paths:
            time_data, value_data = self.parse_data_file(file_path, metric.data_format)
            
            if len(value_data) > 0:
                time_data, metadata = self._convert_time_axis(time_data, metric)
                if time_metadata is None:
                    time_metadata = metadata
                time_clean, value_clean = self._remove_outliers(
                    time_data,
                    value_data,
                    threshold=self.config.outlier_threshold,
                )
                smoothed_values = self._apply_smoothing(value_clean)
                replicate_times.append(np.array(time_clean[:len(smoothed_values)]))
                replicate_data.append(smoothed_values)
            else:
                logger.warning(f"Empty replicate: {file_path}")
        
        if not replicate_data:
            logger.error("No valid replicate data found")
            return None
        
        # Ensure all replicates have same length (truncate to minimum)
        min_length = min(len(rep) for rep in replicate_data)
        replicate_data = [rep[:min_length] for rep in replicate_data]
        
        # Convert to numpy array (rows=replicates, cols=time points)
        replicate_array = np.array(replicate_data)
        
        # Calculate statistics across replicates
        mean_values = replicate_array.mean(axis=0)
        std_values = replicate_array.std(axis=0)
        upper_bound = mean_values + std_values
        lower_bound = mean_values - std_values
        
        # Preserve trajectory x-axis from first replicate.
        time_array = np.array(replicate_times[0][:min_length])
        for idx, rep_time in enumerate(replicate_times[1:], start=2):
            if not np.allclose(rep_time[:min_length], time_array, equal_nan=True):
                logger.warning(
                    "Replicate %s has a different x-axis for %s; using first replicate axis.",
                    idx,
                    metric.name,
                )
                break
        
        # Calculate overall statistics
        stats = self._calculate_statistics(mean_values)
        stats['std_mean'] = np.mean(std_values)
        stats['n_replicates'] = len(replicate_data)
        
        return {
            'time': time_array,
            'mean': mean_values,
            'upper': upper_bound,
            'lower': lower_bound,
            'mode': 'replicate',
            'stats': stats,
            'n_points': min_length,
            'n_replicates': len(replicate_data),
            'replicates': replicate_data,
            'time_metadata': time_metadata,
        }

    def _convert_time_axis(self, time_data: np.ndarray, metric: AnalysisMetric) -> Tuple[np.ndarray, Dict]:
        """Convert trajectory x-axis to configured output unit and report provenance metadata."""
        if len(time_data) == 0:
            return time_data, {
                "input_unit": self.config.time_unit_input,
                "output_unit": self.config.time_unit_output,
                "applied_scale": 1.0,
                "time_step_ps": self.config.time_step_ps,
                "inferred_input_unit": None,
                "conversion_mode": "none",
            }

        if self.config.time_scale is not None:
            converted = np.asarray(time_data, dtype=float) * float(self.config.time_scale)
            return converted, {
                "input_unit": self.config.time_unit_input,
                "output_unit": self.config.time_unit_output,
                "applied_scale": float(self.config.time_scale),
                "time_step_ps": self.config.time_step_ps,
                "inferred_input_unit": self._infer_time_unit(time_data, metric),
                "conversion_mode": "explicit_scale",
            }

        inferred_input = (
            self._infer_time_unit(time_data, metric)
            if self.config.time_unit_input == "auto"
            else self.config.time_unit_input
        )
        output_unit = inferred_input if self.config.time_unit_output == "auto" else self.config.time_unit_output

        converted = np.asarray(time_data, dtype=float)
        scale = 1.0

        ps_values: Optional[np.ndarray]
        ps_values = None
        if inferred_input == "ps":
            ps_values = converted
        elif inferred_input == "ns":
            ps_values = converted * 1000.0
        elif inferred_input == "frame" and self.config.time_step_ps:
            ps_values = converted * float(self.config.time_step_ps)

        if output_unit == inferred_input:
            pass
        elif output_unit == "ps" and ps_values is not None:
            converted = ps_values
            scale = self._safe_scale(time_data, converted)
        elif output_unit == "ns" and ps_values is not None:
            converted = ps_values / 1000.0
            scale = self._safe_scale(time_data, converted)
        elif output_unit == "frame" and inferred_input != "frame" and self.config.time_step_ps:
            if ps_values is not None:
                converted = ps_values / float(self.config.time_step_ps)
                scale = self._safe_scale(time_data, converted)
        else:
            logger.warning(
                "Could not convert time axis from %s to %s for metric %s; keeping original values.",
                inferred_input,
                output_unit,
                metric.name,
            )
            output_unit = inferred_input

        return converted, {
            "input_unit": self.config.time_unit_input,
            "output_unit": output_unit,
            "applied_scale": float(scale),
            "time_step_ps": self.config.time_step_ps,
            "inferred_input_unit": inferred_input,
            "conversion_mode": "unit_conversion",
        }

    @staticmethod
    def _safe_scale(original: np.ndarray, converted: np.ndarray) -> float:
        if len(original) == 0:
            return 1.0
        if np.allclose(original, 0):
            return 1.0
        idx = np.where(np.abs(original) > 0)[0]
        if len(idx) == 0:
            return 1.0
        i = int(idx[0])
        return float(converted[i] / original[i])

    def _infer_time_unit(self, time_data: np.ndarray, metric: AnalysisMetric) -> str:
        """Infer best-effort unit for parsed x-axis."""
        arr = np.asarray(time_data, dtype=float)
        if len(arr) <= 1:
            return "frame"
        if metric.data_format == "single_column" and np.allclose(arr, np.arange(len(arr))):
            return "frame"

        max_val = float(np.nanmax(arr))
        if max_val > 1000:
            return "ps"
        if np.allclose(arr, np.round(arr)) and np.isclose(float(np.nanmedian(np.diff(arr))), 1.0):
            return "frame"
        return "ns"
    
    # ==================== STATISTICS & UTILITIES ====================
    
    def _remove_outliers(self, time_data: np.ndarray, value_data: np.ndarray,
                        threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers using Z-score method.
        
        Args:
            time_data: Time array
            value_data: Value array
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Tuple of (cleaned_time, cleaned_values)
        """
        if len(value_data) < 3:
            return time_data, value_data

        if np.nanstd(value_data) == 0:
            return time_data, value_data
        
        try:
            z_scores = np.abs(scipy_stats.zscore(value_data))
            if not np.all(np.isfinite(z_scores)):
                return time_data, value_data
            mask = z_scores < threshold
            
            n_outliers = len(value_data) - np.sum(mask)
            if n_outliers > 0:
                logger.info(f"Removed {n_outliers} outliers (threshold={threshold})")
            
            return time_data[mask], value_data[mask]
        except Exception as e:
            logger.warning(f"Outlier removal failed: {e}")
            return time_data, value_data

    def _apply_smoothing(self, values: np.ndarray) -> np.ndarray:
        """Apply centered rolling mean when smoothing_window is greater than 1."""
        window = max(1, int(getattr(self.config, "smoothing_window", 1)))
        if window <= 1 or len(values) < window:
            return values

        smoothed = (
            pd.Series(values)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        return smoothed
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict:
        """
        Calculate comprehensive statistics for data.
        
        Args:
            data: Numpy array of values
            
        Returns:
            Dict with mean, std, min, max, median, q1, q3, count
        """
        if len(data) == 0:
            return {}
        
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'count': len(data)
        }
    
    def extract_ligand_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract ligand name from filename.
        
        Handles patterns like:
        - RMSD_K1.dat -> 'K1'
        - RMSD_LigandA_r1.dat -> 'LigandA'
        - comcom_complex_LigandA.dat -> 'LigandA'
        - RMSD_apo_PLpro.dat -> 'APO'
        
        Args:
            filename: Filename string
            
        Returns:
            Ligand name or None
        """
        # Remove extension
        name = filename.replace('.dat', '').replace('.xvg', '')
        
        # Check for APO
        if 'apo' in name.lower():
            return 'APO'
        
        # Split by underscore
        parts = name.split('_')
        
        # Remove metric names (common prefixes)
        metric_names = ['RMSD', 'RMSF', 'comcom', 'hbonds', 'rog', 'RoG', 
                       'SASA', 'sasa', 'complex', 'protein', 'ligand']
        
        filtered_parts = []
        for part in parts:
            if part not in metric_names and not part.startswith('r') or not part[1:].isdigit():
                filtered_parts.append(part)
        
        # Return last meaningful part
        if filtered_parts:
            # Check if last part is replicate number (r1, r2, etc.)
            if filtered_parts[-1].startswith('r') and filtered_parts[-1][1:].isdigit():
                return filtered_parts[-2] if len(filtered_parts) > 1 else None
            return filtered_parts[-1]
        
        return None
    
    def prepare_amino_acid_axis(self) -> Optional[List[str]]:
        """
        Prepare residue labels for RMSF x-axis.

        Uses config.amino_acids (numbers or labels) if provided.
        Otherwise tries to read residue labels from topology/PDB, preferring
        `step3_input.pdb` in system directories (legacy-style behavior),
        then falls back to residue_range.

        Returns:
            List of residue label strings or None
        """
        if self.config.amino_acids:
            return [str(aa) for aa in self.config.amino_acids]

        if self.config.sequence_topology:
            try:
                from ..utils.sequence import (
                    load_sequence_from_topology,
                    build_residue_labels,
                    load_sequence_from_pdb,
                )

                topology_path = Path(self.config.sequence_topology)
                if topology_path.suffix.lower() == ".pdb":
                    residues = load_sequence_from_pdb(topology_path)
                else:
                    residues = load_sequence_from_topology(
                        topology_path,
                        selection=self.config.sequence_selection,
                    )
                return build_residue_labels(residues, include_names=True)
            except Exception as exc:
                logger.warning(f"Failed to load sequence from topology: {exc}")

        # Legacy-compatible fallback: auto-detect step3_input.pdb from systems.
        for pdb_path in self._candidate_step3_pdb_paths():
            try:
                from ..utils.sequence import load_sequence_from_pdb, build_residue_labels

                residues = load_sequence_from_pdb(pdb_path)
                labels = build_residue_labels(residues, include_names=True)
                if labels:
                    logger.info(f"Loaded RMSF residue labels from PDB: {pdb_path}")
                    return labels
            except Exception as exc:
                logger.warning(f"Failed to load RMSF labels from {pdb_path}: {exc}")
        
        if self.config.residue_range:
            start, end = self.config.residue_range
            return [str(i) for i in range(start, end + 1)]
        
        return None

    def _candidate_step3_pdb_paths(self) -> List[Path]:
        """Collect candidate step3 PDB files from configured system directories."""
        candidates: List[Path] = []

        for system in self.config.systems:
            dirs: List[Path] = []

            if system.dir_names:
                dirs = [self.config.base_dir / d for d in system.dir_names]
            elif system.replicates > 1:
                dirs = [self.config.base_dir / system.dir_pattern.format(rep) for rep in range(1, system.replicates + 1)]
            else:
                dirs = [self.config.base_dir / system.dir_pattern.format('')]

            for system_dir in dirs:
                candidate = system_dir / "step3_input.pdb"
                if candidate.exists():
                    candidates.append(candidate)

        # Preserve order and de-duplicate.
        unique: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique
    
    def load_all_data(self) -> Dict:
        """
        Load and process all data for all systems and metrics.
        
        Convenience method that combines file finding and processing.
        
        Returns:
            Nested dict: {system_name: {metric_name: processed_data_dict}}
        """
        # Find all files
        all_files = self.find_data_files()
        
        # Process all data
        all_processed_data = {}
        
        for system in self.config.systems:
            system_name = system.name
            
            if system_name not in all_files:
                logger.warning(f"No files found for system: {system_name}")
                continue
            
            system_data = {}
            
            for metric in self.config.metrics:
                # Skip if not found
                if metric.name not in all_files[system_name]:
                    continue
                
                file_info = all_files[system_name][metric.name]
                processed = self.load_and_process_metric(system, metric, file_info)
                
                if processed:
                    system_data[metric.name] = processed
            
            all_processed_data[system_name] = system_data

        # Apply stability window trimming if enabled
        if self.config.stability.enabled:
            self._apply_stability_window(all_processed_data)
        
        return all_processed_data

    def _apply_stability_window(self, all_data: Dict) -> None:
        """Trim metrics to stable window based on configured metric."""
        target_metric = self.config.stability.metric
        window = max(5, int(self.config.stability.window))
        std_threshold = float(self.config.stability.std_threshold)
        slope_threshold = float(self.config.stability.slope_threshold)
        min_points = int(self.config.stability.min_points)

        for system_name, system_data in all_data.items():
            if target_metric not in system_data:
                continue
            metric_data = system_data[target_metric]

            if metric_data.get("mode") == "replicate":
                series = metric_data.get("mean")
                time = metric_data.get("time")
            else:
                series = metric_data.get("values")
                time = metric_data.get("time")

            if series is None or time is None or len(series) < min_points:
                continue

            start_idx = self._detect_stable_start(time, series, window, std_threshold, slope_threshold)
            if start_idx <= 0:
                continue

            for metric_name, data in system_data.items():
                if not self.config.stability.apply_to_all_metrics and metric_name != target_metric:
                    continue
                self._trim_metric_data(data, start_idx)

            system_data["stable_start_index"] = start_idx
            system_data["stable_start_time"] = float(time[start_idx]) if len(time) > start_idx else None

    def _detect_stable_start(
        self,
        time: np.ndarray,
        values: np.ndarray,
        window: int,
        std_threshold: float,
        slope_threshold: float,
    ) -> int:
        """Detect the start index of a stable window."""
        if len(values) <= window:
            return 0

        for idx in range(0, len(values) - window):
            window_vals = values[idx: idx + window]
            window_time = time[idx: idx + window]
            if np.std(window_vals) > std_threshold:
                continue
            # linear fit slope
            try:
                coeffs = np.polyfit(window_time, window_vals, 1)
                slope = coeffs[0]
            except Exception:
                slope = 0.0
            if abs(slope) <= slope_threshold:
                return idx

        return 0

    def _trim_metric_data(self, data: Dict, start_idx: int) -> None:
        """Trim metric data arrays to start from start_idx and recompute stats."""
        if data.get("mode") == "single":
            data["time"] = data["time"][start_idx:]
            data["values"] = data["values"][start_idx:]
            data["stats"] = self._calculate_statistics(data["values"])
            data["n_points"] = len(data["values"])
            return

        if data.get("mode") == "replicate":
            data["time"] = data["time"][start_idx:]
            data["mean"] = data["mean"][start_idx:]
            data["upper"] = data["upper"][start_idx:]
            data["lower"] = data["lower"][start_idx:]
            if "replicates" in data:
                data["replicates"] = [rep[start_idx:] for rep in data["replicates"]]
                rep_array = np.array(data["replicates"])
                data["mean"] = rep_array.mean(axis=0)
                std = rep_array.std(axis=0)
                data["upper"] = data["mean"] + std
                data["lower"] = data["mean"] - std
            data["stats"] = self._calculate_statistics(data["mean"])
            data["n_points"] = len(data["mean"])
