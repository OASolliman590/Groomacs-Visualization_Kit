"""
MD Trajectory Analysis Module
==============================

Consolidated MD trajectory analysis combining features from:
- holo_md_analysis.py
- apo_md_analysis.py
- holo_apo_vis.py
- mds_data_parser_replicates.py

Features:
---------
- Single trajectory analysis (APO or HOLO)
- Multi-replicate averaging with error bands
- APO vs HOLO comparison
- Multiple ligand comparison (N ligands)
- Configurable via JSON or Python objects
- Publication-quality plots
"""

from .config import MDConfig, SystemConfig, AnalysisMetric, PlotConfig
from .data_processor import MDDataProcessor
from .plotter import MDPlotter
from .analyzer import MDAnalyzer

__all__ = [
    'MDConfig',
    'SystemConfig',
    'AnalysisMetric',
    'PlotConfig',
    'MDDataProcessor',
    'MDPlotter',
    'MDAnalyzer',
]

