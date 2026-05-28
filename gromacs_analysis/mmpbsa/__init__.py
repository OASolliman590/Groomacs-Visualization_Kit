"""
MMPBSA Analysis Module
======================

Consolidated MMPBSA/MMGBSA analysis for:
- result and decomposition table parsing
- replicate summaries
- component and residue-level reports
- publication-quality plots

Features:
---------
- CSV and DAT file parsing
- Single and replicate analysis
- Component and decomposition analysis
- Publication-quality visualizations
- YAML configuration support
"""

from .config import MMPBSAConfig, MMPBSASystemConfig
from .parser import MMPBSAParser
from .processor import MMPBSAProcessor
from .plotter import MMPBSAPlotter
from .analyzer import MMPBSAAnalyzer

__all__ = [
    'MMPBSAConfig',
    'MMPBSASystemConfig',
    'MMPBSAParser',
    'MMPBSAProcessor',
    'MMPBSAPlotter',
    'MMPBSAAnalyzer',
]

