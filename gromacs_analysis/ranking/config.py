"""Configuration for ligand ranking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class RankingConfig:
    output_dir: Path
    protein_name: str
    md_stats_file: Optional[Path] = None
    mmpbsa_data_dir: Optional[Path] = None
    distance_summary_file: Optional[Path] = None
    metric_directions: Dict[str, str] = field(default_factory=lambda: {
        "rmsd_prot": "lower",
        "rmsd_complex": "lower",
        "rmsd_lig": "lower",
        "rmsf": "lower",
        "rog": "lower",
        "sasa": "lower",
        "hbonds": "higher",
        "distance_mean": "lower",
        "mmpbsa_total": "lower",
    })

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.md_stats_file:
            self.md_stats_file = Path(self.md_stats_file)
        if self.mmpbsa_data_dir:
            self.mmpbsa_data_dir = Path(self.mmpbsa_data_dir)
        if self.distance_summary_file:
            self.distance_summary_file = Path(self.distance_summary_file)
