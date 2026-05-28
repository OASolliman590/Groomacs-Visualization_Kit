"""Configuration classes for interaction fingerprint analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..md.config import PlotConfig


@dataclass
class InteractionConfig:
    base_dir: Path
    output_dir: Path
    protein_name: str
    ligand_name: str
    topology: Path
    trajectories: List[Path]
    ligand_selection: str = "resname UNK"
    protein_selection: str = "protein"
    interaction_types: List[str] = field(default_factory=lambda: ["hbond", "salt_bridge", "hydrophobic", "pi_stacking"])
    hbond_cutoff: float = 3.5
    salt_bridge_cutoff: float = 4.0
    hydrophobic_cutoff: float = 4.5
    pi_stack_cutoff: float = 5.0
    stride: int = 1
    ligand_pos_sel: Optional[str] = None
    ligand_neg_sel: Optional[str] = None
    ligand_ring_sel: Optional[str] = None
    plot_config: PlotConfig = field(default_factory=lambda: PlotConfig(
        template="ggplot2",
        style="simple",
        width=1200,
        height=600,
        scale=2,
        font_family="Times New Roman",
        font_size=20,
        save_formats=["html", "svg"],
    ))

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.output_dir = Path(self.output_dir)
        self.topology = Path(self.topology)
        self.trajectories = [Path(p) for p in self.trajectories]
        if self.stride < 1:
            raise ValueError("InteractionConfig.stride must be >= 1")


@dataclass
class InteractionCompareConfig:
    output_dir: Path
    protein_name: str
    systems: List[InteractionConfig]
    compare_mode: str = "intersection"
    plot_config: PlotConfig = field(default_factory=lambda: PlotConfig(
        template="ggplot2",
        style="simple",
        width=1200,
        height=600,
        scale=2,
        font_family="Times New Roman",
        font_size=20,
        save_formats=["html", "svg"],
    ))

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.compare_mode not in ("intersection", "union"):
            raise ValueError("InteractionCompareConfig.compare_mode must be 'intersection' or 'union'")
        if not self.systems:
            raise ValueError("InteractionCompareConfig.systems must not be empty")
