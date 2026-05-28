"""Interaction fingerprint analyzer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import InteractionConfig, InteractionCompareConfig
from .processor import InteractionProcessor
from .plotter import InteractionPlotter

logger = logging.getLogger(__name__)


class InteractionAnalyzer:
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.processor = InteractionProcessor(config)
        self.plotter = InteractionPlotter(config.plot_config, config.protein_name)

    def run_analysis(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Starting Interaction Fingerprint Analysis")
        logger.info("=" * 60)

        self._create_output_dirs()
        results = self.processor.compute_all()

        data_dir = self.config.output_dir / "data"
        plots_dir = self.config.output_dir / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save per-interaction CSV
        rows = []
        for interaction, residues in results.items():
            for res in residues:
                rows.append({
                    "Interaction": interaction,
                    "Residue": res["label"],
                    "ResidueID": res["resid"],
                    "Occupancy": res["occupancy"],
                })
        summary_path = data_dir / "interaction_fingerprint.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)

        # Heatmap (interaction x residue)
        residue_labels = sorted({r["label"] for res in results.values() for r in res})
        interaction_types = list(results.keys())
        matrix = np.zeros((len(interaction_types), len(residue_labels)))
        label_index = {label: idx for idx, label in enumerate(residue_labels)}
        for i, interaction in enumerate(interaction_types):
            for res in results[interaction]:
                matrix[i, label_index[res["label"]]] = res["occupancy"]

        fig = self.plotter.create_interaction_heatmap(
            residue_labels,
            interaction_types,
            matrix,
            title=f"{self.config.protein_name}-{self.config.ligand_name} Interaction Fingerprint",
        )
        fig.write_html(str(plots_dir / "interaction_fingerprint.html"))
        try:
            fig.write_image(str(plots_dir / "interaction_fingerprint.svg"))
        except Exception:
            pass

        # Bar charts per interaction
        for interaction, residues in results.items():
            if not residues:
                continue
            fig = self.plotter.create_occupancy_bar(
                residues,
                interaction,
                title=f"{self.config.ligand_name} {interaction} occupancy",
            )
            fig.write_html(str(plots_dir / f"{interaction}_occupancy.html"))
            try:
                fig.write_image(str(plots_dir / f"{interaction}_occupancy.svg"))
            except Exception:
                pass

        return {
            "success": True,
            "output_dir": str(self.config.output_dir),
            "summary_file": str(summary_path),
        }

    def _create_output_dirs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)


class InteractionComparisonAnalyzer:
    def __init__(self, config: InteractionCompareConfig):
        self.config = config
        self.plotter = InteractionPlotter(config.plot_config, config.protein_name)

    def run_analysis(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Starting Interaction Comparison")
        logger.info("=" * 60)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.config.output_dir / "data"
        plots_dir = self.config.output_dir / "plots"
        data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        systems_data: Dict[str, Dict[str, List[Dict]]] = {}
        for cfg in self.config.systems:
            results = InteractionProcessor(cfg).compute_all()
            systems_data[cfg.ligand_name] = results

        # Build comparison table
        ligand_names = list(systems_data.keys())
        interaction_types = sorted({i for data in systems_data.values() for i in data.keys()})

        rows = []
        for interaction in interaction_types:
            label_sets = []
            for ligand in ligand_names:
                labels = {res["label"] for res in systems_data[ligand].get(interaction, [])}
                label_sets.append(labels)

            if not label_sets:
                continue

            if self.config.compare_mode == "intersection":
                residue_labels = sorted(set.intersection(*label_sets))
            else:
                residue_labels = sorted(set.union(*label_sets))

            for label in residue_labels:
                row = {"Interaction": interaction, "Residue": label}
                for ligand in ligand_names:
                    occ = 0.0
                    for res in systems_data[ligand].get(interaction, []):
                        if res["label"] == label:
                            occ = res["occupancy"]
                            break
                    row[ligand] = occ
                rows.append(row)

        summary_path = data_dir / "interaction_compare.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)

        # Heatmap per interaction (ligands x residues)
        for interaction in interaction_types:
            label_sets = []
            for ligand in ligand_names:
                labels = {res["label"] for res in systems_data[ligand].get(interaction, [])}
                label_sets.append(labels)

            if not label_sets:
                continue

            if self.config.compare_mode == "intersection":
                residue_labels = sorted(set.intersection(*label_sets))
            else:
                residue_labels = sorted(set.union(*label_sets))

            if not residue_labels:
                continue

            matrix = np.zeros((len(ligand_names), len(residue_labels)))
            for i, ligand in enumerate(ligand_names):
                for res in systems_data[ligand].get(interaction, []):
                    if res["label"] in residue_labels:
                        j = residue_labels.index(res["label"])
                        matrix[i, j] = res["occupancy"]
            fig = self.plotter.create_interaction_heatmap(
                residue_labels,
                ligand_names,
                matrix,
                title=f"{self.config.protein_name} {interaction} occupancy (ligands)",
            )
            fig.write_html(str(plots_dir / f"interaction_{interaction}_compare.html"))
            try:
                fig.write_image(str(plots_dir / f"interaction_{interaction}_compare.svg"))
            except Exception:
                pass

        return {
            "success": True,
            "output_dir": str(self.config.output_dir),
            "summary_file": str(summary_path),
        }
