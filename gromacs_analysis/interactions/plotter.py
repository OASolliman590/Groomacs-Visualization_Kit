"""Plotter for interaction fingerprints."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


class InteractionPlotter:
    def __init__(self, plot_config, protein_name: str):
        self.plot_config = plot_config
        self.protein_name = protein_name
        pio.templates.default = plot_config.template

    def create_interaction_heatmap(
        self,
        residue_labels: List[str],
        interaction_types: List[str],
        occupancy_matrix: np.ndarray,
        title: str,
    ) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=occupancy_matrix,
                    x=residue_labels,
                    y=interaction_types,
                    colorscale="Viridis",
                    coloraxis="coloraxis",
                    zsmooth="best",
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title="Residue",
            yaxis_title="Interaction",
            font_family=self.plot_config.font_family,
            font_size=self.plot_config.font_size,
            template=self.plot_config.template,
        )
        return fig

    def create_occupancy_bar(self, residues: List[Dict], interaction: str, title: str) -> go.Figure:
        labels = [r["label"] for r in residues]
        values = [r["occupancy"] for r in residues]
        fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        fig.update_layout(
            title=title,
            xaxis_title="Residue",
            yaxis_title="Occupancy (%)",
            font_family=self.plot_config.font_family,
            font_size=self.plot_config.font_size,
            template=self.plot_config.template,
        )
        return fig
