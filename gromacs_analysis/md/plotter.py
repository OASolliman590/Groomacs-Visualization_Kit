"""
MD Plotter Module
=================

Consolidated plotting functionality with 5 visualization styles:
1. Simple: Lines with error bands (default)
2. Enhanced: With mean lines and annotations
3. Publication: High-quality with professional styling
4. Overview: Multi-panel (2x2 grid)
5. Comparative: With statistical comparison boxes

Combines features from:
- holo_apo_vis.py (simple style, error bands)
- md_analysis_base.py (enhanced style, mean lines)
- MMPBSA_per_res.py (publication style, colors)
- holo_md_analysis.py (overview style, multi-panel)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats as scipy_stats

from .config import MDConfig, AnalysisMetric
from ..utils.mpl_fallback import init_matplotlib, rgb_to_mpl

logger = logging.getLogger(__name__)


class MDPlotter:
    """
    Creates publication-quality plots for MD trajectory analysis.
    
    Supports 5 visualization styles:
    - simple: Clean lines with error bands
    - enhanced: Adds mean lines and hover info
    - publication: Professional styling for papers
    - overview: Multi-panel summary (2x2 grid)
    - comparative: Statistical comparison between systems
    
    Attributes:
        config: MDConfig object with plot settings
        style: Selected visualization style
    """
    
    def __init__(self, config: MDConfig):
        """
        Initialize plotter with configuration.
        
        Args:
            config: MDConfig object
        """
        self.config = config
        self.style = config.plot_config.style
        self.system_color_map = {
            system.name: system.color
            for system in config.systems
            if system.color is not None
        }
        
        # Configure plotly defaults
        pio.defaults.default_format = "svg"
        pio.defaults.default_width = config.plot_config.width
        pio.defaults.default_height = config.plot_config.height
        pio.defaults.default_scale = config.plot_config.scale
        pio.templates.default = config.plot_config.template
        
        logger.info(f"MDPlotter initialized with style: {self.style}")

    @staticmethod
    def _has_values(values) -> bool:
        """Return True when a list-like or array-like object contains values."""
        return values is not None and len(values) > 0

    def _time_axis_label(self) -> str:
        unit = self.config.time_unit_output
        if unit == "auto":
            unit = self.config.time_unit_input if self.config.time_unit_input != "auto" else "time"
        if unit == "frame":
            return "Frame"
        if unit in {"ps", "ns"}:
            return f"Time ({unit})"
        return "Time"
    
    # ==================== MAIN PLOTTING DISPATCHER ====================
    
    def create_plot(self, data: Dict, metric: AnalysisMetric, 
                    x_values: Optional[List] = None) -> go.Figure:
        """
        Create plot based on selected style.
        
        Args:
            data: Dict of {system_name: processed_data_dict}
            metric: AnalysisMetric object
            x_values: Optional custom x-axis values (for RMSF)
            
        Returns:
            Plotly Figure object
        """
        if self.style == 'simple':
            return self._plot_simple(data, metric, x_values)
        elif self.style == 'enhanced':
            return self._plot_enhanced(data, metric, x_values)
        elif self.style == 'publication':
            return self._plot_publication(data, metric, x_values)
        elif self.style == 'comparative':
            return self._plot_comparative(data, metric, x_values)
        else:
            # Default to simple
            return self._plot_simple(data, metric, x_values)
    
    def create_overview_plot(self, all_data: Dict) -> Optional[go.Figure]:
        """
        Create multi-panel overview plot (2x2 grid).
        
        Args:
            all_data: Complete data structure from processor
            
        Returns:
            Plotly Figure or None
        """
        return self._plot_overview(all_data)
    
    # ==================== STYLE 1: SIMPLE (Lines + Error Bands) ====================
    
    def _plot_simple(self, data: Dict, metric: AnalysisMetric, 
                     x_values: Optional[List] = None) -> go.Figure:
        """
        Simple line plots with error bands (from holo_apo_vis.py).
        
        Clean, minimal design with shaded standard deviation regions.
        """
        fig = go.Figure()
        
        # Determine x-axis
        if x_values is None:
            # Use time from first system
            first_system_data = next(iter(data.values()))
            if 'time' in first_system_data:
                x_values = first_system_data['time']
            else:
                x_values = np.arange(len(first_system_data.get('values', first_system_data.get('mean', []))))
        else:
            # For RMSF: keep as strings (matching reference implementation)
            # Reference uses: x=list(map(str, aa_num)) for RMSF
            if metric.name == 'rmsf' and self._has_values(x_values):
                # Ensure x_values are list (not numpy array)
                if isinstance(x_values, np.ndarray):
                    x_values = x_values.tolist()
                # Keep as strings for RMSF (matching reference)
                if self._has_values(x_values) and not isinstance(x_values[0], str):
                    x_values = [str(x) for x in x_values]
        
        # Plot each system
        for system_name, system_data in data.items():
            # Determine color (matching old code: [175,0,0] for HOLO, [0,175,0] for APO)
            color = self._get_system_color(system_name)
            
            # Format legend label based on metric
            display_name = self._format_legend_name(system_name, metric.name)
            
            # Opacity: 0.5 for H-bonds, 1.0 for others (matching old code)
            opacity = 0.5 if 'Hydrogen' in metric.title or 'hbonds' in metric.name else 1.0
            
            if system_data['mode'] == 'single':
                # Single trajectory: just line
                # For RMSF, ensure x_values match data length
                y_vals = system_data['values']
                if metric.name == 'rmsf' and self._has_values(x_values) and len(x_values) != len(y_vals):
                    # Use data length to determine x_values
                    if len(x_values) > len(y_vals):
                        x_plot = x_values[:len(y_vals)]
                    else:
                        # Generate x_values from data length if config x_values are shorter
                        x_plot = list(range(len(y_vals)))
                else:
                    x_plot = x_values
                
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=y_vals,
                    name=display_name,
                    mode='lines',
                    opacity=opacity,
                    line=dict(
                        color=f'rgb({color[0]},{color[1]},{color[2]})',
                        width=2
                    )
                ))
            else:
                # Replicate: mean + error bands
                # Verify data structure
                if system_data['mode'] != 'replicate':
                    logger.warning(f"Expected replicate mode for {system_name}/{metric.name}, got {system_data['mode']}")
                
                # Ensure x_values and y_values have same length
                mean_vals = system_data['mean']
                upper_vals = system_data['upper']
                lower_vals = system_data['lower']
                
                # Debug logging: log initial state
                logger.debug(f"RMSF Error Band Debug - {system_name}/{metric.name}:")
                logger.debug(f"  Mode: {system_data.get('mode', 'unknown')}")
                logger.debug(f"  Mean length: {len(mean_vals)}, type: {type(mean_vals)}")
                logger.debug(f"  Upper length: {len(upper_vals)}, type: {type(upper_vals)}")
                logger.debug(f"  Lower length: {len(lower_vals)}, type: {type(lower_vals)}")
                logger.debug(f"  X_values length: {len(x_values) if x_values is not None else 0}, type: {type(x_values)}")
                
                # Verify upper and lower bounds exist
                if 'upper' not in system_data or 'lower' not in system_data:
                    logger.error(f"Missing upper/lower bounds for {system_name}/{metric.name}")
                if len(upper_vals) == 0 or len(lower_vals) == 0:
                    logger.warning(f"Empty upper/lower bounds for {system_name}/{metric.name}: upper={len(upper_vals)}, lower={len(lower_vals)}")
                
                # For RMSF, ensure x_values match data length
                if metric.name == 'rmsf' and self._has_values(x_values):
                    data_len = len(mean_vals)
                    if len(x_values) != data_len:
                        if len(x_values) > data_len:
                            # Truncate x_values to match data length
                            x_plot = list(x_values)[:data_len]
                        else:
                            # Generate x_values from data length if config x_values are shorter
                            x_plot = [str(i) for i in range(data_len)]
                    else:
                        x_plot = list(x_values) if not isinstance(x_values, np.ndarray) else x_values.tolist()
                    
                    # Ensure all arrays have same length for RMSF
                    min_len = min(len(x_plot), len(mean_vals), len(upper_vals), len(lower_vals))
                    x_plot = x_plot[:min_len]
                    mean_plot = mean_vals[:min_len]
                    upper_plot = upper_vals[:min_len]
                    lower_plot = lower_vals[:min_len]
                    
                else:
                    # For non-RMSF, use standard truncation
                    min_len = min(len(x_values), len(mean_vals), len(upper_vals), len(lower_vals))
                    x_plot = x_values[:min_len] if isinstance(x_values, np.ndarray) else list(x_values)[:min_len]
                    mean_plot = mean_vals[:min_len]
                    upper_plot = upper_vals[:min_len]
                    lower_plot = lower_vals[:min_len]
                
                logger.debug(f"  After truncation - x_plot: {len(x_plot)}, mean: {len(mean_plot)}, upper: {len(upper_plot)}, lower: {len(lower_plot)}")
                
                fig.add_trace(go.Scatter(
                    x=x_plot,
                    y=mean_plot,
                    name=display_name,
                    mode='lines',
                    opacity=opacity,
                    line=dict(
                        color=f'rgb({color[0]},{color[1]},{color[2]})',
                        width=2
                    )
                ))
                
                # Add error band - match reference implementation pattern exactly
                # Use same x-values as main line (matching reference: list(x)+list(x)[::-1])
                x_list = list(x_plot)
                upper_list = list(upper_plot)
                lower_list = list(lower_plot)
                
                logger.debug(f"  After conversion - x_list type: {type(x_list)}, length: {len(x_list)}")
                logger.debug(f"  upper_list type: {type(upper_list)}, length: {len(upper_list)}")
                logger.debug(f"  lower_list type: {type(lower_list)}, length: {len(lower_list)}")
                
                # Create error band polygon: forward with upper, backward with lower
                # Matching reference: list(x)+list(x)[::-1] and list(data[1])+list(data[2])[::-1]
                error_x = x_list + x_list[::-1]
                error_y = upper_list + lower_list[::-1]
                
                logger.debug(f"  Error band polygon - error_x length: {len(error_x)}, error_y length: {len(error_y)}")
                
                # Only add error band if we have valid data
                if len(error_x) > 0 and len(error_y) > 0 and len(error_x) == len(error_y):
                    logger.debug(f"  Adding error band trace for {system_name}/{metric.name}")
                    # Match reference implementation exactly: fill='toself', opacity=0.35, no fillcolor
                    # Reference uses: fill='toself', opacity=0.35, line=dict(color=f'rgba(...,0)')
                    fig.add_trace(go.Scatter(
                        x=error_x,
                        y=error_y,
                        fill='toself',
                        opacity=0.35,
                        line=dict(color=f'rgba({color[0]},{color[1]},{color[2]},0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                else:
                    logger.warning(f"  Skipping error band for {system_name}/{metric.name} - invalid data: error_x={len(error_x)}, error_y={len(error_y)}")
        
        # Update layout
        self._update_layout_simple(fig, metric, x_values)
        
        return fig
    
    # ==================== STYLE 2: ENHANCED (Mean Lines + Annotations) ====================
    
    def _plot_enhanced(self, data: Dict, metric: AnalysisMetric,
                       x_values: Optional[List] = None) -> go.Figure:
        """
        Enhanced plots with horizontal mean lines and statistics.
        From md_analysis_base.py create_enhanced_plot().
        """
        fig = self._plot_simple(data, metric, x_values)
        
        # Add horizontal mean lines for each system
        colors = [self._get_system_color(name) for name in data.keys()]
        
        for idx, (system_name, system_data) in enumerate(data.items()):
            color = colors[idx]
            display_name = self._format_legend_name(system_name, metric.name)
            
            # Calculate mean
            if system_data['mode'] == 'single':
                mean_val = np.mean(system_data['values'])
            else:
                mean_val = np.mean(system_data['mean'])
            
            # Add horizontal line
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color=f'rgba({color[0]},{color[1]},{color[2]},0.7)',
                opacity=0.7,
                annotation_text=f"{display_name} mean: {mean_val:.3f}",
                annotation_position="top right",
                annotation_font_size=10
            )
        
        return fig
    
    # ==================== STYLE 3: PUBLICATION (High Quality) ====================
    
    def _plot_publication(self, data: Dict, metric: AnalysisMetric,
                          x_values: Optional[List] = None) -> go.Figure:
        """
        Publication-quality plots with professional styling.
        Based on MMPBSA_per_res.py styling.
        """
        fig = self._plot_simple(data, metric, x_values)
        
        # Apply publication styling with grey background (matching ggplot2)
        fig.update_layout(
            font=dict(
                family="Times New Roman, serif" if 'Times' in self.config.plot_config.font_family else self.config.plot_config.font_family,
                size=self.config.plot_config.font_size,
                color='black'
            ),
            title_font=dict(
                size=self.config.plot_config.font_size + 4,
                color='black'
            ),
            paper_bgcolor='white',
            plot_bgcolor='#E5ECF6',  # Grey background (ggplot2 style)
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        )
        
        # Clean axis styling
        fig.update_xaxes(
            showgrid=True,
            gridcolor='white',  # White grid lines on grey background
            showline=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor='white',  # White grid lines on grey background
            showline=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True,
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.2)'
        )
        
        return fig
    
    # ==================== STYLE 4: OVERVIEW (Multi-Panel) ====================
    
    def _plot_overview(self, all_data: Dict) -> Optional[go.Figure]:
        """
        Create 2x2 multi-panel overview plot.
        From holo_md_analysis.py create_overview_plot().
        
        Shows key metrics: RMSD, COM-COM, H-bonds, Rg
        """
        # Select key metrics for overview
        key_metrics = ['rmsd_prot', 'comcom', 'hbonds', 'rog']
        
        # Find available metrics
        available_metrics = []
        metric_objects = {}
        
        # Get first system's data to find available metrics
        first_system = next(iter(all_data.values()))
        
        for metric_name in key_metrics:
            if metric_name in first_system:
                available_metrics.append(metric_name)
                # Find metric object
                for m in self.config.metrics:
                    if m.name == metric_name:
                        metric_objects[metric_name] = m
                        break
        
        if len(available_metrics) < 2:
            logger.warning("Not enough metrics for overview plot")
            return None
        
        # Create subplots
        n_plots = min(len(available_metrics), 4)
        rows = 2
        cols = 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[metric_objects[m].title for m in available_metrics[:4]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # Add traces for each metric
        for i, metric_name in enumerate(available_metrics[:4]):
            if i >= 4:
                break
            
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            metric = metric_objects[metric_name]
            
            # Plot each system
            for system_name, system_data in all_data.items():
                if metric_name not in system_data:
                    continue
                
                data = system_data[metric_name]
                color = self._get_system_color(system_name)
                display_name = self._format_legend_name(system_name, metric_name)
                
                # Get y-values
                if data['mode'] == 'single':
                    y_values = data['values']
                else:
                    y_values = data['mean']
                
                # Generate x-values
                x_values = data.get('time', np.arange(len(y_values)))
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        name=display_name,
                        line=dict(color=f'rgb({color[0]},{color[1]},{color[2]})', width=2),
                        showlegend=(i == 0)  # Only show legend once
                    ),
                    row=row, col=col
                )
            
            # Update axes
            fig.update_xaxes(title_text=self._time_axis_label() if row == rows else "", row=row, col=col)
            fig.update_yaxes(title_text=metric.ylabel, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f"{self.config.protein_name} - MD Analysis Overview",
            height=800,
            template=self.config.plot_config.template,
            font=dict(size=self.config.plot_config.font_size - 2)
        )
        
        return fig
    
    # ==================== STYLE 5: COMPARATIVE (Statistical Comparison) ====================
    
    def _plot_comparative(self, data: Dict, metric: AnalysisMetric,
                          x_values: Optional[List] = None) -> go.Figure:
        """
        Comparative plots with statistical comparison boxes.
        Shows mean, std, and if possible, statistical significance.
        """
        fig = self._plot_simple(data, metric, x_values)
        
        # Add statistics annotation box
        stats_text = self._generate_statistics_text(data)
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            bordercolor='black',
            borderwidth=1,
            borderpad=10,
            bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(size=10, family='monospace')
        )
        
        return fig
    
    # ==================== HELPER METHODS ====================
    
    def _get_system_color(self, system_name: str) -> Tuple[int, int, int]:
        """
        Get RGB color for a system (matching old code).
        
        Old code uses: [[175,0,0], [0,175,0]]
        - HOLO: [175, 0, 0] (red)
        - APO: [0, 175, 0] (green)
        
        Args:
            system_name: System name
            
        Returns:
            RGB tuple
        """
        if system_name in self.system_color_map:
            return self.system_color_map[system_name]
        # Check if APO
        if 'APO' in system_name.upper():
            # Green: [0, 175, 0] (matching old code)
            return self.config.plot_config.colors.get('apo', (0, 175, 0))
        else:
            # Red: [175, 0, 0] (matching old code)
            return self.config.plot_config.colors.get('holo', (175, 0, 0))
    
    def _format_legend_name(self, system_name: str, metric_name: str) -> str:
        """
        Format system name for legend display based on metric type.
        
        Rules:
        - COM-COM: ligand name only (ligand-only figure)
        - Ligand RMSD: ligand name only (ligand-only figure)
        - H-bonds: Protein_name_Ligand_name_Complex and Apo
        - RMSD Complex: Protein_name_Ligand_name_Complex
        - RMSD Protein: Protein_name Backbone (for both HOLO and APO)
        - RMSF: Protein_name_Ligand_name_Complex
        - Rg: Protein_name_Ligand_name_Complex
        - SASA: Protein_name_Ligand_name_Complex
        
        Args:
            system_name: System name from config
            metric_name: Name of the metric being plotted
            
        Returns:
            Formatted display name
        """
        if 'APO' in system_name.upper():
            # APO systems
            if metric_name == 'rmsd_prot':
                return f"{self.config.protein_name}_Backbone_Apo"
            else:
                return 'Apo'
        else:
            # HOLO systems
            ligand_name = system_name
            
            if metric_name in ['comcom', 'rmsd_lig']:
                # Ligand-only figures: just ligand name
                return ligand_name
            elif metric_name == 'rmsd_prot':
                # Protein RMSD: Protein_name_Backbone_Holo (with distinction)
                return f"{self.config.protein_name}_Backbone_Holo"
            else:
                # All other metrics: Protein_name_Ligand_name_Complex
                return f"{self.config.protein_name}_{ligand_name}_Complex"
    
    def _update_layout_simple(self, fig: go.Figure, metric: AnalysisMetric, 
                              x_values: List):
        """Update layout for simple style plots"""
        # Determine x-axis title
        if metric.name == 'rmsf':
            xaxis_title = "Amino Acid Number"
        else:
            xaxis_title = self._time_axis_label()
        
        title_text = metric.title
        if metric.name == 'comcom' and metric.title == 'COM-COM Distance':
            title_text = (
                f"Distance from the center of mass of {self.config.protein_name} "
                "to the center of mass of the ligands"
            )

        fig.update_layout(
            title=dict(
                text=title_text,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis_title=xaxis_title,
            yaxis_title=metric.ylabel,
            legend=dict(
                y=-0.4,  # Matching old code: y=-0.4
                xanchor="center",
                x=0.5,
                orientation='h'
            ),
            font_family=self.config.plot_config.font_family,
            font_color="black",
            font_size=self.config.plot_config.font_size,
            title_font_family=self.config.plot_config.font_family,  # Matching old code
            title_font_color="black",  # Matching old code
            template=self.config.plot_config.template
        )
        
        # Apply metric-specific y-axis limits
        self._apply_axis_limits(fig, metric, list(fig.data))
    
    def _apply_axis_limits(self, fig: go.Figure, metric: AnalysisMetric, traces: List):
        """
        Apply metric-specific y-axis limits.
        From holo_apo_vis.py modify_limits().
        """
        if not traces:
            return
        
        # Extract all y-values
        all_y_values = []
        for trace in traces:
            if hasattr(trace, 'y') and trace.y is not None:
                all_y_values.extend([y for y in trace.y if not np.isnan(y)])
        
        if not all_y_values:
            return
        
        metric_name = metric.name
        
        try:
            if metric_name == 'comcom':
                upper_limit = max(all_y_values) + 1
                fig.update_yaxes(range=[0, upper_limit])
            
            elif metric_name == 'hbonds':
                upper_limit = max(all_y_values) + 2
                fig.update_yaxes(range=[0, upper_limit])
            
            elif metric_name == 'rog':
                lower_limit = min(all_y_values) - 0.5
                upper_limit = max(all_y_values) + 0.5
                fig.update_yaxes(range=[lower_limit, upper_limit])
            
            elif metric_name == 'sasa':
                lower_limit = min(all_y_values) - 1000
                upper_limit = max(all_y_values) + 1000
                fig.update_yaxes(range=[lower_limit, upper_limit])
            
            elif metric_name in ['rmsd_prot', 'rmsd_lig', 'rmsd_complex', 'rmsd_combined']:
                upper_limit = max(all_y_values) + 1
                fig.update_yaxes(range=[0, upper_limit])
            
            elif metric_name == 'rmsf':
                upper_limit = max(all_y_values) + 1
                fig.update_yaxes(range=[0, upper_limit])
        
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not set axis limits for {metric_name}: {e}")
    
    def _generate_statistics_text(self, data: Dict) -> str:
        """
        Generate statistics text box for comparative plots.
        
        Args:
            data: Dict of {system_name: processed_data_dict}
            
        Returns:
            Formatted statistics string
        """
        lines = ["<b>Statistics</b>", ""]
        
        for system_name, system_data in data.items():
            stats = system_data.get('stats', {})
            
            lines.append(f"<b>{system_name}:</b>")
            lines.append(f"  Mean: {stats.get('mean', 0):.3f}")
            lines.append(f"  Std:  {stats.get('std', 0):.3f}")
            lines.append(f"  Min:  {stats.get('min', 0):.3f}")
            lines.append(f"  Max:  {stats.get('max', 0):.3f}")
            lines.append("")
        
        return "<br>".join(lines)
    
    # ==================== SAVE PLOTS ====================
    
    def save_plot(self, fig: go.Figure, output_path: Path, base_name: str, fallback: Optional[Dict] = None):
        """
        Save plot in multiple formats.
        
        Args:
            fig: Plotly Figure object
            output_path: Output directory
            base_name: Base filename (without extension)
            fallback: Optional fallback payload for matplotlib SVG export
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.config.plot_config.save_formats:
            filepath = output_path / f"{base_name}.{fmt}"
            
            try:
                if fmt == 'html':
                    fig.write_html(filepath)
                elif fmt in ['svg', 'png', 'pdf']:
                    fig.write_image(
                        filepath, 
                        width=self.config.plot_config.width, 
                        height=self.config.plot_config.height,
                        scale=self.config.plot_config.scale
                    )
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                if fmt == 'svg' and fallback:
                    if self._save_svg_fallback(filepath, fallback):
                        logger.info(f"Saved SVG via matplotlib fallback: {filepath}")
                        continue
                logger.error(f"Could not save {filepath}: {e}")
    
    def create_all_plots(self, all_data: Dict) -> List[str]:
        """
        Create all plots for all metrics.
        
        Args:
            all_data: Complete data structure from processor
            
        Returns:
            List of created plot names
        """
        plots_created = []
        output_dir = self.config.output_dir / "plots"
        
        # Get all metrics
        metrics_to_plot = set()
        for system_data in all_data.values():
            metrics_to_plot.update(system_data.keys())
        
        # Create plot for each metric
        for metric_name in metrics_to_plot:
            # Find metric object
            metric = None
            for m in self.config.metrics:
                if m.name == metric_name:
                    metric = m
                    break
            
            if not metric:
                logger.warning(f"Metric object not found for: {metric_name}")
                continue
            
            # Collect data for this metric from all systems
            metric_data = {}
            for system_name, system_data in all_data.items():
                if metric_name in system_data:
                    metric_data[system_name] = system_data[metric_name]
            
            if not metric_data:
                continue
            
            # Prepare x-values for RMSF
            x_values = None
            if metric.name == 'rmsf':
                if self.config.amino_acids:
                    x_values = [str(aa) for aa in self.config.amino_acids]
                else:
                    try:
                        from .data_processor import MDDataProcessor
                        x_values = MDDataProcessor(self.config).prepare_amino_acid_axis()
                    except Exception:
                        x_values = None
            
            # Create plot
            fig = self.create_plot(metric_data, metric, x_values)
            
            # Save plot
            self.save_plot(
                fig,
                output_dir,
                metric_name,
                fallback={
                    'kind': 'metric',
                    'metric': metric,
                    'metric_data': metric_data,
                    'x_values': x_values
                }
            )
            plots_created.append(metric_name)
        
        # Create overview plot if not in overview style mode
        if self.style != 'overview':
            overview_fig = self.create_overview_plot(all_data)
            if overview_fig:
                self.save_plot(
                    overview_fig,
                    output_dir,
                    'overview',
                    fallback={'kind': 'overview', 'all_data': all_data}
                )
                plots_created.append('overview')

        # Create combined RMSD plot (APO protein vs HOLO complex)
        combined_fig = self.create_combined_rmsd_plot(all_data)
        if combined_fig:
            self.save_plot(
                combined_fig,
                output_dir,
                'rmsd_combined',
                fallback={'kind': 'combined', 'all_data': all_data}
            )
            plots_created.append('rmsd_combined')
        
        return plots_created

    def _save_svg_fallback(self, filepath: Path, fallback: Dict) -> bool:
        kind = fallback.get('kind')
        try:
            if kind == 'metric':
                return self._save_metric_svg(
                    filepath,
                    fallback.get('metric'),
                    fallback.get('metric_data', {}),
                    fallback.get('x_values')
                )
            if kind == 'overview':
                return self._save_overview_svg(filepath, fallback.get('all_data', {}))
            if kind == 'combined':
                return self._save_combined_svg(filepath, fallback.get('all_data', {}))
        except Exception as exc:
            logger.error(f"Matplotlib fallback failed for {filepath}: {exc}")
        return False

    def _save_metric_svg(self, filepath: Path, metric: AnalysisMetric,
                         metric_data: Dict, x_values: Optional[List]) -> bool:
        if not metric or not metric_data:
            return False

        _, plt = init_matplotlib()
        fig, ax = plt.subplots(
            figsize=(self.config.plot_config.width / 100, self.config.plot_config.height / 100),
            dpi=100 * self.config.plot_config.scale
        )

        plt.rcParams['font.family'] = self.config.plot_config.font_family
        plt.rcParams['font.size'] = self.config.plot_config.font_size

        all_y_values = []
        label_map = None

        for system_name, system_data in metric_data.items():
            color = rgb_to_mpl(self._get_system_color(system_name))
            display_name = self._format_legend_name(system_name, metric.name)
            opacity = 0.5 if 'Hydrogen' in metric.title or 'hbonds' in metric.name else 1.0

            if system_data['mode'] == 'single':
                y_vals = list(system_data.get('values', []))
                x_plot, label_map = self._get_x_values(metric, system_data, y_vals, x_values)
                ax.plot(x_plot, y_vals[:len(x_plot)], color=color, linewidth=2, alpha=opacity, label=display_name)
                all_y_values.extend(y_vals)
            else:
                mean_vals = list(system_data.get('mean', []))
                upper_vals = list(system_data.get('upper', []))
                lower_vals = list(system_data.get('lower', []))
                x_plot, label_map = self._get_x_values(metric, system_data, mean_vals, x_values)
                min_len = min(len(x_plot), len(mean_vals), len(upper_vals), len(lower_vals))
                x_plot = x_plot[:min_len]
                mean_vals = mean_vals[:min_len]
                upper_vals = upper_vals[:min_len]
                lower_vals = lower_vals[:min_len]
                ax.plot(x_plot, mean_vals, color=color, linewidth=2, alpha=opacity, label=display_name)
                ax.fill_between(x_plot, lower_vals, upper_vals, color=color, alpha=0.35)
                all_y_values.extend(mean_vals)
                all_y_values.extend(upper_vals)
                all_y_values.extend(lower_vals)

        ax.set_title(metric.title)
        ax.set_xlabel("Amino Acid Number" if metric.name == 'rmsf' else self._time_axis_label())
        ax.set_ylabel(metric.ylabel)
        ax.grid(True, alpha=0.2)

        y_limits = self._compute_axis_limits(metric.name, all_y_values)
        if y_limits:
            ax.set_ylim(y_limits)

        if label_map:
            step = max(1, len(label_map) // 30)
            ax.set_xticks(list(range(0, len(label_map), step)))
            ax.set_xticklabels([label_map[i] for i in range(0, len(label_map), step)], rotation=90)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)
        return True

    def _save_overview_svg(self, filepath: Path, all_data: Dict) -> bool:
        if not all_data:
            return False

        key_metrics = ['rmsd_prot', 'comcom', 'hbonds', 'rog']
        available_metrics = []
        metric_objects = {}
        first_system = next(iter(all_data.values()))

        for metric_name in key_metrics:
            if metric_name in first_system:
                available_metrics.append(metric_name)
                for m in self.config.metrics:
                    if m.name == metric_name:
                        metric_objects[metric_name] = m
                        break

        if len(available_metrics) < 2:
            return False

        _, plt = init_matplotlib()
        fig, axes = plt.subplots(
            2, 2,
            figsize=(self.config.plot_config.width / 100, self.config.plot_config.height / 100),
            dpi=100 * self.config.plot_config.scale
        )
        plt.rcParams['font.family'] = self.config.plot_config.font_family
        plt.rcParams['font.size'] = self.config.plot_config.font_size - 2

        handles = []
        labels = []

        for idx, metric_name in enumerate(available_metrics[:4]):
            row = idx // 2
            col = idx % 2
            ax = axes[row][col]
            metric = metric_objects[metric_name]

            all_y_values = []
            for system_name, system_data in all_data.items():
                if metric_name not in system_data:
                    continue
                data = system_data[metric_name]
                color = rgb_to_mpl(self._get_system_color(system_name))
                display_name = self._format_legend_name(system_name, metric_name)
                y_values = data['values'] if data['mode'] == 'single' else data['mean']
                x_values = data.get('time', list(range(len(y_values))))
                ax.plot(x_values, y_values, color=color, linewidth=2, label=display_name)
                all_y_values.extend(y_values)

            ax.set_title(metric.title)
            ax.set_xlabel(self._time_axis_label() if row == 1 else "")
            ax.set_ylabel(metric.ylabel)
            ax.grid(True, alpha=0.2)

            y_limits = self._compute_axis_limits(metric.name, all_y_values)
            if y_limits:
                ax.set_ylim(y_limits)

            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()

        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)

        fig.suptitle(f"{self.config.protein_name} - MD Analysis Overview")
        fig.subplots_adjust(bottom=0.2, top=0.9, hspace=0.4, wspace=0.3)
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)
        return True

    def _save_combined_svg(self, filepath: Path, all_data: Dict) -> bool:
        if not all_data:
            return False

        apo_system = next((s for s in self.config.systems if s.is_apo), None)
        holo_system = next((s for s in self.config.systems if not s.is_apo), None)
        if not apo_system or not holo_system:
            return False

        apo_data = all_data.get(apo_system.name, {}).get('rmsd_prot')
        holo_data = all_data.get(holo_system.name, {}).get('rmsd_complex')
        if apo_data is None or holo_data is None:
            return False

        _, plt = init_matplotlib()
        fig, ax = plt.subplots(
            figsize=(self.config.plot_config.width / 100, self.config.plot_config.height / 100),
            dpi=100 * self.config.plot_config.scale
        )
        plt.rcParams['font.family'] = self.config.plot_config.font_family
        plt.rcParams['font.size'] = self.config.plot_config.font_size

        def add_series(label: str, data: Dict, color: Tuple[int, int, int]):
            color = rgb_to_mpl(color)
            if data.get('mode') == 'single':
                y_values = list(data.get('values', []))
                x_values = data.get('time', list(range(len(y_values))))
            else:
                y_values = list(data.get('mean', []))
                x_values = data.get('time', list(range(len(y_values))))
                upper = data.get('upper')
                lower = data.get('lower')
                if upper is not None and lower is not None:
                    min_len = min(len(x_values), len(upper), len(lower))
                    ax.fill_between(
                        x_values[:min_len],
                        list(lower)[:min_len],
                        list(upper)[:min_len],
                        color=color,
                        alpha=0.25
                    )

            ax.plot(x_values, y_values, color=color, linewidth=2, label=label)
            return y_values

        all_y = []
        all_y.extend(add_series("APO Protein", apo_data, self._get_system_color(apo_system.name)))
        all_y.extend(add_series("Complex", holo_data, self._get_system_color(holo_system.name)))

        ax.set_title("Combined Protein RMSD")
        ax.set_xlabel(self._time_axis_label())
        ax.set_ylabel("RMSD (Å)")
        ax.grid(True, alpha=0.2)

        y_limits = self._compute_axis_limits('rmsd_combined', all_y)
        if y_limits:
            ax.set_ylim(y_limits)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)
        return True

    def _get_x_values(self, metric: AnalysisMetric, system_data: Dict,
                      y_vals: List, x_values: Optional[List]):
        if self._has_values(x_values):
            x_vals = list(x_values)
        else:
            x_vals = system_data.get('time')
            if x_vals is None or len(x_vals) == 0:
                x_vals = list(range(len(y_vals)))
        if isinstance(x_vals, np.ndarray):
            x_vals = x_vals.tolist()

        x_vals = list(x_vals)[:len(y_vals)]

        if metric.name == 'rmsf':
            try:
                x_numeric = [int(float(x)) for x in x_vals]
                return x_numeric, None
            except Exception:
                return list(range(len(y_vals))), [str(x) for x in x_vals]

        return x_vals, None

    def _compute_axis_limits(self, metric_name: str, all_y_values: List):
        cleaned = [y for y in all_y_values if y is not None and not np.isnan(y)]
        if not cleaned:
            return None

        try:
            if metric_name == 'comcom':
                return (0, max(cleaned) + 1)
            if metric_name == 'hbonds':
                return (0, max(cleaned) + 2)
            if metric_name == 'rog':
                return (min(cleaned) - 0.5, max(cleaned) + 0.5)
            if metric_name == 'sasa':
                return (min(cleaned) - 1000, max(cleaned) + 1000)
            if metric_name in ['rmsd_prot', 'rmsd_lig', 'rmsd_complex', 'rmsd_combined', 'rmsf']:
                return (0, max(cleaned) + 1)
        except (ValueError, TypeError):
            return None

        return None

    def create_combined_rmsd_plot(self, all_data: Dict) -> Optional[go.Figure]:
        """
        Create combined RMSD plot (APO protein vs HOLO complex).

        Uses the standard combined RMSD output name: rmsd_combined.
        """
        apo_system = next((s for s in self.config.systems if s.is_apo), None)
        holo_system = next((s for s in self.config.systems if not s.is_apo), None)

        if not apo_system or not holo_system:
            return None

        apo_data = all_data.get(apo_system.name, {}).get('rmsd_prot')
        holo_data = all_data.get(holo_system.name, {}).get('rmsd_complex')

        if apo_data is None or holo_data is None:
            return None

        fig = go.Figure()

        def add_series(label: str, data: Dict, color: Tuple[int, int, int]):
            if data.get('mode') == 'single':
                y_values = data.get('values', [])
                x_values = data.get('time', np.arange(len(y_values)))
            else:
                y_values = data.get('mean', [])
                x_values = data.get('time', np.arange(len(y_values)))

                upper = data.get('upper')
                lower = data.get('lower')
                if upper is not None and lower is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=upper,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=lower,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({color[0]}, {color[1]}, {color[2]}, 0.25)',
                            showlegend=False
                        )
                    )

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=label,
                    line=dict(color=f'rgb({color[0]}, {color[1]}, {color[2]})', width=2)
                )
            )

        add_series("APO Protein", apo_data, self._get_system_color(apo_system.name))
        add_series("Complex", holo_data, self._get_system_color(holo_system.name))

        metric = AnalysisMetric(
            name='rmsd_combined',
            file_pattern='',
            title='Combined Protein RMSD',
            ylabel='RMSD (Å)',
            data_format='single_column'
        )
        self._update_layout_simple(fig, metric, [])

        return fig
