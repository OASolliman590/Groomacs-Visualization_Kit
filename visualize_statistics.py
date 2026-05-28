#!/usr/bin/env python3
"""
Visualize summary statistics from MD analysis.

Creates a single summary figure with a separate, independently-scaled,
box plot for each analysis type, without detailed statistics on hover.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import logging
from pathlib import Path
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_summary_statistics(csv_file, output_dir):
    """Reads a CSV file and creates a merged summary plot without hover-disclosed stats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully read {csv_file}")
    except FileNotFoundError:
        logger.error(f"Error: The file {csv_file} was not found.")
        return

    analysis_types = df['Analysis_Type'].unique()
    n_types = len(analysis_types)
    cols = 3
    rows = math.ceil(n_types / cols)

    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=analysis_types,
        shared_yaxes=False,
        vertical_spacing=0.25
    )

    for i, analysis_type in enumerate(analysis_types):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        analysis_df = df[df['Analysis_Type'] == analysis_type]
        
        for _, series in analysis_df.iterrows():
            system_name = series.get('System', f"{series.get('Ligand', 'Apo')}")
            
            fig.add_trace(go.Box(
                name=system_name,
                q1=[series['Q1']],
                median=[series['Median']],
                q3=[series['Q3']],
                lowerfence=[series['Min']],
                upperfence=[series['Max']],
                mean=[series['Mean']],
                sd=[series['Std']],
                boxpoints=False,
                showlegend=(i==0)
            ), row=row, col=col)

    fig.update_layout(
        title_text='<b>MD Analysis Statistics Summary</b>',
        height=400 * rows,
        width=1200,
        template='plotly_dark',
        legend_title='<b>System</b>',
        title_font_size=22,
        hovermode='closest'
    )

    output_file = output_path / "statistics_summary_merged_no_hover.html"
    fig.write_html(output_file)
    logger.info(f"Merged summary plot without hover details saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize MD Analysis Summary Statistics.")
    parser.add_argument(
        "--csv_file",
        default="enhanced_analysis/summary_statistics.csv",
        help="Path to the summary_statistics.csv file."
    )
    parser.add_argument(
        "--output_dir",
        default="enhanced_analysis/plots",
        help="Directory to save the visualization."
    )
    args = parser.parse_args()
    visualize_summary_statistics(args.csv_file, args.output_dir)

if __name__ == "__main__":
    main()