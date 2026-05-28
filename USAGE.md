# ClarityDynamics Usage

This guide explains practical use cases for the maintained package pipeline, from quick checks to full multi-ligand reporting.

## 1) Setup

```bash
cd streamlined_pipeline/gromacs_analysis_toolkit
pip install -e .
pip install -e ".[all]"
```

Create your config from either template:

```bash
cp gromacs_pipeline_template.yaml my_pipeline.yaml
# or
cp pipeline_guide_empty.yaml my_pipeline.yaml
```

Validate and run:

```bash
CONFIG=my_pipeline.yaml ./run_pipeline.sh --dry-run
CONFIG=my_pipeline.yaml ./run_pipeline.sh
```

## 2) Core Use Cases

### Use Case A: Preflight Scientific QC Before Any Analysis

Enable `qc` first in `run`.
Recommended when ingesting new trajectories or when replicate quality is uncertain.

What it checks:
- frame counts and atom counts
- timestep/time-axis detectability
- box/PBC metadata
- protein/ligand selection sanity
- minimum ligand-protein distance
- optional index/group consistency

### Use Case B: Single-System MD Stability and Dynamics

Enable `md`.
Typical outputs: RMSD, RMSF, Rg, SASA, H-bonds, COM distance, summary stats, and plots.

Key controls:
- `time_unit_input`, `time_unit_output`, optional `time_step_ps`
- `smoothing_window`
- `outlier_threshold`
- RMSF labeling via `amino_acids.sequence_start` or explicit labels

### Use Case C: Binding Energy Analysis with Replicate Integrity

Enable `mmpbsa`.

Key controls:
- `incomplete_replicates_policy`: `ignore|warn|error`
- `require_decomp_replicates`
- `compare_binding_key`

Recommended pattern:
- run `qc` and `md` first
- inspect MMPBSA QC metadata before ranking

### Use Case D: PCA and FEL for Conformational Landscape

Enable `pca`.
Optional: `run_gromacs.enabled: true` for automated `covar/anaeig` preparation.

Use for:
- scree and PC projection
- cosine content over windows
- FEL maps (`fel.enabled: true`)
- clustering in PCA space (`clustering.enabled: true`)

### Use Case E: Residue Distance Mapping

Enable `distance` for one ligand or `distance_batch` for many ligands.
Use `distance_compare` for cross-ligand heatmaps and deltas.

### Use Case F: Classical Interaction Fingerprints

Enable `interactions` or `interactions_batch`.
Use for basic interaction occupancy profiling independent of ProLIF chemistry typing.

### Use Case G: ProLIF Contact Chemistry and Visualization

Enable `prolif` or `prolif_batch`, then `prolif_compare` for cross-ligand analysis.

Current visualization defaults support:
- ligand network with full occurrence display (`lignetwork_display_all: true`)
- count-aware rendering (`lignetwork_count_aware: true`)
- barcode controls (`n_frame_ticks`, `residues_tick_location`, `xlabel`, `figsize`, `dpi`)
- residue-level barcode QC filtering (`only_interacting_residues`, `min_interaction_frames`, optional `max_residues`)

Expected barcode outputs:
- static SVG barcode
- interactive HTML barcode
- `barcode_qc_repN.csv` with per-residue interaction-frame counts

### Use Case H: TTClust Conformational Clustering

Enable `ttclust` or `ttclust_batch`.

Typical outputs:
- dendrogram
- distance matrix
- linear and histogram views
- representative cluster PDBs

Use `stride` and selection fields to control runtime and clustering scope.

### Use Case I: Cross-Ligand Scoring and Prioritization

Enable `ranking` after upstream stages complete.
Recommended inputs:
- MD compare stats
- MMPBSA QC-aware summaries
- distance compare summaries

## 3) Recommended Stage Recipes

### Conservative full scientific workflow

```yaml
run: ["qc", "md_batch", "md_compare", "mmpbsa", "pca_batch", "pca_compare", "prolif_batch", "prolif_compare", "ttclust_batch", "ranking"]
```

### Fast screening workflow

```yaml
run: ["qc", "md_batch", "mmpbsa", "prolif_batch", "ranking"]
```

### Visualization-focused workflow

```yaml
run: ["prolif_batch", "prolif_compare", "ttclust_batch", "pca_batch", "pca_compare"]
```

## 4) Utility Workflows (`utils` stage)

Available helper task patterns include:
- `examine_csv`
- `convert_dat`
- `renumber_gro`
- `modevectors`
- `generate_upstream_scripts`
- `duivy_inspect_guide`

`duivy_inspect_guide` is intended for raw `.xvg/.xpm` inspection guidance and not a replacement for maintained toolkit outputs.

## 5) Validation Gates

Run these before sharing results:

```bash
python -m compileall -q gromacs_analysis
pytest -q
python -m gromacs_analysis.cli.pipeline_cli --generate-template
CONFIG=my_pipeline.yaml ./run_pipeline.sh --dry-run
```

## 6) Data and Provenance Practices

- Keep raw trajectories and bulky generated outputs outside source control.
- Set `paths.output_root` to project storage.
- Keep project YAMLs free of personal absolute paths when sharing publicly.
- Store system-specific private paths in local, non-committed configs.
