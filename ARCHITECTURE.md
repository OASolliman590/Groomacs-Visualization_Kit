# Architecture

The toolkit is organized as a package with a thin CLI layer and a YAML-driven pipeline runner.

## Layers

```text
CLI / wrapper scripts
        |
pipeline YAML loader
        |
PipelineConfig / PipelineStage
        |
stage analyzers
        |
processors + parsers + plotters
        |
CSV/HTML/SVG/PNG/Markdown outputs
```

## Stage Pattern

Most analysis modules follow the same structure:

- `config.py` - dataclasses for validated stage configuration.
- `processor.py` or `parser.py` - data discovery, loading, and cleaning.
- `plotter.py` - visualization/export helpers.
- `analyzer.py` - orchestration and returned result summary.

## Pipeline Runner

`gromacs_analysis.pipeline.runner.run_pipeline()` executes enabled stages in config order. A stage can be a single config object or a list of configs for batch modes.

The runner supports:

- stage selection through `--run`,
- stage skipping through `--skip`,
- `--continue-on-error`,
- lightweight cache signatures for stages that enable cache support,
- optional utility tasks.

## Configuration

Use `gromacs_pipeline_template.yaml` as the generic starting point. Project-specific config files should live outside the package or be passed in through:

```bash
CONFIG=/path/to/project_pipeline.yaml ./run_pipeline.sh
```

## Data Boundaries

Source code, templates, and docs stay in the repo. Raw trajectories and generated outputs stay outside the repo and are referenced by YAML paths.
