# anchor-op tutorials

Task-oriented notebooks covering the anchor-op public API end-to-end. Each notebook is **self-contained** (uses synthetic data drawn in the notebook itself), so it runs anywhere without external downloads. If you have Replogle 2022 or Jost 2020 h5ads available, the `../reproduction/` directory shows how the paper's figures are generated on real data.

## Read in order

| Notebook | Covers | Runtime |
|---|---|---:|
| `01_quickstart.ipynb` | Minimal end-to-end pipeline on d=6 synthetic ground truth | ~1 min |
| `02_efficiency_estimators.ipynb` | The three κ estimators + `"auto"` router + `min_control_detection_rate` filter | ~2 min |
| `03_measure_operator_end_to_end.ipynb` | Full pipeline on synthetic Perturb-seq-like AnnData, including the `AnchorReport` fields and how to read them | ~2 min |
| `04_linearity_diagnostics_power_analysis.ipynb` | Both linearity diagnostics + how to interpret observed values against a matched-scale positive control (the paper's central methodology, §3.5–3.6) | ~5 min |
| `05_compare_benchmark_archetypes.ipynb` | `ao.compare`, `ao.analyses.benchmark_report`, `ao.fit_archetypes`, `ao.transfer_test` — the downstream API | ~2 min |
| `06_program_basis_and_analyses_report.ipynb` | `ao.fit_programs`, `ao.make_program_basis`, `ao.projection_helpers`, and the high-level `ao.analyses.*_report` shortcut wrappers | ~2 min |

## How to run

```bash
cd /path/to/anchor-op-source
PYTHONPATH=src jupyter lab tutorial/
```

Or run each notebook headless:

```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace tutorial/01_quickstart.ipynb
```

Or convert to a script and just run it:

```bash
PYTHONPATH=src jupyter nbconvert --to script tutorial/01_quickstart.ipynb --stdout | python3
```

## What this directory is NOT

- **Not the paper's figure reproduction**: use `../reproduction/` for that.
- **Not the tool's test suite**: use `pytest tests/` for that.
- **Not the API reference**: use docstrings via `help(ao.measure_operator)` etc.

## What it IS

A step-by-step tour of every public entry point in `anchorop`, written for someone who wants to apply the tool to their own Perturb-seq data. Each notebook can be read as prose or executed as code; expected outputs are shown inline.
