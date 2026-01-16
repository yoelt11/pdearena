# Reproducibility Information

This directory contains the minimal set of files needed to reproduce the experimental results reported in this repository.

## Experimental Setup

### Seeds Used
- **Seed 0**: Initial baseline run
- **Seed 1**: First replication
- **Seed 2**: Second replication

### Data Splits Configuration
All experiments used the following balanced split configuration:
- **Training samples**: 10
- **Interpolation test samples**: 20 (within parameter convex hull)
- **Extrapolation test samples**: 20 (outside parameter convex hull)
- **Total test samples**: 40
- **Balance strategy**: Random (deterministic with fixed seed)
- **Balance enabled**: True

### Split Indices
The exact train/interp/extrap indices for each seed are provided in:
- `split_indices/seed_0_split_indices.yaml`
- `split_indices/seed_1_split_indices.yaml`
- `split_indices/seed_2_split_indices.yaml`

These files contain the exact sample indices used for each split, ensuring complete reproducibility.

## Results

### Aggregated Summary Tables
The file `aggregated_summary_tables.txt` contains the final aggregated results across all three seeds (0, 1, 2), showing:
- **Interpolation results**: Mean ± std relative L2 error for zero-shot evaluation
- **Extrapolation results**: Mean ± std relative L2 error for zero-shot evaluation
- **Inference time**: Mean ± std milliseconds per sample
- **Fine-tuning results**: Currently N/A (not yet implemented)

The aggregation was performed using:
```bash
uv run python scripts/aggregate_seed_results.py --seeds 0,1,2
```

### Example Metrics Files
Example `metrics.json` files from each seed are provided in `example_metrics/` to show the structure of the output. These contain:
- Per-solution relative L2 errors
- Aggregated statistics (mean, std)
- Inference timing information
- Model and data configuration metadata

## Reproducing the Results

### Step 1: Run All Experiments
```bash
# Run for all seeds
uv run python scripts/run_all_eff_experiments.py --seeds 0,1,2

# Or run individually
uv run python scripts/run_all_eff_experiments.py --seeds 0
uv run python scripts/run_all_eff_experiments.py --seeds 1
uv run python scripts/run_all_eff_experiments.py --seeds 2
```

This script will:
1. Automatically find all `eff_*.yaml` config files in `configs/`
2. Run training (fit) for each config and seed
3. Run testing for each trained model
4. Save results to `outputs/seed_{seed}/eff_{equation}_{model_type}/`
5. Generate `metrics.json` and `split_indices.yaml` for each run

### Step 2: Aggregate Results
```bash
uv run python scripts/aggregate_seed_results.py --seeds 0,1,2
```

This generates `outputs/aggregated_summary_tables.txt` with mean ± std across all seeds.

### Expected Output Structure
```
outputs/
├── seed_0/
│   ├── eff_allen_cahn_parametric/
│   │   ├── metrics.json
│   │   ├── split_indices.yaml
│   │   └── lightning_logs/...
│   ├── eff_allen_cahn_fno/
│   │   └── ...
│   └── ... (8 total configs)
├── seed_1/
│   └── ... (same structure)
├── seed_2/
│   └── ... (same structure)
└── aggregated_summary_tables.txt
```

## Notes

- All experiments use deterministic data splits (same indices for same seed)
- The balanced split ensures equal-sized interpolation and extrapolation test sets
- Results are organized by seed to enable statistical aggregation
- Full metrics and checkpoints are saved in `outputs/` (not included in this repo for size)
