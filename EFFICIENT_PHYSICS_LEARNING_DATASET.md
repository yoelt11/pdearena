# Efficient Physics Learning Dataset Integration

This document describes the modifications made to this repository to test **U-Net** and **FNO (Fourier Neural Operator)** models on parametric PDE datasets from the [eff-physics-learn-dataset](https://github.com/yoelt11/eff-physics-learn-dataset) repository.

## Overview

The modifications enable training and evaluation of both U-Net and FNO architectures on parametric PDE datasets with standardized metrics export using [metrics-structures](https://github.com/yoelt11/metrics-structures). The framework supports:

- **2D datasets**: burgers, allen_cahn, convection, helmholtz2D
- **3D datasets**: helmholtz3D, flow_mixing (U-Net only)
- **Few-shot learning**: Training on small parametric splits (typically 10 samples)
- **Interpolation and extrapolation evaluation**: Separate test splits for within-range and out-of-range parameter values

## Quick Start: Run All Experiments

To run all experiments (training + testing) for all available configs with a single command:

### Single Seed (Default)

```bash
uv run python scripts/run_all_eff_experiments.py
# or explicitly:
uv run python scripts/run_all_eff_experiments.py --seeds 0
```

### Multiple Seeds (Recommended)

For robust, publication-ready results, run with multiple seeds and aggregate:

```bash
# Run experiments for seeds 1 and 2 (seed 0 already completed)
uv run python scripts/run_all_eff_experiments.py --seeds 1,2

# Or run all seeds from scratch
uv run python scripts/run_all_eff_experiments.py --seeds 0,1,2
```

**After running multiple seeds, aggregate the results:**

```bash
uv run python scripts/aggregate_seed_results.py --seeds 0,1,2
```

This generates aggregated summary tables with mean ± std across seeds.

### What the Scripts Do

**`run_all_eff_experiments.py`:**
1. **Finds all configs**: Automatically discovers all `eff_*.yaml` config files in `configs/`
2. **Runs for each seed**: For each specified seed, runs training then testing
3. **Organizes outputs**: Results are saved in `outputs/seed_{seed}/` directories
4. **Generates metrics**: Produces `metrics.json` and `split_indices.yaml` for each run

**`aggregate_seed_results.py`:**
1. **Reads all seed directories**: Loads metrics from `outputs/seed_{seed}/` for each seed
2. **Computes statistics**: Calculates mean ± std across seeds for each equation/model
3. **Generates tables**: Creates aggregated summary tables with robust statistics

**Output organization:**
- Per-seed results: `outputs/seed_{seed}/eff_{equation}_{model_type}/`
- Aggregated tables: `outputs/aggregated_summary_tables.txt`

The scripts automatically:
- Determine which training script to use (FNO vs U-Net) based on config filename
- Find the correct checkpoint path for testing
- Update config files with the correct seed and output directory
- Provide progress updates and error reporting

**Note**: Make sure all required datasets are downloaded before running (see [Downloading Datasets](#downloading-datasets) below).

## Dataset Repository

The datasets are provided by the [eff-physics-learn-dataset](https://github.com/yoelt11/eff-physics-learn-dataset) repository. Each dataset contains parametric PDE solutions where each solution depends on a set of parameters (e.g., diffusion coefficient, reaction rate, etc.).

## Downloading Datasets

The `eff-physics-learn-dataset` download utility expects a `dataset_links.toml` file. You can fetch it and download datasets as follows:

### Download a single dataset (e.g., burgers):

```bash
curl -L -o eff_dataset_links.toml https://raw.githubusercontent.com/yoelt11/eff-physics-learn-dataset/main/configs/datasets/dataset_links.toml

source .venv/bin/activate
EFF_PHYSICS_LEARN_DATASET_LINKS=eff_dataset_links.toml python - <<'PY'
from eff_physics_learn_dataset.download import load_dataset_links, download_dataset
from pathlib import Path
links = load_dataset_links()
download_dataset("burgers", links["burgers"], Path("datasets"))
PY
```

### Download all datasets:

```bash
source .venv/bin/activate
EFF_PHYSICS_LEARN_DATASET_LINKS=eff_dataset_links.toml python - <<'PY'
from eff_physics_learn_dataset.download import load_dataset_links, download_dataset
from pathlib import Path

outdir = Path("datasets")
links = load_dataset_links()
for name, fid in links.items():
    download_dataset(name, fid, outdir)
PY
```

## Training

### U-Net Model

Train a U-Net model on a parametric dataset:

```bash
python scripts/eff_parametric_train.py fit --config configs/eff_{equation}_parametric.yaml
```

**Available U-Net configs:**
- `configs/eff_burgers_parametric.yaml`
- `configs/eff_allen_cahn_parametric.yaml`
- `configs/eff_convection_parametric.yaml`
- `configs/eff_helmholtz2D_parametric.yaml`
- `configs/eff_helmholtz3D_parametric.yaml` (3D)
- `configs/eff_flow_mixing_parametric.yaml` (3D)

**Example:**
```bash
python scripts/eff_parametric_train.py fit --config configs/eff_burgers_parametric.yaml
```

### FNO Model

Train an FNO model on a parametric dataset:

```bash
python scripts/eff_parametric_fno_train.py fit --config configs/eff_{equation}_fno.yaml
```

**Available FNO configs (2D only):**
- `configs/eff_burgers_fno.yaml`
- `configs/eff_allen_cahn_fno.yaml`
- `configs/eff_convection_fno.yaml`
- `configs/eff_helmholtz2D_fno.yaml`

**Example:**
```bash
python scripts/eff_parametric_fno_train.py fit --config configs/eff_burgers_fno.yaml
```

## Testing

After training, evaluate the model on interpolation and extrapolation test splits:

**Quick option**: Use the automated script to run all experiments (see [Quick Start](#quick-start-run-all-experiments) above).

**Manual option**: Run training and testing individually as shown below:

### U-Net Model

```bash
python scripts/eff_parametric_train.py test \
  --config configs/eff_{equation}_parametric.yaml \
  --ckpt_path outputs/eff_{equation}_parametric/lightning_logs/version_*/checkpoints/epoch=999-step=1000.ckpt
```

**Example:**
```bash
python scripts/eff_parametric_train.py test \
  --config configs/eff_burgers_parametric.yaml \
  --ckpt_path outputs/eff_burgers_parametric/lightning_logs/version_0/checkpoints/epoch=999-step=1000.ckpt
```

### FNO Model

```bash
python scripts/eff_parametric_fno_train.py test \
  --config configs/eff_{equation}_fno.yaml \
  --ckpt_path outputs/eff_{equation}_fno/lightning_logs/version_*/checkpoints/epoch=999-step=1000.ckpt
```

**Example:**
```bash
python scripts/eff_parametric_fno_train.py test \
  --config configs/eff_burgers_fno.yaml \
  --ckpt_path outputs/eff_burgers_fno/lightning_logs/version_0/checkpoints/epoch=999-step=1000.ckpt
```

## Outputs

Each training/evaluation run produces outputs under `trainer.default_root_dir` (set in the config):

### Checkpoints and Logs

- **Checkpoints**: `outputs/eff_{equation}_{model_type}/lightning_logs/version_*/checkpoints/`
- **Training logs**: TensorBoard logs in the same directory

### Metrics (metrics-structures RunData)

Metrics are exported to JSON format: `outputs/eff_{equation}_{model_type}/metrics.json`

#### Split-Level Metrics

- **Interpolation metrics**:
  - `metadata.interp_relative_l2_mean`: Mean relative L2 error on interpolation split
  - `metadata.interp_relative_l2_std`: Standard deviation of relative L2 error
  - `metadata.interp_relative_l2`: Alias for mean (backward compatibility)

- **Extrapolation metrics**:
  - `metadata.extrap_relative_l2_mean`: Mean relative L2 error on extrapolation split
  - `metadata.extrap_relative_l2_std`: Standard deviation of relative L2 error
  - `metadata.extrap_relative_l2`: Alias for mean (backward compatibility)

#### Per-Solution Metrics

Detailed metrics for each test sample:

- `metadata.per_solution.interp[*].z`: Parameter vector for each interpolation sample
- `metadata.per_solution.interp[*].rl2`: Relative L2 error for each interpolation sample
- `metadata.per_solution.extrap[*].z`: Parameter vector for each extrapolation sample
- `metadata.per_solution.extrap[*].rl2`: Relative L2 error for each extrapolation sample

#### Inference Speed Metrics

- `wall_time`: Total test wall time in seconds
- `it_per_sec`: Batches processed per second
- `metadata.test_samples_per_sec`: Samples processed per second
- `metadata.interp_samples_per_sec`: Interpolation samples per second
- `metadata.extrap_samples_per_sec`: Extrapolation samples per second
- `metadata.test_ms_per_sample`: Milliseconds per sample

#### Run Configuration

The full run configuration (hyperparameters, dataset settings, etc.) is stored in `metadata` for reproducibility.

### Qualitative Plots

Visualization plots showing ground truth, predictions, and absolute errors:

- `outputs/eff_{equation}_{model_type}/gt_pred_abs_error_interp.png`: Interpolation split visualization
- `outputs/eff_{equation}_{model_type}/gt_pred_abs_error_extrap.png`: Extrapolation split visualization

Each plot shows up to `plot_max_rows` examples (configurable in YAML) with three columns:
- **GT**: Ground truth solution
- **Pred**: Model prediction
- **|Error|**: Absolute error between prediction and ground truth

## Metrics Details

### Relative L2 Error

The primary evaluation metric is the **relative L2 error** (also called relative L2 norm):

```
relative_l2 = ||pred - target||_2 / ||target||_2
```

This metric is computed per sample and then aggregated:
- **Mean**: Average relative L2 across all test samples in a split
- **Std**: Standard deviation of relative L2 across all test samples
- **Per-solution**: Individual relative L2 for each test sample (stored in `per_solution`)

### Data Splits

The framework uses parametric splits from `eff-physics-learn-dataset`:

- **train_few**: Small training set (typically 10 samples) for few-shot learning
- **interp**: Interpolation test set (parameters within the training parameter range)
- **extrap**: Extrapolation test set (parameters outside the training parameter range)

#### Balanced Splits (Default)

By default, the framework uses **balanced splits** to ensure deterministic, equal-sized test sets that are consistent across projects:

- **Equal-sized splits**: With `n_each=20` (default), you get exactly 20 interpolation samples + 20 extrapolation samples = 40 total test samples
- **Deterministic selection**: Using a fixed `seed` ensures the same samples are selected across different runs and projects
  - **Important**: Even with `balance_strategy: random`, the selection is deterministic because the random number generator is seeded by `seed=0`
  - Same seed → same "random" selection every time
  - Different seed → different selection
- **Reproducible evaluation**: This enables fair comparison across different methods and projects

The balanced split approach:
1. Trains on `n_train=10` samples (few-shot setting)
2. Partitions remaining samples into interpolation and extrapolation candidates based on convex hull membership
3. Selects exactly `n_each` samples from each partition using the specified `balance_strategy` and `seed`
   - The `seed` parameter controls the random number generator, making even "random" selection deterministic and reproducible

This is in contrast to the full partition approach (when `balance=False`), which uses all remaining samples and may result in unequal interp/extrap sizes that vary by dataset.

### Normalization

Solution fields (`u`) are normalized using statistics computed from the `train_few` split only:
- Mean and standard deviation are computed from training data
- All splits (train, interp, extrap) use the same normalization
- Predictions are denormalized before computing metrics and generating plots

## Configuration Files

Each dataset has separate configuration files for U-Net and FNO models. Key configuration options:

### Model Configuration

- `n_input_channels`: Number of input channels (coordinate grids)
- `n_output_channels`: Number of output channels (solution field, typically 1)
- `n_params`: Number of PDE parameters
- `hidden_channels`: Hidden channel dimension
- `lr`: Learning rate
- `metrics_out_path`: Path for metrics JSON export
- `plot_out_path`: Path for visualization plots

### Data Configuration

- `equation`: Dataset name (burgers, allen_cahn, convection, etc.)
- `data_dir`: Directory containing downloaded datasets
- `n_train`: Number of training samples (few-shot setting, typically 10)
- `batch_size`: Training batch size
- `include_grids`: Whether to include coordinate grids as input
- `standardize_u`: Whether to normalize solution fields
- `cache`: Whether to cache loaded datasets

#### Balanced Split Configuration (Default)

- `balance`: Enable balanced splits (default: `true`) - ensures equal-sized interp/extrap test sets
- `n_each`: Number of samples in each split (default: `20`) - results in 20 interp + 20 extrap = 40 total test samples
- `balance_strategy`: Selection strategy (default: `'random'`) - options: `'random'` or `'solution_nn'`
  - `'random'`: Random selection within each partition (seeded by `seed` parameter, so deterministic)
  - `'solution_nn'`: Solution-aware selection based on distance in solution embedding space (also seeded)
- `diversify`: Enable diversity selection (default: `false`) - uses farthest-point selection to encourage diversity within chosen subsets
- `seed`: Random seed (default: `0`) - controls all random operations, ensuring reproducibility

### FNO-Specific Configuration

- `modes1`, `modes2`: Number of Fourier modes in each spatial dimension
- `num_blocks`: ResNet block configuration

## Architecture Details

### U-Net Model

- Architecture: Conditioned U-Net with parameter conditioning
- Supports: 2D and 3D spatial problems
- Parameter conditioning: Multi-parameter support via `scalar_N` format
- Final layer: Zero-initialized for stable training

### FNO Model

- Architecture: Fourier Neural Operator (ResNet with FourierBasicBlock)
- Supports: 2D spatial problems only
- Parameter conditioning: Multi-parameter support via `scalar_N` format
- Initialization: Default PyTorch initialization (Kaiming/He)

## Notes

- Both models use the same data loading pipeline and metrics export format
- FNO models require coordinate grids as input (auto-constructed if missing from dataset)
- 3D datasets are only supported by U-Net models
- All metrics are exported using the `metrics-structures` format for standardized comparison

