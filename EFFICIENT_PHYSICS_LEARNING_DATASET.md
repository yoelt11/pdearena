# Efficient Physics Learning Dataset Integration

This document describes the modifications made to this repository to test **U-Net** and **FNO (Fourier Neural Operator)** models on parametric PDE datasets from the [eff-physics-learn-dataset](https://github.com/yoelt11/eff-physics-learn-dataset) repository.

## Overview

The modifications enable training and evaluation of both U-Net and FNO architectures on parametric PDE datasets with standardized metrics export using [metrics-structures](https://github.com/yoelt11/metrics-structures). The framework supports:

- **2D datasets**: burgers, allen_cahn, convection, helmholtz2D
- **3D datasets**: helmholtz3D, flow_mixing (U-Net only)
- **Few-shot learning**: Training on small parametric splits (typically 10 samples)
- **Interpolation and extrapolation evaluation**: Separate test splits for within-range and out-of-range parameter values

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
- `n_train`: Number of training samples (few-shot setting)
- `batch_size`: Training batch size
- `include_grids`: Whether to include coordinate grids as input
- `standardize_u`: Whether to normalize solution fields

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

