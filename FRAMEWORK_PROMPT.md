# High-Level Framework Prompt: Integrating eff-physics-learn-dataset and metrics-structures

## Overview

Create a training and evaluation framework for parametric PDE datasets that integrates:
- **`eff-physics-learn-dataset`**: Repository for loading parametric PDE datasets with standardized train/interpolation/extrapolation splits
- **`metrics-structures`**: Repository for standardized metrics export and storage

The framework should provide consistent entry points for training and testing, regardless of the underlying neural architecture or training framework.

---

## Core Integration Points

### 1. Dataset Integration (`eff-physics-learn-dataset`)

**Purpose**: Integrate the `eff-physics-learn-dataset` repository for data loading and splitting.

**Key Functionality**:
- **Dataset Loading**: Use `load_pde_dataset(equation, data_dir, cache)` to load parametric PDE datasets
- **Parametric Splits**: Generate deterministic splits via `ds.parametric_splits(seed, n_train)` which returns:
  - `train_few`: Small training set (few-shot learning scenario)
  - `interp`: Interpolation test set (parameters within training range)
  - `extrap`: Extrapolation test set (parameters outside training range)
- **Data Structure**: Each sample contains:
  - `u`: Solution field (2D or 3D spatial grid)
  - `params`: Parameter vector (P parameters per sample)
  - Optional coordinate grids: `X_grid`, `Y_grid`, `Z_grid`, `T_grid`

**Data Processing**:
- **Normalization**: Compute mean/std statistics from `train_few` split only, apply to all splits
- **Grid Handling**: 
  - Stack coordinate grids into input channels if available
  - Auto-construct normalized coordinate grids [0, 1] if missing from dataset
  - Support 2D (X, T) and 3D (X, Y, Z) spatial problems
- **Batch Format**: Return batches as `(x, y, z)` where:
  - `x`: Input features (coordinate grids or other inputs)
  - `y`: Target solution field
  - `z`: Parameter conditioning vector

**Intent**: Provide a standardized interface to parametric PDE datasets with consistent train/test splits and normalization.

---

### 2. Metrics Export Integration (`metrics-structures`)

**Purpose**: Export evaluation results using the `metrics-structures` repository for standardized storage and analysis.

**Key Functionality**:
- **Metrics Collection**: During evaluation, collect:
  - **Aggregate metrics**: Mean squared error (MSE), relative L2 error per split (interp/extrap)
  - **Per-solution metrics**: Individual error metrics for each test sample
  - **Timing metrics**: Wall time, batches per second
  - **Run configuration**: All hyperparameters and dataset settings
- **Export Format**: Create `RunData` object from `metrics_structures`:
  - Store aggregate metrics and run config in `metadata` field
  - Store per-solution metrics in nested structure
  - Populate `wall_time` and `it_per_sec` fields if available
  - Save as JSON (`.json`) or pickle format
- **Visualization**: Generate qualitative plots (ground truth, prediction, absolute error) for interp and extrap splits separately

**Intent**: Ensure all evaluation results are stored in a standardized format that can be easily compared across different models, datasets, and runs.

---

## General Architecture

### Data Flow

1. **Training Phase**:
   - Load dataset using `eff-physics-learn-dataset`
   - Generate parametric splits (train_few, interp, extrap)
   - Compute normalization from train_few only
   - Train model on train_few split

2. **Evaluation Phase**:
   - Load trained model checkpoint
   - Evaluate on interp and extrap splits separately
   - Collect metrics per split
   - Export metrics using `metrics-structures` format
   - Generate visualization plots

### Key Design Principles

1. **Separation of Concerns**:
   - Data loading: Wrapper around `eff-physics-learn-dataset`
   - Model training: Framework-agnostic (can use PyTorch, JAX, etc.)
   - Metrics export: Wrapper around `metrics-structures`

2. **Standardization**:
   - Use `eff-physics-learn-dataset` for consistent data loading and splits
   - Use `metrics-structures` for consistent metrics export
   - Store full run configuration in metrics export

3. **Reproducibility**:
   - Deterministic splits via `parametric_splits(seed, n_train)`
   - Store complete configuration in metrics export
   - Normalization computed only from training split

4. **Flexibility**:
   - Support any neural architecture
   - Support 2D and 3D spatial problems
   - Auto-construct coordinate grids if missing
   - Framework-agnostic (PyTorch, JAX, TensorFlow, etc.)

---

## Entry Points

### Training
```bash
python scripts/train.py --config configs/{equation}_{model_type}.yaml
```

### Testing
```bash
python scripts/train.py test \
  --config configs/{equation}_{model_type}.yaml \
  --ckpt_path outputs/{equation}_{model_type}/checkpoints/best.ckpt
```

**Intent**: Provide simple, consistent command-line interfaces for training and evaluation.

---

## Configuration Structure

Configuration files should specify:
- **Dataset settings**: Equation name, data directory, split parameters (seed, n_train)
- **Model settings**: Architecture hyperparameters
- **Training settings**: Learning rate, batch size, epochs, etc.
- **Export settings**: Paths for metrics export and visualization

**Intent**: All settings should be configurable via configuration files for reproducibility.

---

## Implementation Checklist

- [ ] Integrate `eff-physics-learn-dataset` for dataset loading
- [ ] Implement parametric split generation (train_few, interp, extrap)
- [ ] Implement normalization computation from training split
- [ ] Implement coordinate grid handling (stacking and auto-construction)
- [ ] Integrate `metrics-structures` for metrics export
- [ ] Implement metric collection during evaluation (aggregate and per-solution)
- [ ] Implement visualization generation
- [ ] Create training and testing entry points
- [ ] Create configuration file structure
- [ ] Ensure reproducibility (seeding, deterministic splits)

---

## Dependencies

- `eff-physics-learn-dataset`: Dataset loading and splitting
- `metrics-structures`: Metrics export and storage
- Deep learning framework (PyTorch, JAX, TensorFlow, etc.)
- Numerical libraries (NumPy, etc.)
- Visualization libraries (Matplotlib, etc.)

---

## Notes

- The framework assumes parametric PDE datasets where each solution depends on a set of parameters
- Training uses a "few-shot" approach with `n_train` samples
- Evaluation separates interpolation (within parameter range) and extrapolation (outside parameter range)
- Coordinate grids can be provided by the dataset or auto-constructed as normalized coordinates [0, 1]
- Solution fields are normalized using statistics from the training split only
- The framework is architecture-agnostic and can work with any neural network model

