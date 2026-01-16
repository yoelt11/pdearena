# PDE Arena - Efficient Physics Learning Dataset Integration

This repository contains the implementation and results for training U-Net and FNO (Fourier Neural Operator) models on parametric PDE datasets from the [eff-physics-learn-dataset](https://github.com/yoelt11/eff-physics-learn-dataset) repository.

## 📖 Documentation

**👉 [Complete Documentation: EFFICIENT_PHYSICS_LEARNING_DATASET.md](EFFICIENT_PHYSICS_LEARNING_DATASET.md)**

The main documentation includes:
- Quick start guide for running all experiments
- Dataset setup and configuration
- Training and testing procedures
- Reproducibility information with exact experimental setup
- Complete results tables with aggregated statistics
- Dataset indices for all seeds

## 🚀 Quick Start

### Run All Experiments

```bash
# Run experiments for seeds 0, 1, 2
uv run python scripts/run_all_eff_experiments.py --seeds 0,1,2

# Aggregate results across seeds
uv run python scripts/aggregate_seed_results.py --seeds 0,1,2
```

## 📊 Results

The repository includes complete reproducibility information:

- **Seeds Used**: 0, 1, 2
- **Equations Tested**: allen_cahn, burgers, convection, helmholtz2D
- **Models**: U-Net and FNO
- **Results**: Aggregated mean ± std across all seeds

See [EFFICIENT_PHYSICS_LEARNING_DATASET.md](EFFICIENT_PHYSICS_LEARNING_DATASET.md#reproducibility) for complete results tables and dataset indices.

## 📁 Repository Structure

```
├── configs/                    # Configuration files for all experiments
├── scripts/                    # Training and evaluation scripts
│   ├── run_all_eff_experiments.py    # Multi-seed experiment runner
│   ├── aggregate_seed_results.py     # Results aggregation script
│   └── ...
├── results/reproducibility/    # Reproducibility artifacts
│   ├── split_indices/         # Exact data splits for each seed
│   ├── example_metrics/       # Example output files
│   └── aggregated_summary_tables.txt
└── EFFICIENT_PHYSICS_LEARNING_DATASET.md  # Complete documentation
```

## 🔬 Reproducibility

This repository is designed for full reproducibility:

- **Exact data splits**: Train/interp/extrap indices for each seed
- **Complete configuration**: All hyperparameters and settings
- **Aggregated results**: Mean ± std statistics across seeds
- **Detailed documentation**: Step-by-step reproduction instructions

See the [Reproducibility section](EFFICIENT_PHYSICS_LEARNING_DATASET.md#reproducibility) for details.

## 📝 Citation

If you use this code or results, please cite:

```bibtex
@software{pdearena_eff_physics,
  title = {PDE Arena - Efficient Physics Learning Dataset Integration},
  author = {Torres, Edgar},
  year = {2026},
  url = {https://github.com/yoelt11/pdearena}
}
```

## 📄 License

See LICENSE file for details.
