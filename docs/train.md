# Training PDE surrogates

Thanks to [PyTorch Lightning](https://github.com/Lightning-AI/lightning), whether it's a single GPU experiment or multiple GPUs (even multiple nodes), setting up scalable training experiments should be fairly simple.

!!! tip

    We recommend a warmup learning rate schedule for distributed training.


## Standard PDE Surrogate Learning

```bash
python scripts/train.py -c <path/to/config>
```

For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```bash
python scripts/train.py -c configs/navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_smoke \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --data.time_gap=0 --data.time_history=4 --data.time_future=1 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

## Conditioned PDE Surrogate Learning

```bash
python scripts/cond_train.py -c <path/to/config>
```

For example, to run modern U-Net on Navier Stokes dataset on 4 GPUs use:

```bash
python scripts/cond_train.py -c configs/cond_navierstokes2d.yaml \
    --data.data_dir=/mnt/data/NavierStokes2D_cond_smoke_v1 \
    --trainer.strategy=ddp --trainer.devices=4 \
    --trainer.max_epochs=50 \
    --data.batch_size=8 \
    --model.name=Unetmod-64 \
    --model.lr=2e-4 \
    --optimizer=AdamW --optimizer.lr=2e-4 --optimizer.weight_decay=1e-5 \
    --lr_scheduler=LinearWarmupCosineAnnealingLR \
    --lr_scheduler.warmup_epochs=5 \
    --lr_scheduler.max_epochs=50 --lr_scheduler.eta_min=1e-7
```

## PDE Surrogate Learning with PDE-Refiner

```bash
python scripts/pderefiner_train.py -c <path/to/config>
```

## Parametric modality (eff-physics-learn-dataset)

This repo includes a small training/eval pipeline that uses the **parametric modality**
from [`eff-physics-learn-dataset`](https://github.com/yoelt11/eff-physics-learn-dataset) and exports
results using [`metrics-structures`](https://github.com/yoelt11/metrics-structures).

### What scripts/configs were added

- **U-Net entrypoint**: `scripts/eff_parametric_train.py`
  - Uses PyTorch Lightning's CLI, so it supports subcommands like `fit` and `test`
- **FNO entrypoint**: `scripts/eff_parametric_fno_train.py`
  - Uses PyTorch Lightning's CLI, so it supports subcommands like `fit` and `test`
- **U-Net Configs** (one per dataset):
  - `configs/eff_burgers_parametric.yaml`
  - `configs/eff_allen_cahn_parametric.yaml`
  - `configs/eff_convection_parametric.yaml`
  - `configs/eff_helmholtz2D_parametric.yaml`
  - `configs/eff_flow_mixing_parametric.yaml` (3D)
  - `configs/eff_helmholtz3D_parametric.yaml` (3D)
- **FNO Configs** (one per dataset, 2D only):
  - `configs/eff_burgers_fno.yaml`
  - `configs/eff_allen_cahn_fno.yaml`
  - `configs/eff_convection_fno.yaml`
  - `configs/eff_helmholtz2D_fno.yaml`

### Download the dataset (once)

The `eff-physics-learn-dataset` download utility expects a `dataset_links.toml`. You can fetch it and download `burgers` like this:

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

To download **all** datasets listed in `eff_dataset_links.toml`:

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

### Train (few-shot parametric split)

**U-Net model:**
```bash
source .venv/bin/activate
python scripts/eff_parametric_train.py fit --config configs/eff_burgers_parametric.yaml
```

**FNO model:**
```bash
source .venv/bin/activate
python scripts/eff_parametric_fno_train.py fit --config configs/eff_burgers_fno.yaml
```

### Test (interp + extrap) and write metrics artifact

Important: to test the **trained** weights, pass `--ckpt_path` (otherwise Lightning will test a freshly initialized model).

Find checkpoints:

```bash
# U-Net checkpoints
find outputs/eff_burgers_parametric -name "*.ckpt"

# FNO checkpoints
find outputs/eff_burgers_fno -name "*.ckpt"
```

Then test using (for example) `last.ckpt`:

**U-Net model:**
```bash
source .venv/bin/activate
python scripts/eff_parametric_train.py test \
  --config configs/eff_burgers_parametric.yaml \
  --ckpt_path outputs/eff_burgers_parametric/lightning_logs/version_*/checkpoints/last.ckpt
```

**FNO model:**
```bash
source .venv/bin/activate
python scripts/eff_parametric_fno_train.py test \
  --config configs/eff_burgers_fno.yaml \
  --ckpt_path outputs/eff_burgers_fno/lightning_logs/version_*/checkpoints/last.ckpt
```

### Outputs

Each run writes under `trainer.default_root_dir` (set in the config). For example, `configs/eff_burgers_parametric.yaml` uses:

- **Checkpoints / Lightning logs**: `outputs/eff_burgers_parametric/lightning_logs/version_*/`
- **Metrics (metrics-structures RunData)**: `outputs/eff_burgers_parametric/metrics.json`
  - Split summaries:
    - `metadata.interp_relative_l2_mean`, `metadata.interp_relative_l2_std`
    - `metadata.extrap_relative_l2_mean`, `metadata.extrap_relative_l2_std`
  - Per-solution metrics:
    - `metadata.per_solution.interp[*].{z, rl2}`
    - `metadata.per_solution.extrap[*].{z, rl2}`
  - Inference speed:
    - `wall_time` (seconds)
    - `it_per_sec` (batches/sec)
    - `metadata.test_samples_per_sec`, `metadata.test_ms_per_sample`, etc.
- **Qualitative plots**:
  - `outputs/eff_burgers_parametric/gt_pred_abs_error_interp.png`
  - `outputs/eff_burgers_parametric/gt_pred_abs_error_extrap.png`


For example, to run PDE-Refiner with modern U-Net on Kuramoto-Sivashinsky use:

```bash
python scripts/pderefiner_train.py -c configs/kuramotosivashinsky1d.yaml \
    --data.data_dir /mnt/data/KuramotoSivashinsky1D/ \
    --trainer.devices=1
```

## Dataloading philosophy

- Use modern [`torchdata`](https://pytorch.org/data/) [iterable datapipes](https://pytorch.org/data/beta/torchdata.datapipes.iter.html#torchdata.datapipes.iter.IterDataPipe) as they scale better with cloud storage.
- Use equally sized shards for simpler scaling with PyTorch DDP.
