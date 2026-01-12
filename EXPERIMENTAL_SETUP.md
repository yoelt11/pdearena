# Experimental Setup

## Framework and Models

The PDEArena framework [^1] was used to implement and evaluate two neural operator architectures—Fourier Neural Operator (FNO) and U-Net—to compare their generalization capabilities on parametric partial differential equation (PDE) datasets. Both models were adapted to work with the parametric modality from the eff-physics-learn-dataset repository [^2], enabling standardized evaluation across multiple PDE families with varying parameter dependencies.

### Model Architectures

**U-Net**: A conditioned U-Net architecture with parameter conditioning support. The model uses skip connections and zero-initialized final layers for stable training. The architecture supports both 2D and 3D spatial problems, with multi-parameter conditioning implemented via a `scalar_N` format where N is the number of PDE parameters.

**FNO**: A Fourier Neural Operator implemented as a ResNet with FourierBasicBlock layers. The model uses Fourier transforms in the spectral domain to learn mappings between function spaces. The architecture supports 2D spatial problems with multi-parameter conditioning via the same `scalar_N` format as U-Net.

### Datasets

Experiments were conducted on parametric PDE datasets from eff-physics-learn-dataset, including:

- **2D datasets**: Burgers equation, Allen-Cahn equation, Convection equation, Helmholtz 2D
- **3D datasets**: Helmholtz 3D, Flow Mixing (U-Net only)

Each dataset contains 200 solution samples parameterized by PDE-specific coefficients (e.g., diffusion coefficient, reaction rate, wave number). Solutions are provided on spatial grids of size 64×64 for 2D problems and 64×64×64 for 3D problems.

### Training Configuration

**Data Splits**: All experiments used a few-shot learning setup with 10 training samples (`n_train=10`). The datasets were split into three sets using deterministic parametric splits (seed=0):
- **train_few**: 10 samples for training
- **interp**: Interpolation test set (parameters within training range)
- **extrap**: Extrapolation test set (parameters outside training range)

**Data Preprocessing**: 
- Solution fields were normalized using mean and standard deviation computed from the training split only
- Coordinate grids (X, Y, Z, T) were included as input channels; grids were auto-constructed as normalized coordinates [0,1] when not provided in the dataset
- All data was cached in memory for faster training

**Hyperparameters**:
- **Optimizer**: AdamW with learning rate 1×10⁻³ and weight decay 0.0
- **Loss function**: Mean Squared Error (MSE)
- **Batch size**: 8
- **Training epochs**: 1000
- **Hidden channels**: 64
- **Activation**: GELU
- **U-Net specific**: Normalization enabled (`norm=true`), zero-initialized final layer
- **FNO specific**: 16 Fourier modes in each spatial dimension (`modes1=16`, `modes2=16`), ResNet block configuration `[1,1,1,1]`, no normalization (`norm=false`)

**Training Framework**: PyTorch Lightning with automatic mixed precision and checkpointing enabled. All random seeds were set to 0 for reproducibility.

### Evaluation Methodology

Models were evaluated on both interpolation and extrapolation test splits to assess generalization capabilities:

- **Interpolation**: Tests generalization within the parameter range seen during training
- **Extrapolation**: Tests generalization outside the training parameter range

**Metrics**:
- **Relative L2 Error**: Primary evaluation metric computed as ||pred - target||₂ / ||target||₂, reported as mean and standard deviation per split
- **Per-solution metrics**: Individual relative L2 errors for each test sample, stored with corresponding parameter vectors
- **Inference speed**: Wall time, samples per second, and milliseconds per sample

All metrics were exported using the metrics-structures format [^3] for standardized comparison and reproducibility.

### Hardware and Software

**Hardware**: 
- GPU: NVIDIA RTX A4500
- CUDA: Enabled for GPU acceleration

**Software**:
- Framework: PDEArena (modified) [^1]
- Deep Learning: PyTorch with PyTorch Lightning
- Dataset: eff-physics-learn-dataset [^2]
- Metrics: metrics-structures [^3]

### Reproducibility

All experimental configurations are provided as YAML files in the repository:
- U-Net configs: `configs/eff_{equation}_parametric.yaml`
- FNO configs: `configs/eff_{equation}_fno.yaml`

Training and evaluation scripts:
- U-Net: `scripts/eff_parametric_train.py`
- FNO: `scripts/eff_parametric_fno_train.py`

All random seeds were fixed (seed=0) for deterministic data splits and model initialization. The complete experimental setup, including data loading, model architectures, and evaluation procedures, is documented in the repository modifications [^1].

---

[^1]: Repository modifications: [Link to be provided]
[^2]: eff-physics-learn-dataset: https://github.com/yoelt11/eff-physics-learn-dataset
[^3]: metrics-structures: https://github.com/yoelt11/metrics-structures

