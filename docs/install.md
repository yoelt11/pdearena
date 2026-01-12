# Installation Guide

```bash title="clone the repo"
git clone https://github.com/microsoft/pdearena
```

=== "`conda`"


    ```bash title="create and activate env"
    cd pdearena
    conda env create --file docker/environment.yml
    conda activate pdearena
    ```

    ```bash title="install this package"
    # install so the project is in PYTHONPATH
    pip install -e .
    ```

    ```bash title="additionally installing Clifford Neural Layers"
    pip install "cliffordlayers @ git+https://github.com/microsoft/cliffordlayers"
    ```

    If you also want to do data generation:

    ```bash
    pip install -e ".[datagen]"
    ```

=== "`uv`"

    This repository is a normal Python package (see `setup.py`), so you can use `uv` to create a virtualenv and install it in editable mode.

    ```bash title="create and activate env (Python 3.8 to match docker/environment.yml)"
    cd pdearena

    # Install a matching Python (optional but recommended for reproducibility)
    uv python install 3.8

    # Create a venv in-project
    uv venv --python 3.8 .venv
    source .venv/bin/activate
    ```

    ```bash title="install this package"
    uv pip install -e .
    ```

    ```bash title="PyTorch (CPU vs GPU wheels)"
    # If you want CPU-only PyTorch, the default index is fine (no extra flags needed).
    #
    # If you want CUDA-enabled wheels, install using the official PyTorch wheel index
    # that matches your CUDA version. Example (CUDA 12.1):
    # uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    ```

    ```bash title="(optional) approximate the pinned docker environment via pip"
    uv pip install -r requirements/uv-repro.txt
    ```

    ```bash title="(optional) extra research packages"
    uv pip install \
      "eff-physics-learn-dataset @ git+https://github.com/yoelt11/eff-physics-learn-dataset" \
      "metrics-structures @ git+https://github.com/yoelt11/metrics-structures"
    ```

    ```bash title="additionally installing Clifford Neural Layers"
    uv pip install "cliffordlayers @ git+https://github.com/microsoft/cliffordlayers"
    ```

    If you also want to do data generation:

    ```bash
    uv pip install -e ".[datagen]"
    ```

    !!! note

        The conda environment includes GPU runtime packages like `cudatoolkit`. With `uv`/pip, GPU support comes from:

        - Your system NVIDIA driver/runtime, and/or
        - Installing a CUDA-enabled PyTorch wheel from the official PyTorch wheel index (see PyTorch install instructions).


=== "`docker`"


    ```bash title="build docker container"
    cd docker
    docker build -t pdearena .
    ```

    ```bash title="run docker container"
    cd pdearena
    docker run --gpus all -it --rm --user $(id -u):$(id -g) \
        -v $(pwd):/code -v /mnt/data:/data --workdir /code -e PYTHONPATH=/code \
        pdearena:latest
    ```


    !!! note

        - `--gpus all -it --rm --user $(id -u):$(id -g)`: enables using all GPUs and runs an interactive session with current user's UID/GUID to avoid `docker` writing files as root.
        - `-v $(pwd):/code -v /mnt/data:/data --workdir /code`: mounts current directory and data directory (i.e. the cloned git repo) to `/code` and `/data` respectively, and use the `code` directory as the current working directory.
