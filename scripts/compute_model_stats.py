#!/usr/bin/env python3
"""Compute parameter counts and model sizes for parametric U-Net and FNO models."""

import torch
from pdearena.models.eff_parametric_unet import EffParametricUNet
from pdearena.models.eff_parametric_fno import EffParametricFNO

# Model configurations matching the experiments
configs = {
    "U-Net": [
        {"n_params": 3, "n_dims": 2},  # Burgers
        {"n_params": 2, "n_dims": 2},  # Allen Cahn
        {"n_params": 1, "n_dims": 2},  # Convection
        {"n_params": 2, "n_dims": 2},  # Helmholtz2D
    ],
    "FNO": [
        {"n_params": 3, "n_dims": 2},  # Burgers
        {"n_params": 2, "n_dims": 2},  # Allen Cahn
        {"n_params": 1, "n_dims": 2},  # Convection
        {"n_params": 2, "n_dims": 2},  # Helmholtz2D
    ],
}

def count_parameters(model):
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())

def compute_model_size_mb(num_params):
    """Compute model size in MB (assuming float32 = 4 bytes per parameter)."""
    return (num_params * 4) / (1024 * 1024)

def main():
    print("Computing model statistics...")
    print("=" * 60)
    
    # U-Net configurations
    unet_params = []
    for config in configs["U-Net"]:
        model = EffParametricUNet(
            n_input_channels=2,
            n_output_channels=1,
            n_params=config["n_params"],
            hidden_channels=64,
            activation="gelu",
            norm=True,
            n_dims=config["n_dims"],
        )
        num_params = count_parameters(model)
        model_size_mb = compute_model_size_mb(num_params)
        unet_params.append(num_params)
        print(f"U-Net (n_params={config['n_params']}): {num_params:,} params, {model_size_mb:.2f} MB")
    
    # Average U-Net (they should be similar)
    avg_unet_params = sum(unet_params) / len(unet_params)
    avg_unet_size = compute_model_size_mb(avg_unet_params)
    print(f"\nU-Net average: {avg_unet_params:,.0f} params, {avg_unet_size:.2f} MB")
    
    print("\n" + "-" * 60)
    
    # FNO configurations
    fno_params = []
    for config in configs["FNO"]:
        model = EffParametricFNO(
            n_input_channels=2,
            n_output_channels=1,
            n_params=config["n_params"],
            hidden_channels=64,
            modes1=16,
            modes2=16,
            num_blocks=[1, 1, 1, 1],
            activation="gelu",
            norm=False,
        )
        num_params = count_parameters(model)
        model_size_mb = compute_model_size_mb(num_params)
        fno_params.append(num_params)
        print(f"FNO (n_params={config['n_params']}): {num_params:,} params, {model_size_mb:.2f} MB")
    
    # Average FNO (they should be similar)
    avg_fno_params = sum(fno_params) / len(fno_params)
    avg_fno_size = compute_model_size_mb(avg_fno_params)
    print(f"\nFNO average: {avg_fno_params:,.0f} params, {avg_fno_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("\nSummary for LaTeX table:")
    print(f"U-Net: {avg_unet_params:,.0f} params, {avg_unet_size:.2f} MB")
    print(f"FNO: {avg_fno_params:,.0f} params, {avg_fno_size:.2f} MB")

if __name__ == "__main__":
    main()
