#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to run all eff-physics-learn-dataset experiments.

This script:
1. Finds all eff_*.yaml config files in configs/
2. For each config and seed, runs training (fit) then testing
3. Automatically determines which training script to use (FNO vs U-Net)
4. Finds the checkpoint path automatically for testing
5. Organizes outputs by seed in outputs/seed_{seed}/ directories

Usage:
    uv run python scripts/run_all_eff_experiments.py --seeds 0,1,2
    uv run python scripts/run_all_eff_experiments.py --seeds 1,2  # Skip seed 0 if already run
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from collections import defaultdict


def find_configs():
    """Find all eff_*.yaml config files."""
    config_dir = Path("configs")
    configs = sorted(glob.glob(str(config_dir / "eff_*.yaml")))
    return configs


def get_script_for_config(config_path):
    """Determine which training script to use based on config filename."""
    config_name = Path(config_path).stem
    if "_fno" in config_name:
        return "scripts/eff_parametric_fno_train.py"
    else:
        return "scripts/eff_parametric_train.py"


def get_output_dir(config_path, seed=None):
    """Extract output directory from config file, optionally modified for seed."""
    with open(config_path, "r") as f:
        for line in f:
            if "default_root_dir:" in line:
                # Extract the path after default_root_dir:
                parts = line.split("default_root_dir:", 1)
                if len(parts) > 1:
                    output_dir = parts[1].strip()
                    # If seed is specified, modify output directory
                    if seed is not None:
                        # Change outputs/eff_* to outputs/seed_{seed}/eff_*
                        if output_dir.startswith("outputs/"):
                            base_name = output_dir.replace("outputs/", "")
                            output_dir = f"outputs/seed_{seed}/{base_name}"
                    return output_dir
    return None


def update_config_seed(config_path, seed, temp_dir):
    """Create a temporary config file with updated seed values and output directory."""
    with open(config_path, "r") as f:
        lines = f.readlines()
    
    updated_lines = []
    in_data_section = False
    in_model_section = False
    
    for i, line in enumerate(lines):
        # Update seed_everything
        if line.strip().startswith("seed_everything:"):
            updated_lines.append(f"seed_everything: {seed}\n")
        # Update default_root_dir to include seed directory
        elif "default_root_dir:" in line:
            parts = line.split("default_root_dir:", 1)
            if len(parts) > 1:
                original_dir = parts[1].strip()
                # Modify to include seed directory
                if original_dir.startswith("outputs/"):
                    base_name = original_dir.replace("outputs/", "")
                    new_dir = f"outputs/seed_{seed}/{base_name}"
                    updated_lines.append(f"  default_root_dir: {new_dir}\n")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        # Update metrics_out_path to include seed directory
        elif "metrics_out_path:" in line:
            parts = line.split("metrics_out_path:", 1)
            if len(parts) > 1:
                original_path = parts[1].strip()
                # Modify to include seed directory
                if original_path.startswith("outputs/"):
                    base_name = original_path.replace("outputs/", "")
                    new_path = f"outputs/seed_{seed}/{base_name}"
                    updated_lines.append(f"  metrics_out_path: {new_path}\n")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        # Update plot_out_path to include seed directory
        elif "plot_out_path:" in line:
            parts = line.split("plot_out_path:", 1)
            if len(parts) > 1:
                original_path = parts[1].strip()
                # Modify to include seed directory
                if original_path.startswith("outputs/"):
                    base_name = original_path.replace("outputs/", "")
                    new_path = f"outputs/seed_{seed}/{base_name}"
                    updated_lines.append(f"  plot_out_path: {new_path}\n")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        # Update data.seed
        elif line.strip().startswith("data:"):
            in_data_section = True
            in_model_section = False
            updated_lines.append(line)
        elif line.strip().startswith("model:"):
            in_model_section = True
            in_data_section = False
            updated_lines.append(line)
        elif in_data_section and line.strip().startswith("seed:"):
            updated_lines.append(f"  seed: {seed}\n")
            in_data_section = False
        else:
            updated_lines.append(line)
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith(" ") and ":" in line:
                in_data_section = False
                in_model_section = False
    
    # Write to temp file
    temp_config = temp_dir / Path(config_path).name
    with open(temp_config, "w") as f:
        f.writelines(updated_lines)
    
    return str(temp_config)


def find_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if output_dir is None:
        return None
    
    # Look for checkpoints in lightning_logs
    checkpoint_pattern = Path(output_dir) / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt"
    checkpoints = sorted(glob.glob(str(checkpoint_pattern)), reverse=True)
    
    if checkpoints:
        return checkpoints[0]
    
    # Try to find epoch=999-step=1000.ckpt specifically
    checkpoint_pattern = Path(output_dir) / "lightning_logs" / "version_*" / "checkpoints" / "epoch=999-step=1000.ckpt"
    checkpoints = glob.glob(str(checkpoint_pattern))
    if checkpoints:
        return checkpoints[0]
    
    return None


def load_metrics(output_dir):
    """Load metrics from metrics.json file."""
    metrics_file = Path(output_dir) / "metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metrics from {metrics_file}: {e}")
        return None


def get_equation_name(config_path):
    """Extract equation name from config path."""
    name = Path(config_path).stem
    # Remove eff_ prefix and _fno/_parametric suffix
    name = name.replace("eff_", "")
    if "_fno" in name:
        name = name.replace("_fno", "")
    elif "_parametric" in name:
        name = name.replace("_parametric", "")
    # Capitalize first letter of each word
    return " ".join(word.capitalize() for word in name.split("_"))


def get_model_name(config_path):
    """Extract model name from config path."""
    name = Path(config_path).stem
    if "_fno" in name:
        return "FNO"
    else:
        return "U-Net"


def format_error(mean, std):
    """Format error as mean ± std."""
    if mean is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.2f}" if std is not None else f"{mean:.4f}"


def format_time(ms):
    """Format time in milliseconds."""
    if ms is None:
        return "N/A"
    return f"{ms:.2f}"


def generate_summary_tables(configs):
    """Generate summary tables for interpolation and extrapolation."""
    # Collect data by equation
    equation_data = defaultdict(lambda: {"U-Net": {}, "FNO": {}})
    
    for config in configs:
        output_dir = get_output_dir(config)
        if output_dir is None:
            continue
        
        metrics = load_metrics(output_dir)
        if metrics is None:
            continue
        
        equation = get_equation_name(config)
        model = get_model_name(config)
        
        metadata = metrics.get("metadata", {})
        
        # Extract metrics
        interp_mean = metadata.get("interp_relative_l2_mean")
        interp_std = metadata.get("interp_relative_l2_std")
        extrap_mean = metadata.get("extrap_relative_l2_mean")
        extrap_std = metadata.get("extrap_relative_l2_std")
        ms_per_sample = metadata.get("test_ms_per_sample")
        
        equation_data[equation][model] = {
            "interp_mean": interp_mean,
            "interp_std": interp_std,
            "extrap_mean": extrap_mean,
            "extrap_std": extrap_std,
            "ms_per_sample": ms_per_sample,
        }
    
    # Generate tables
    tables = []
    
    # Interpolation table
    tables.append("\n" + "="*120)
    tables.append("INTERPOLATION RESULTS")
    tables.append("="*120)
    tables.append("")
    
    # Header
    header = (
        f"{'Equation':<20} {'Model':<10} {'Zero-shot (L2 Error)':<25} {'ACC Time (ms)':<15} "
        f"{'FT after 50 epochs':<20} {'ACC Time':<12} {'FT after 250 epochs':<20} {'ACC Time':<12} "
        f"{'FT after 500 epochs':<20} {'ACC Time':<12} {'FT after 750 epochs':<20} {'ACC Time':<12} "
        f"{'FT after 1000 epochs':<20} {'ACC Time':<12}\n"
    )
    tables.append(header)
    tables.append("-" * 120)
    
    # Sort equations
    sorted_equations = sorted(equation_data.keys())
    
    for equation in sorted_equations:
        data = equation_data[equation]
        for model in ["U-Net", "FNO"]:
            if model not in data or not data[model]:
                continue
            
            m = data[model]
            interp_error = format_error(m.get("interp_mean"), m.get("interp_std"))
            acc_time = format_time(m.get("ms_per_sample"))
            
            row = (
                f"{equation:<20} {model:<10} {interp_error:<25} {acc_time:<15} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12}"
            )
            tables.append(row)
    
    # Extrapolation table
    tables.append("\n" + "="*120)
    tables.append("EXTRAPOLATION RESULTS")
    tables.append("="*120)
    tables.append("")
    
    # Header (same as interpolation)
    tables.append(header)
    tables.append("-" * 120)
    
    for equation in sorted_equations:
        data = equation_data[equation]
        for model in ["U-Net", "FNO"]:
            if model not in data or not data[model]:
                continue
            
            m = data[model]
            extrap_error = format_error(m.get("extrap_mean"), m.get("extrap_std"))
            acc_time = format_time(m.get("ms_per_sample"))
            
            row = (
                f"{equation:<20} {model:<10} {extrap_error:<25} {acc_time:<15} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12}"
            )
            tables.append(row)
    
    return "\n".join(tables)


def run_command(cmd, config_name):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Command failed for {config_name}")
        print(f"Command: {' '.join(cmd)}")
        return False
    else:
        print(f"\n✅ SUCCESS: Completed for {config_name}")
        return True


def run_experiments_for_seed(configs, seed):
    """Run all experiments for a specific seed."""
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENTS FOR SEED {seed}")
    print("="*80)
    
    # Create seed output directory
    seed_dir = Path(f"outputs/seed_{seed}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for modified configs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("\n" + "="*80)
        print(f"PHASE 1: TRAINING (fit) - SEED {seed}")
        print("="*80)
        
        # Phase 1: Training
        training_results = {}
        for config in configs:
            config_name = Path(config).stem
            script = get_script_for_config(config)
            
            # Create temp config with updated seed
            temp_config = update_config_seed(config, seed, temp_path)
            
            cmd = ["uv", "run", "python", script, "fit", "--config", temp_config]
            success = run_command(cmd, f"{config_name} (seed={seed})")
            training_results[config] = success
        
        print("\n" + "="*80)
        print(f"PHASE 2: TESTING - SEED {seed}")
        print("="*80)
        
        # Phase 2: Testing
        test_results = {}
        for config in configs:
            config_name = Path(config).stem
            script = get_script_for_config(config)
            output_dir = get_output_dir(config, seed=seed)
            checkpoint = find_checkpoint(output_dir)
            
            if checkpoint is None:
                print(f"\n⚠️  WARNING: No checkpoint found for {config_name} (seed={seed})")
                print(f"   Skipping test for {config_name}")
                test_results[config] = False
                continue
            
            # Create temp config with updated seed
            temp_config = update_config_seed(config, seed, temp_path)
            
            cmd = [
                "uv", "run", "python", script, "test",
                "--config", temp_config,
                "--ckpt_path", checkpoint
            ]
            success = run_command(cmd, f"{config_name} (seed={seed})")
            test_results[config] = success
        
        # Summary for this seed
        print("\n" + "="*80)
        print(f"SUMMARY - SEED {seed}")
        print("="*80)
        
        print("\nTraining Results:")
        for config, success in training_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {Path(config).stem}")
        
        print("\nTesting Results:")
        for config, success in test_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {Path(config).stem}")
        
        return training_results, test_results


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description="Run all eff-physics-learn-dataset experiments")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated list of seeds to run (e.g., '0,1,2' or '1,2')"
    )
    args = parser.parse_args()
    
    # Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    except ValueError:
        print(f"Error: Invalid seeds format: {args.seeds}")
        print("Expected format: --seeds 0,1,2")
        sys.exit(1)
    
    configs = find_configs()
    
    if not configs:
        print("No eff_*.yaml config files found in configs/")
        sys.exit(1)
    
    print(f"Found {len(configs)} config files:")
    for config in configs:
        print(f"  - {config}")
    print(f"\nRunning experiments for seeds: {seeds}")
    
    # Run experiments for each seed
    all_training_results = {}
    all_test_results = {}
    
    for seed in seeds:
        training_results, test_results = run_experiments_for_seed(configs, seed)
        all_training_results[seed] = training_results
        all_test_results[seed] = test_results
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL SEEDS")
    print("="*80)
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        training_success = all(all_training_results[seed].values())
        testing_success = all(all_test_results[seed].values())
        status = "✅" if (training_success and testing_success) else "❌"
        print(f"  {status} Training: {sum(all_training_results[seed].values())}/{len(all_training_results[seed])} succeeded")
        print(f"  {status} Testing: {sum(all_test_results[seed].values())}/{len(all_test_results[seed])} succeeded")
    
    # Check if all succeeded
    all_training_success = all(
        all(all_training_results[s].values()) for s in seeds
    )
    all_testing_success = all(
        all(all_test_results[s].values()) for s in seeds
    )
    
    if all_training_success and all_testing_success:
        print("\n🎉 All experiments completed successfully!")
        print(f"\nResults are organized in:")
        for seed in seeds:
            print(f"  - outputs/seed_{seed}/")
        return 0
    else:
        print("\n⚠️  Some experiments failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
