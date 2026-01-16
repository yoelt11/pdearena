#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Script to aggregate results across multiple seeds.

This script:
1. Reads metrics from outputs/seed_{seed}/ directories
2. Computes mean ± std across seeds for each equation/model combination
3. Generates aggregated summary tables

Usage:
    uv run python scripts/aggregate_seed_results.py --seeds 0,1,2
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def find_configs():
    """Find all eff_*.yaml config files."""
    config_dir = Path("configs")
    configs = sorted(glob.glob(str(config_dir / "eff_*.yaml")))
    return configs


def get_equation_name(config_path):
    """Extract equation name from config path."""
    name = Path(config_path).stem
    name = name.replace("eff_", "")
    if "_fno" in name:
        name = name.replace("_fno", "")
    elif "_parametric" in name:
        name = name.replace("_parametric", "")
    return " ".join(word.capitalize() for word in name.split("_"))


def get_model_name(config_path):
    """Extract model name from config path."""
    name = Path(config_path).stem
    if "_fno" in name:
        return "FNO"
    else:
        return "U-Net"


def load_metrics(output_dir):
    """Load metrics from metrics.json file."""
    metrics_file = Path(output_dir) / "metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except Exception as e:
        return None


def aggregate_results(seeds, configs):
    """Aggregate results across seeds."""
    # Structure: equation -> model -> seed -> metrics
    all_data = defaultdict(lambda: defaultdict(dict))
    
    for seed in seeds:
        seed_dir = Path(f"outputs/seed_{seed}")
        if not seed_dir.exists():
            print(f"Warning: {seed_dir} does not exist, skipping seed {seed}")
            continue
        
        for config in configs:
            equation = get_equation_name(config)
            model = get_model_name(config)
            
            # Find output directory for this config and seed
            # Config name like "eff_allen_cahn_parametric" -> directory "eff_allen_cahn_parametric"
            config_stem = Path(config).stem
            output_dir = seed_dir / config_stem
            
            # Try to find by pattern if exact match doesn't exist
            if not output_dir.exists():
                pattern = str(seed_dir / f"eff_*{equation.replace(' ', '_').lower()}*")
                matches = glob.glob(pattern)
                if matches:
                    output_dir = Path(matches[0])
                else:
                    # Last resort: try to find any directory matching the equation
                    for subdir in seed_dir.iterdir():
                        if subdir.is_dir() and equation.replace(" ", "_").lower() in subdir.name.lower():
                            output_dir = subdir
                            break
            
            metrics = load_metrics(output_dir)
            if metrics is None:
                continue
            
            metadata = metrics.get("metadata", {})
            
            all_data[equation][model][seed] = {
                "interp_mean": metadata.get("interp_relative_l2_mean"),
                "interp_std": metadata.get("interp_relative_l2_std"),
                "extrap_mean": metadata.get("extrap_relative_l2_mean"),
                "extrap_std": metadata.get("extrap_relative_l2_std"),
                "ms_per_sample": metadata.get("test_ms_per_sample"),
            }
    
    return all_data


def compute_statistics(values):
    """Compute mean and std from a list of values, ignoring None."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None, None
    
    import numpy as np
    mean = np.mean(valid_values)
    std = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
    return float(mean), float(std)


def format_error(mean, std):
    """Format error as mean ± std."""
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def format_time(mean, std):
    """Format time in milliseconds."""
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def generate_aggregated_tables(all_data):
    """Generate aggregated summary tables."""
    tables = []
    
    # Interpolation table
    tables.append("\n" + "="*120)
    tables.append("INTERPOLATION RESULTS (Aggregated across seeds)")
    tables.append("="*120)
    tables.append("")
    
    header = (
        f"{'Equation':<20} {'Model':<10} {'Zero-shot (Rel. L2 Error)':<30} {'ACC Time (ms)':<20} "
        f"{'FT after 50 epochs':<20} {'ACC Time':<12} {'FT after 250 epochs':<20} {'ACC Time':<12} "
        f"{'FT after 500 epochs':<20} {'ACC Time':<12} {'FT after 750 epochs':<20} {'ACC Time':<12} "
        f"{'FT after 1000 epochs':<20} {'ACC Time':<12}\n"
    )
    tables.append(header)
    tables.append("-" * 120)
    
    sorted_equations = sorted(all_data.keys())
    
    for equation in sorted_equations:
        for model in ["U-Net", "FNO"]:
            if model not in all_data[equation]:
                continue
            
            seed_data = all_data[equation][model]
            
            # Aggregate interp errors across seeds
            interp_means = [seed_data[s]["interp_mean"] for s in seed_data.keys()]
            interp_mean, interp_std = compute_statistics(interp_means)
            
            # Aggregate times across seeds
            times = [seed_data[s]["ms_per_sample"] for s in seed_data.keys()]
            time_mean, time_std = compute_statistics(times)
            
            interp_error = format_error(interp_mean, interp_std)
            acc_time = format_time(time_mean, time_std)
            
            row = (
                f"{equation:<20} {model:<10} {interp_error:<30} {acc_time:<20} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12}"
            )
            tables.append(row)
    
    # Extrapolation table
    tables.append("\n" + "="*120)
    tables.append("EXTRAPOLATION RESULTS (Aggregated across seeds)")
    tables.append("="*120)
    tables.append("")
    
    tables.append(header)
    tables.append("-" * 120)
    
    for equation in sorted_equations:
        for model in ["U-Net", "FNO"]:
            if model not in all_data[equation]:
                continue
            
            seed_data = all_data[equation][model]
            
            # Aggregate extrap errors across seeds
            extrap_means = [seed_data[s]["extrap_mean"] for s in seed_data.keys()]
            extrap_mean, extrap_std = compute_statistics(extrap_means)
            
            # Aggregate times across seeds
            times = [seed_data[s]["ms_per_sample"] for s in seed_data.keys()]
            time_mean, time_std = compute_statistics(times)
            
            extrap_error = format_error(extrap_mean, extrap_std)
            acc_time = format_time(time_mean, time_std)
            
            row = (
                f"{equation:<20} {model:<10} {extrap_error:<30} {acc_time:<20} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<12} "
                f"{'N/A':<20} {'N/A':<12}"
            )
            tables.append(row)
    
    return "\n".join(tables)


def main():
    """Main function to aggregate results."""
    parser = argparse.ArgumentParser(description="Aggregate results across multiple seeds")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated list of seeds to aggregate (e.g., '0,1,2')"
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
    
    print(f"Aggregating results for seeds: {seeds}")
    print(f"Found {len(configs)} config files")
    
    # Aggregate results
    all_data = aggregate_results(seeds, configs)
    
    if not all_data:
        print("No results found. Make sure experiments have been run for the specified seeds.")
        sys.exit(1)
    
    # Generate tables
    tables = generate_aggregated_tables(all_data)
    print(tables)
    
    # Save to file
    summary_file = Path("outputs") / "aggregated_summary_tables.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        f.write(tables)
    print(f"\n📊 Aggregated summary tables saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
