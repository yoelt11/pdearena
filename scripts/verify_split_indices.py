#!/usr/bin/env python3
"""Verify split indices match expected values from dataset utilities."""

import yaml
from pathlib import Path
from eff_physics_learn_dataset.datasets import load_pde_dataset

# Expected indices from the user's document
expected_indices = {
    "allen_cahn": {
        0: {
            "train": [159, 160, 121, 99, 54, 9, 3, 61, 38, 16],
            "interp": [22, 45, 97, 186, 130, 119, 169, 120, 104, 78, 17, 4, 66, 35, 105, 189, 11, 192, 47, 108],
            "extrap": [196, 96, 15, 153, 28, 94, 41, 187, 164, 33, 84, 180, 193, 23, 143, 151, 166, 191, 65, 147],
        },
        1: {
            "train": [64, 88, 160, 8, 185, 99, 31, 144, 182, 52],
            "interp": [135, 149, 116, 98, 188, 86, 174, 53, 39, 47, 25, 113, 16, 119, 130, 173, 192, 56, 97, 14],
            "extrap": [159, 141, 175, 80, 148, 126, 161, 140, 94, 112, 183, 102, 196, 82, 172, 46, 15, 129, 6, 9],
        },
        2: {
            "train": [119, 59, 80, 24, 157, 193, 52, 21, 87, 68],
            "interp": [131, 5, 20, 48, 73, 14, 95, 57, 81, 56, 136, 155, 18, 67, 41, 167, 192, 105, 36, 158],
            "extrap": [177, 147, 178, 182, 151, 31, 107, 166, 32, 113, 134, 108, 180, 174, 160, 111, 26, 93, 112, 184],
        },
    },
    "burgers": {
        0: {
            "train": [160, 161, 118, 95, 50, 7, 2, 57, 34, 13],
            "interp": [22, 37, 97, 153, 90, 105, 128, 137, 124, 106, 96, 0, 79, 32, 43, 169, 3, 185, 47, 81],
            "extrap": [142, 102, 11, 171, 18, 100, 31, 10, 177, 26, 73, 178, 146, 12, 150, 188, 183, 17, 56, 157],
        },
        1: {
            "train": [60, 87, 160, 6, 188, 95, 27, 143, 185, 48],
            "interp": [166, 137, 198, 97, 169, 56, 158, 164, 133, 22, 12, 96, 3, 172, 138, 140, 8, 43, 88, 116],
            "extrap": [167, 148, 181, 80, 152, 123, 165, 144, 98, 110, 122, 102, 196, 85, 179, 49, 18, 132, 11, 15],
        },
        2: {
            "train": [117, 55, 78, 18, 158, 195, 48, 16, 86, 63],
            "interp": [8, 15, 21, 32, 39, 41, 96, 105, 124, 128, 138, 140, 181, 185, 187, 196],
            "extrap": [5, 93, 42, 66, 102, 156, 24, 177, 20, 126, 70, 81, 26, 180, 60, 77],
        },
    },
    "convection": {
        0: {
            "train": [160, 161, 118, 95, 50, 7, 2, 57, 34, 13],
            "interp": [36, 59, 104, 193, 143, 123, 172, 126, 106, 92, 28, 6, 78, 47, 112, 134, 18, 58, 61, 114],
            "extrap": [190, 163, 4, 108, 5, 85, 11, 173, 149, 128, 187, 142, 136, 169, 122, 139, 131, 175, 17, 99],
        },
        1: {
            "train": [60, 87, 160, 6, 188, 95, 27, 143, 185, 48],
            "interp": [140, 162, 119, 102, 171, 96, 182, 62, 47, 61, 33, 128, 21, 123, 198, 29, 196, 69, 98, 19],
            "extrap": [151, 124, 177, 82, 136, 131, 184, 142, 79, 109, 135, 100, 197, 63, 178, 31, 7, 104, 5, 167],
        },
        2: {
            "train": [117, 55, 78, 18, 158, 195, 48, 16, 86, 63],
            "interp": [134, 6, 24, 53, 71, 15, 91, 61, 79, 57, 141, 156, 20, 66, 39, 171, 196, 111, 37, 164],
            "extrap": [175, 149, 100, 178, 160, 28, 187, 163, 31, 114, 136, 108, 109, 174, 197, 166, 23, 99, 115, 181],
        },
    },
    "helmholtz2D": {
        0: {
            "train": [162, 163, 123, 99, 50, 8, 3, 57, 33, 14],
            "interp": [39, 40, 78, 178, 110, 136, 166, 172, 149, 127, 5, 199, 182, 173, 43, 183, 0, 82, 196, 97],
            "extrap": [134, 96, 10, 161, 18, 94, 29, 9, 171, 25, 71, 174, 139, 11, 143, 186, 181, 16, 54, 151],
        },
        1: {
            "train": [61, 87, 162, 7, 188, 99, 27, 146, 184, 48],
            "interp": [176, 119, 199, 114, 173, 78, 166, 70, 40, 39, 22, 102, 5, 136, 196, 149, 183, 53, 97, 82],
            "extrap": [163, 145, 179, 76, 151, 123, 159, 140, 94, 109, 122, 101, 195, 85, 174, 46, 16, 132, 10, 12],
        },
        2: {
            "train": [122, 55, 78, 20, 160, 195, 48, 17, 86, 64],
            "interp": [196, 5, 9, 166, 167, 142, 82, 97, 40, 155, 31, 178, 149, 53, 173, 148, 177, 16, 146, 96],
            "extrap": [171, 144, 99, 32, 198, 33, 192, 175, 36, 118, 125, 105, 187, 172, 153, 112, 27, 103, 108, 191],
        },
    },
}

def get_indices_from_splits(splits):
    """Extract indices from split objects."""
    train_indices = [sample.get('index', i) for i, sample in enumerate(splits['train_few'])]
    interp_indices = [sample.get('index', i) for i, sample in enumerate(splits['interp'])]
    extrap_indices = [sample.get('index', i) for i, sample in enumerate(splits['extrap'])]
    return train_indices, interp_indices, extrap_indices

def verify_splits(equation, seed, data_dir="datasets"):
    """Verify splits match expected values."""
    print(f"\n{'='*80}")
    print(f"Verifying {equation} with seed {seed}")
    print(f"{'='*80}")
    
    # Load dataset
    dataset = load_pde_dataset(equation, data_dir=data_dir, cache=True)
    
    # Generate splits
    splits = dataset.parametric_splits(
        seed=seed,
        n_train=10,
        balance=True,
        n_each=20,
    )
    
    # Extract indices
    train_indices, interp_indices, extrap_indices = get_indices_from_splits(splits)
    
    # Compare with expected
    expected = expected_indices[equation][seed]
    
    train_match = train_indices == expected["train"]
    interp_match = interp_indices == expected["interp"]
    extrap_match = extrap_indices == expected["extrap"]
    
    print(f"\nTrain indices: {train_indices}")
    print(f"Expected:      {expected['train']}")
    print(f"Match: {train_match}")
    
    print(f"\nInterp indices: {interp_indices}")
    print(f"Expected:       {expected['interp']}")
    print(f"Match: {interp_match}")
    
    print(f"\nExtrap indices: {extrap_indices}")
    print(f"Expected:       {expected['extrap']}")
    print(f"Match: {extrap_match}")
    
    all_match = train_match and interp_match and extrap_match
    print(f"\n{'✓ ALL MATCH' if all_match else '✗ MISMATCH'}")
    
    return all_match

def main():
    """Verify all splits."""
    equations = ["allen_cahn", "burgers", "convection", "helmholtz2D"]
    seeds = [0, 1, 2]
    
    all_passed = True
    for equation in equations:
        for seed in seeds:
            passed = verify_splits(equation, seed)
            if not passed:
                all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("✗ SOME VERIFICATIONS FAILED")
    print("="*80)

if __name__ == "__main__":
    main()
