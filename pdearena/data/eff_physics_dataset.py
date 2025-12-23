# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EffPhysicsNormalization:
    """Simple per-dataset normalization config."""

    u_mean: float
    u_std: float


def _to_f32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _stack_grids(sample: dict, spatial_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Build input channels from available coordinate grids.

    Supports 2D and 3D problems:
      - 2D: X_grid (H, W), T_grid (H, W) or T_grid (T,) -> expanded to (H, W)
      - 3D: X_grid (H, W, D) or (H, W) -> expanded to (H, W, D), 
            Y_grid (H, W, D) or (H, W) -> expanded to (H, W, D),
            Z_grid (H, W, D) or (H, W) -> expanded to (H, W, D),
            T_grid (T,) -> expanded to (H, W, D)

    If grids are not available in the sample, they will be constructed as normalized
    coordinate grids based on the spatial shape.

    Args:
        sample: Dictionary containing grid data
        spatial_shape: Optional spatial shape (H, W) or (H, W, D) to expand grids.
                      If None, inferred from first non-1D grid.

    Returns:
      np.ndarray with shape (C, H, W) for 2D or (C, H, W, D) for 3D
    """

    grids = []
    reference_shape = None
    
    # First pass: collect grids and find reference spatial shape
    grid_data = {}
    for k in ("X_grid", "Y_grid", "Z_grid", "T_grid"):
        if k in sample:
            g = _to_f32(sample[k])
            grid_data[k] = g
            # Use first non-1D grid as reference for spatial shape
            if reference_shape is None and g.ndim > 1:
                reference_shape = g.shape
    
    # Use provided spatial_shape if available, otherwise use reference_shape
    if spatial_shape is not None:
        target_spatial_shape = spatial_shape
    elif reference_shape is not None:
        target_spatial_shape = reference_shape
    else:
        # Fallback: if all grids are 1D, we can't determine spatial shape
        # This shouldn't happen in practice, but handle gracefully
        raise ValueError("Cannot determine spatial shape from grids. All grids appear to be 1D.")
    
    # Second pass: process grids and expand 1D grids to match spatial dimensions
    for k in ("X_grid", "Y_grid", "Z_grid", "T_grid"):
        if k in grid_data:
            g = grid_data[k]
            if g.ndim == 1:
                # Expand 1D grid (e.g., T_grid) to match spatial dimensions
                # For time grids, we typically want to broadcast a single value or use the first value
                # If T_grid has multiple values, we'll use the first one and broadcast it
                if g.size == 1:
                    # Single scalar: broadcast to spatial shape
                    g_value = g[0]
                else:
                    # Multiple values: use first value (or could use mean, but first is simpler)
                    # This handles cases where T_grid is (64,) but we need a single time value
                    g_value = g[0]
                
                # Broadcast to target spatial shape
                if len(target_spatial_shape) == 2:
                    g_expanded = np.full(target_spatial_shape, g_value, dtype=g.dtype)
                elif len(target_spatial_shape) == 3:
                    g_expanded = np.full(target_spatial_shape, g_value, dtype=g.dtype)
                else:
                    raise ValueError(f"Unsupported spatial dimension: {len(target_spatial_shape)}D")
                grids.append(g_expanded)
            elif g.ndim == 2:
                # 2D grid
                if len(target_spatial_shape) == 2:
                    # 2D problem: use as is if shapes match
                    if g.shape == target_spatial_shape:
                        grids.append(g)
                    else:
                        raise ValueError(f"2D grid {k} shape {g.shape} incompatible with spatial shape {target_spatial_shape}")
                elif len(target_spatial_shape) == 3:
                    # 3D problem but 2D grid: expand by broadcasting along the third dimension
                    # Check if first two dimensions match
                    if g.shape == target_spatial_shape[:2]:
                        # Broadcast (H, W) -> (H, W, D) by adding a new axis and broadcasting
                        g_expanded = np.broadcast_to(g[:, :, np.newaxis], target_spatial_shape).copy()
                        grids.append(g_expanded)
                    else:
                        raise ValueError(f"2D grid {k} shape {g.shape} incompatible with spatial shape {target_spatial_shape} (first 2 dims must match)")
                else:
                    raise ValueError(f"2D grid {k} incompatible with {len(target_spatial_shape)}D spatial shape")
            elif g.ndim == 3:
                # 3D grid - use as is
                if len(target_spatial_shape) == 3 and g.shape == target_spatial_shape:
                    grids.append(g)
                else:
                    raise ValueError(f"3D grid {k} shape {g.shape} incompatible with spatial shape {target_spatial_shape}")
            else:
                raise ValueError(f"Unsupported grid dimension for {k}: {g.ndim}D (shape={g.shape})")
    
    # If no grids were found, construct normalized coordinate grids
    if not grids:
        if spatial_shape is None:
            raise ValueError("Cannot construct grids: spatial_shape is required when grids are not in sample.")
        
        # Construct normalized coordinate grids
        # Use meshgrid to create proper 2D coordinate grids
        if len(spatial_shape) == 2:
            # 2D: create X and T grids
            H, W = spatial_shape
            # Create meshgrid: X varies along W (columns), T varies along H (rows)
            x_coords = np.linspace(0, 1, W, dtype=np.float32)
            t_coords = np.linspace(0, 1, H, dtype=np.float32)
            X_grid, T_grid = np.meshgrid(x_coords, t_coords, indexing='xy')
            grids = [X_grid.astype(np.float32), T_grid.astype(np.float32)]
        elif len(spatial_shape) == 3:
            # 3D: create X, Y, Z grids
            H, W, D = spatial_shape
            x_coords = np.linspace(0, 1, W, dtype=np.float32)
            y_coords = np.linspace(0, 1, H, dtype=np.float32)
            z_coords = np.linspace(0, 1, D, dtype=np.float32)
            X_grid = np.broadcast_to(x_coords[np.newaxis, :, np.newaxis], (H, W, D))
            Y_grid = np.broadcast_to(y_coords[:, np.newaxis, np.newaxis], (H, W, D))
            Z_grid = np.broadcast_to(z_coords[np.newaxis, np.newaxis, :], (H, W, D))
            grids = [X_grid, Y_grid, Z_grid]
        else:
            raise ValueError(f"Cannot construct grids for {len(spatial_shape)}D spatial shape")
    
    return np.stack(grids, axis=0)


class EffPhysicsParametricDataset(Dataset):
    """Torch Dataset wrapper around an eff-physics-learn-dataset PDEDataset split."""

    def __init__(
        self,
        split_ds,
        *,
        normalization: Optional[EffPhysicsNormalization] = None,
        include_grids: bool = True,
    ) -> None:
        super().__init__()
        self.split_ds = split_ds
        self.normalization = normalization
        self.include_grids = include_grids

    def __len__(self) -> int:
        return len(self.split_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.split_ds[int(idx)]

        # Conditioning parameters
        z_np = _to_f32(s["params"])  # (P,)
        z = torch.from_numpy(z_np).unsqueeze(0)  # (1, P) so collate can cat on dim=0

        # Targets: solution field (used to determine spatial shape)
        u_np = _to_f32(s["u"])
        
        # Determine spatial dimensions from solution field
        if u_np.ndim == 2:
            # 2D problem: (H, W)
            spatial_shape = u_np.shape
            spatial_dims = 2
        elif u_np.ndim == 3:
            # 3D problem: (H, W, D)
            spatial_shape = u_np.shape
            spatial_dims = 3
        else:
            raise ValueError(f"Expected u to be 2D or 3D, got shape={u_np.shape}")

        # Inputs: coordinate grids only (acts like a conditional generator)
        if self.include_grids:
            x_np = _stack_grids(s, spatial_shape=spatial_shape)  # (C, H, W) or (C, H, W, D)
        else:
            x_np = np.zeros((1, *u_np.shape), dtype=np.float32)

        # Normalize solution field
        if self.normalization is not None:
            u_np = (u_np - self.normalization.u_mean) / max(self.normalization.u_std, 1e-12)

        # Shape to match conditioned UNet: (B=1, T=1, C, H, W) for 2D or (B=1, T=1, C, H, W, D) for 3D
        x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W) or (1, 1, C, H, W, D)
        if spatial_dims == 2:
            y = torch.from_numpy(u_np).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
        else:  # spatial_dims == 3
            y = torch.from_numpy(u_np).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W, D)

        # Dummy "time" conditioning expected by conditioned UNet API (shape: (B,))
        t = torch.zeros((1,), dtype=torch.float32)

        return x, y, t, z


