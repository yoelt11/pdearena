# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule

from pdearena import utils
from pdearena.modules.conditioned.twod_resnet import FourierBasicBlock, ResNet


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute mean relative L2 over batch.

    Args:
        pred/target: (B, 1, 1, H, W)
    """

    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1)
    targ = target.reshape(b, -1)
    num = torch.linalg.vector_norm(diff, ord=2, dim=1)
    den = torch.linalg.vector_norm(targ, ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()


class EffParametricFNO(LightningModule):
    """Parameter-conditioned FNO (Fourier Neural Operator) for eff-physics-learn-dataset parametric modality."""

    def __init__(
        self,
        *,
        n_input_channels: int = 2,  # e.g. X_grid + T_grid
        n_output_channels: int = 1,  # u
        n_params: int = 3,  # Number of PDE parameters
        hidden_channels: int = 64,
        modes1: int = 16,  # Fourier modes in first spatial dimension
        modes2: int = 16,  # Fourier modes in second spatial dimension
        num_blocks: list = [1, 1, 1, 1],  # ResNet block configuration
        activation: str = "gelu",
        norm: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # If provided, will de-normalize predictions for metric computation / export.
        # (May also be set later via `set_u_normalization()`.)
        u_mean: Optional[float] = None,
        u_std: Optional[float] = None,
        metrics_out_path: Optional[str] = None,
        # Optional: write a qualitative figure during `test`.
        # The figure will have columns: GT, Pred, |Error| and one row per pde_param case.
        plot_out_path: Optional[str] = None,
        plot_max_rows: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Create FNO block class with specified modes using partialclass
        # This creates a class with modes1 and modes2 pre-set
        fno_block_class = utils.partialclass(
            "CustomFourierBasicBlock", FourierBasicBlock, modes1=modes1, modes2=modes2
        )

        # Use ResNet with FourierBasicBlock for FNO architecture
        # Use "scalar_N" format for all cases (consistent with our implementation)
        self.model = ResNet(
            n_input_scalar_components=int(n_input_channels),
            n_input_vector_components=0,
            n_output_scalar_components=int(n_output_channels),
            n_output_vector_components=0,
            block=fno_block_class,
            num_blocks=num_blocks,
            time_history=1,
            time_future=1,
            hidden_channels=int(hidden_channels),
            activation=str(activation),
            norm=bool(norm),
            diffmode=False,
            usegrid=False,
            param_conditioning=f"scalar_{int(n_params)}",
        )

        # Use default PyTorch initialization (same as original FNO implementation)
        # No special initialization needed - PyTorch's default Kaiming/He init works fine
        self.criterion = torch.nn.MSELoss()

        # running aggregates per split during test
        self._test_sum: Dict[str, torch.Tensor] = {}
        self._test_count: Dict[str, torch.Tensor] = {}

        # normalization (can be updated after DM setup)
        self._u_mean: Optional[float] = float(u_mean) if u_mean is not None else None
        self._u_std: Optional[float] = float(u_std) if u_std is not None else None

        # plotting cache (rank 0 only)
        self._plot_examples: Dict[str, list] = {}
        # per-solution metrics cache (rank 0 only)
        self._per_solution: Dict[str, list] = {}
        # inference timing (rank 0 only)
        self._test_t0: Optional[float] = None
        self._test_num_batches: int = 0
        self._test_num_samples: int = 0
        self._test_num_samples_by_split: Dict[str, int] = {"interp": 0, "extrap": 0}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.model(x, time=t, z=z)

    def _maybe_denorm(self, y: torch.Tensor) -> torch.Tensor:
        if self._u_mean is None or self._u_std is None:
            return y
        return y * float(self._u_std) + float(self._u_mean)

    def set_u_normalization(self, *, mean: Optional[float], std: Optional[float]) -> None:
        """Set (or clear) target normalization used for metric computation/export."""
        self._u_mean = None if mean is None else float(mean)
        self._u_std = None if std is None else float(std)

    def _step(self, batch: Any) -> Dict[str, torch.Tensor]:
        x, y, t, z = batch
        pred = self(x, t, z)
        loss = self.criterion(pred, y)
        return {"loss": loss, "pred": pred, "y": y}

    def training_step(self, batch: Any, batch_idx: int):
        out = self._step(batch)
        self.log("train/loss", out["loss"], prog_bar=True)
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        out = self._step(batch)
        pred_dn = self._maybe_denorm(out["pred"])
        y_dn = self._maybe_denorm(out["y"])
        rel = relative_l2(pred_dn, y_dn)
        self.log("valid/loss", out["loss"], prog_bar=True)
        self.log("valid/relative_l2", rel, prog_bar=True)
        return {"val_loss": out["loss"], "val_rel_l2": rel}

    def on_test_start(self) -> None:
        self._test_sum = {}
        self._test_count = {}
        self._plot_examples = {"interp": [], "extrap": []}
        self._per_solution = {"interp": [], "extrap": []}
        self._test_t0 = time.perf_counter()
        self._test_num_batches = 0
        self._test_num_samples = 0
        self._test_num_samples_by_split = {"interp": 0, "extrap": 0}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        split = "interp" if dataloader_idx == 0 else "extrap"
        out = self._step(batch)
        pred_dn = self._maybe_denorm(out["pred"])
        y_dn = self._maybe_denorm(out["y"])
        rel = relative_l2(pred_dn, y_dn)

        # Inference throughput accounting (batches/samples)
        bsz = int(pred_dn.shape[0])
        self._test_num_batches += 1
        self._test_num_samples += bsz
        self._test_num_samples_by_split[split] = self._test_num_samples_by_split.get(split, 0) + bsz

        # Per-solution RL2 (for export). Rank 0 only; for multi-GPU we'd need a gather.
        if self.trainer.is_global_zero:
            x, y, t, z = batch
            b = pred_dn.shape[0]
            diff = (pred_dn - y_dn).reshape(b, -1)
            targ = y_dn.reshape(b, -1)
            num = torch.linalg.vector_norm(diff, ord=2, dim=1)
            den = torch.linalg.vector_norm(targ, ord=2, dim=1).clamp_min(1e-12)
            rel_vec = (num / den).detach().float().cpu().numpy().reshape(-1).tolist()
            z_vec = z.detach().float().cpu().numpy()
            for i, rl2_i in enumerate(rel_vec):
                self._per_solution[split].append(
                    {
                        "z": z_vec[i].reshape(-1).tolist(),
                        "rl2": float(rl2_i),
                    }
                )

        # Optionally stash a few examples for a qualitative plot (rank 0 only).
        if self.hparams.plot_out_path and self.trainer.is_global_zero:
            remaining = int(self.hparams.plot_max_rows) - len(self._plot_examples[split])
            if remaining > 0:
                x, y, t, z = batch
                b = y_dn.shape[0]
                take = min(remaining, b)
                for i in range(take):
                    self._plot_examples[split].append(
                        {
                            "z": z[i].detach().float().cpu().numpy().reshape(-1).tolist(),
                            "gt": y_dn[i, 0, 0].detach().float().cpu().numpy(),
                            "pred": pred_dn[i, 0, 0].detach().float().cpu().numpy(),
                        }
                    )

        # keep synced aggregates
        self._test_sum.setdefault(split, torch.tensor(0.0, device=self.device))
        self._test_count.setdefault(split, torch.tensor(0.0, device=self.device))
        self._test_sum[split] = self._test_sum[split] + rel.detach()
        self._test_count[split] = self._test_count[split] + torch.tensor(1.0, device=self.device)

        self.log(f"test/{split}/relative_l2", rel, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {f"{split}_rel_l2": rel}

    def on_test_end(self) -> None:
        # Aggregate split metrics.
        results: Dict[str, float] = {}
        if self.trainer.is_global_zero:
            import numpy as np

            for split in ("interp", "extrap"):
                vals = [d["rl2"] for d in self._per_solution.get(split, []) if "rl2" in d]
                if vals:
                    arr = np.asarray(vals, dtype=np.float64)
                    results[f"{split}_relative_l2_mean"] = float(arr.mean())
                    results[f"{split}_relative_l2_std"] = float(arr.std(ddof=0))
                    # Back-compat key (previously stored mean here)
                    results[f"{split}_relative_l2"] = float(arr.mean())

            # Inference speed (wall time + throughput)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if self._test_t0 is not None:
                wall = max(0.0, float(t1 - self._test_t0))
                results["test_wall_time_sec"] = wall
                if wall > 0:
                    results["test_batches_per_sec"] = float(self._test_num_batches) / wall
                    results["test_samples_per_sec"] = float(self._test_num_samples) / wall
                    # per split throughput (samples/sec) using the same wall clock
                    results["interp_samples_per_sec"] = float(self._test_num_samples_by_split.get("interp", 0)) / wall
                    results["extrap_samples_per_sec"] = float(self._test_num_samples_by_split.get("extrap", 0)) / wall
                    results["test_ms_per_sample"] = 1e3 * wall / max(1, int(self._test_num_samples))
        else:
            # Fallback (shouldn't happen because only global zero exports)
            for split in ("interp", "extrap"):
                if split in self._test_sum and split in self._test_count:
                    mean = (self._test_sum[split] / self._test_count[split].clamp_min(1.0)).detach().cpu().item()
                    results[f"{split}_relative_l2"] = float(mean)

        # Optional qualitative figure
        if self.hparams.plot_out_path and self.trainer.is_global_zero:
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            def _plot_rows(rows, out_path: Path, title_prefix: str):
                if not rows:
                    return
                out_path.parent.mkdir(parents=True, exist_ok=True)

                nrows = len(rows)
                fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(10, max(2.5, 2.2 * nrows)))
                if nrows == 1:
                    axes = np.asarray([axes])

                # Global scales across rows in this split
                gt_pred_min = float("inf")
                gt_pred_max = float("-inf")
                err_max = float("-inf")
                for ex in rows:
                    gt = np.asarray(ex["gt"])
                    pred = np.asarray(ex["pred"])
                    # FNO is 2D only, so no need to handle 3D
                    if gt.ndim != 2:
                        continue  # Skip unsupported dimensions
                    e = np.abs(pred - gt)
                    gt_pred_min = float(min(gt_pred_min, float(np.nanmin(gt)), float(np.nanmin(pred))))
                    gt_pred_max = float(max(gt_pred_max, float(np.nanmax(gt)), float(np.nanmax(pred))))
                    err_max = float(max(err_max, float(np.nanmax(e))))

                levels_main = np.linspace(gt_pred_min, gt_pred_max, 21)
                levels_err = np.linspace(0.0, err_max, 21) if err_max > 0 else 21

                main_mappable = None
                err_mappable = None

                for r, ex in enumerate(rows):
                    gt = np.asarray(ex["gt"])
                    pred = np.asarray(ex["pred"])
                    z = ex["z"]

                    # FNO is 2D only
                    if gt.ndim != 2:
                        # Skip plotting for unsupported dimensions
                        continue
                    
                    err = np.abs(pred - gt)

                    ax0, ax1, ax2 = axes[r]
                    c0 = ax0.contourf(gt, levels=levels_main, cmap="viridis")
                    ax0.contour(gt, levels=levels_main, colors="k", linewidths=0.3, alpha=0.4)

                    ax1.contourf(pred, levels=levels_main, cmap="viridis")
                    ax1.contour(pred, levels=levels_main, colors="k", linewidths=0.3, alpha=0.4)

                    c2 = ax2.contourf(err, levels=levels_err, cmap="magma")
                    ax2.contour(err, levels=levels_err, colors="k", linewidths=0.3, alpha=0.4)

                    if main_mappable is None:
                        main_mappable = c0
                    if err_mappable is None:
                        err_mappable = c2

                    ax0.set_title("GT")
                    ax1.set_title("Pred")
                    ax2.set_title("|Error|")

                    for ax in (ax0, ax1, ax2):
                        ax.set_xticks([])
                        ax.set_yticks([])

                    z_str = "[" + ", ".join(f"{float(v):.4g}" for v in z) + "]"
                    ax0.set_ylabel(f"{title_prefix}\nz={z_str}", rotation=0, labelpad=45, va="center", fontsize=8)

                # Reserve space on the right for colorbars.
                fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
                if main_mappable is not None:
                    cax_main = fig.add_axes([0.88, 0.55, 0.02, 0.35])
                    fig.colorbar(main_mappable, cax=cax_main, label="u (GT/Pred)")
                if err_mappable is not None:
                    cax_err = fig.add_axes([0.88, 0.10, 0.02, 0.35])
                    fig.colorbar(err_mappable, cax=cax_err, label="|Error|")

                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

            base_path = Path(str(self.hparams.plot_out_path))
            # Write two separate figures instead of mixing splits in one plot.
            if base_path.suffix.lower() == ".png":
                interp_path = base_path.with_name(f"{base_path.stem}_interp{base_path.suffix}")
                extrap_path = base_path.with_name(f"{base_path.stem}_extrap{base_path.suffix}")
            else:
                interp_path = Path(str(base_path) + "_interp.png")
                extrap_path = Path(str(base_path) + "_extrap.png")

            _plot_rows(self._plot_examples.get("interp", []), interp_path, "interp")
            _plot_rows(self._plot_examples.get("extrap", []), extrap_path, "extrap")

        # Optional metrics-structures export handled in a separate module to keep this class minimal.
        if self.hparams.metrics_out_path:
            from pdearena.metrics_export import export_eff_parametric_metrics

            export_eff_parametric_metrics(
                out_path=str(self.hparams.metrics_out_path),
                results=results,
                run_config=dict(self.hparams),
                per_solution={"interp": self._per_solution.get("interp", []), "extrap": self._per_solution.get("extrap", [])},
            )


