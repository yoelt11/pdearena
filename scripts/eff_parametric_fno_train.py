# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path

import yaml
from pytorch_lightning.cli import LightningCLI

from pdearena.data.eff_physics_datamodule import EffPhysicsParametricDataModule
from pdearena.models.eff_parametric_fno import EffParametricFNO


class EffParametricFNOCLI(LightningCLI):
    """LightningCLI with a small hook to pass dataset normalization into the model."""

    def _sync_norm(self) -> None:
        dm = getattr(self, "datamodule", None)
        model = getattr(self, "model", None)
        if dm is None or model is None:
            return
        norm = getattr(dm, "_norm", None)
        if norm is not None and hasattr(model, "set_u_normalization"):
            model.set_u_normalization(mean=norm.u_mean, std=norm.u_std)

    def before_fit(self) -> None:
        # Ensure datamodule has computed normalization
        if hasattr(self, "datamodule") and self.datamodule is not None:
            self.datamodule.setup("fit")
        self._sync_norm()

    def before_test(self) -> None:
        if hasattr(self, "datamodule") and self.datamodule is not None:
            self.datamodule.setup("test")
        self._sync_norm()
        self._save_split_indices()
    
    def _save_split_indices(self) -> None:
        """Save train, interp, and extrap indices to a YAML file for reference."""
        if not hasattr(self, "datamodule") or self.datamodule is None:
            return
        
        indices = self.datamodule.get_split_indices()
        if not indices:
            return
        
        # Get output directory from trainer config
        output_dir = None
        if hasattr(self, "trainer") and self.trainer is not None:
            output_dir = Path(self.trainer.default_root_dir)
        elif hasattr(self, "config") and self.config is not None:
            # Try to get from config
            trainer_cfg = self.config.get("trainer", {})
            output_dir = Path(trainer_cfg.get("default_root_dir", "outputs"))
        
        if output_dir is None:
            output_dir = Path("outputs")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        indices_file = output_dir / "split_indices.yaml"
        
        # Prepare YAML content with metadata
        yaml_content = {
            "equation": self.datamodule.hparams.equation,
            "seed": int(self.datamodule.hparams.seed),
            "n_train": int(self.datamodule.hparams.n_train),
            "balance": bool(self.datamodule.hparams.balance) if hasattr(self.datamodule.hparams, 'balance') else True,
            "n_each": int(self.datamodule.hparams.n_each) if hasattr(self.datamodule.hparams, 'n_each') else 20,
            "balance_strategy": str(self.datamodule.hparams.balance_strategy) if hasattr(self.datamodule.hparams, 'balance_strategy') else 'random',
            "split_sizes": {
                "train": len(indices.get("train", [])),
                "interp": len(indices.get("interp", [])),
                "extrap": len(indices.get("extrap", [])),
            },
            "indices": indices,
        }
        
        with open(indices_file, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"Saved split indices to: {indices_file}")


def main():
    # Optional: user can point eff dataset download utilities at a custom links file.
    # (Used only if they call download outside this script.)
    if "EFF_PHYSICS_LEARN_DATASET_LINKS" not in os.environ:
        # Allow local override for convenience if a file exists in repo root.
        if os.path.exists("eff_dataset_links.toml"):
            os.environ["EFF_PHYSICS_LEARN_DATASET_LINKS"] = "eff_dataset_links.toml"

    EffParametricFNOCLI(
        EffParametricFNO,
        EffPhysicsParametricDataModule,
        seed_everything_default=0,
        run=True,
    )


if __name__ == "__main__":
    main()



