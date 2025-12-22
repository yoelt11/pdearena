# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from pytorch_lightning.cli import LightningCLI

from pdearena.models.eff_parametric_unet import EffParametricUNet
from pdearena.data.eff_physics_datamodule import EffPhysicsParametricDataModule


class EffParametricCLI(LightningCLI):
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


def main():
    # Optional: user can point eff dataset download utilities at a custom links file.
    # (Used only if they call download outside this script.)
    if "EFF_PHYSICS_LEARN_DATASET_LINKS" not in os.environ:
        # Allow local override for convenience if a file exists in repo root.
        if os.path.exists("eff_dataset_links.toml"):
            os.environ["EFF_PHYSICS_LEARN_DATASET_LINKS"] = "eff_dataset_links.toml"

    EffParametricCLI(
        EffParametricUNet,
        EffPhysicsParametricDataModule,
        seed_everything_default=0,
        run=True,
    )


if __name__ == "__main__":
    main()


