# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .eff_physics_dataset import EffPhysicsNormalization, EffPhysicsParametricDataset


@dataclass(frozen=True)
class EffParametricSplitSizes:
    train_few: int
    interp: int
    extrap: int


class EffPhysicsParametricDataModule(LightningDataModule):
    """LightningDataModule for eff-physics-learn-dataset parametric splits.

    Produces batches compatible with a conditioned U-Net style LightningModule:
      - train: (x, y, t, z)
      - test:  [interp_loader, extrap_loader]

    Each tensor already includes a leading sample-batch dimension so our existing
    `collate_fn_cat` can concatenate along dim=0.
    """

    def __init__(
        self,
        *,
        equation: str = "burgers",
        data_dir: str = "datasets",
        seed: int = 0,
        n_train: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        include_grids: bool = True,
        standardize_u: bool = True,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._norm: Optional[EffPhysicsNormalization] = None
        self.split_sizes: Optional[EffParametricSplitSizes] = None

        self.train_ds = None
        self.interp_ds = None
        self.extrap_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        from eff_physics_learn_dataset.datasets import load_pde_dataset

        ds = load_pde_dataset(self.hparams.equation, data_dir=self.hparams.data_dir, cache=self.hparams.cache)
        splits = ds.parametric_splits(seed=int(self.hparams.seed), n_train=int(self.hparams.n_train))

        train_few = splits["train_few"]
        interp = splits["interp"]
        extrap = splits["extrap"]

        self.split_sizes = EffParametricSplitSizes(train_few=len(train_few), interp=len(interp), extrap=len(extrap))

        # Compute normalization from train_few solutions only.
        if bool(self.hparams.standardize_u):
            idx = np.asarray(train_few.indices, dtype=np.int64)
            u = np.asarray(ds.u[idx], dtype=np.float32)
            u_mean = float(u.mean())
            u_std = float(u.std(ddof=0))
            self._norm = EffPhysicsNormalization(u_mean=u_mean, u_std=u_std)
        else:
            self._norm = None

        self.train_ds = EffPhysicsParametricDataset(
            train_few,
            normalization=self._norm,
            include_grids=bool(self.hparams.include_grids),
        )
        self.interp_ds = EffPhysicsParametricDataset(
            interp,
            normalization=self._norm,
            include_grids=bool(self.hparams.include_grids),
        )
        self.extrap_ds = EffPhysicsParametricDataset(
            extrap,
            normalization=self._norm,
            include_grids=bool(self.hparams.include_grids),
        )

    @staticmethod
    def collate_fn_cat(batch):
        # Same semantics as pdearena.data.datamodule.collate_fn_cat, but duplicated here
        # to avoid importing torchdata-dependent PDEArena datapipes when using a newer torchdata.
        b1 = torch.cat([b[0] for b in batch], dim=0)
        b2 = torch.cat([b[1] for b in batch], dim=0)
        if len(batch[0]) > 2:
            b3 = torch.cat([b[2] for b in batch], dim=0)
            b4 = torch.cat([b[3] for b in batch], dim=0)
            return b1, b2, b3, b4
        return b1, b2

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=int(self.hparams.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            collate_fn=self.collate_fn_cat,
        )

    def test_dataloader(self):
        interp_loader = DataLoader(
            dataset=self.interp_ds,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            collate_fn=self.collate_fn_cat,
        )
        extrap_loader = DataLoader(
            dataset=self.extrap_ds,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            collate_fn=self.collate_fn_cat,
        )
        return [interp_loader, extrap_loader]

    def val_dataloader(self):
        # Keep validation simple: use interp split as val.
        return DataLoader(
            dataset=self.interp_ds,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            collate_fn=self.collate_fn_cat,
        )


