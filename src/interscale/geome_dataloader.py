from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal

import lightning.pytorch as pl
import torch
from torch_geometric.loader import DataLoader
from yacs.config import CfgNode as CN

from interscale.data import GraphDataset, MaskedGraphDataset
from interscale.geome.datamodule import _Prefetch


class GraphAnnDataModule(pl.LightningDataModule):
    """Lightning DataModule over InterScale contract datasets, masked by :class:`~interscale.data.MaskedGraphDataset`.

    Parameters
    ----------
    datas
        ``[train, val]`` or ``[train, val, test]`` :class:`~interscale.data.GraphDataset` s.
    batch_size
        Graphs per batch.
    num_workers
        DataLoader worker processes, re-created every epoch (see ``MaskedGraphDataset``). 0, the
        default, loads batches on a prefetch thread instead, which measured faster on both paths.
    mask_percentage, mask_strategy, seed
        Passed to :class:`~interscale.data.MaskedGraphDataset`.
    learning_type
        ``"node"`` masks every split, since the masked nodes are the targets. ``"graph"`` masks
        only train, where masking is input augmentation.
    """

    def __init__(
        self,
        datas: Sequence[GraphDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        mask_percentage: float = 0.5,
        mask_strategy: Literal["node", "gene"] = "node",
        learning_type: Literal["node", "graph"] = "node",
        seed: int = 0,
    ):
        super().__init__()
        if datas is None or len(datas) not in (2, 3):
            raise ValueError("datas must hold a train and a validation set, and optionally a test set.")
        if learning_type not in ("node", "graph"):
            raise ValueError(f"learning_type must be 'node' or 'graph', got {learning_type!r}.")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data, self.val_data, *rest = datas
        self.test_data = rest[0] if rest else None
        masked = partial(MaskedGraphDataset, mask_strategy=mask_strategy, mask_percentage=mask_percentage, seed=seed)
        eval_mask = learning_type == "node"
        self.train_masked = masked(self.train_data, resample=True)
        self.val_masked = masked(self.val_data, enabled=eval_mask)
        self.test_masked = None if self.test_data is None else masked(self.test_data, enabled=eval_mask)

    @classmethod
    def from_cfg(cls, datas: Sequence[GraphDataset], cfg: CN, num_workers: int = 0) -> GraphAnnDataModule:
        """The datamodule a training run with ``cfg`` uses."""
        return cls(
            datas,
            batch_size=int(cfg.dataset.batch_size),
            num_workers=num_workers,
            mask_percentage=cfg.dataset.mask_percentage,
            mask_strategy=cfg.dataset.mask_strategy,
            learning_type=cfg.dataset.prediction_level,
            seed=cfg.optim.seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Select the training masks for ``epoch``; takes effect at the next train iterator."""
        self.train_masked.epoch = int(epoch)

    def _loader(self, dataset: MaskedGraphDataset, shuffle: bool) -> DataLoader | _Prefetch:
        # Pinned batches copy to the GPU faster and asynchronously (Lightning moves them with
        # non_blocking=True); with num_workers=0 the pinning runs on the _Prefetch thread too.
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader if self.num_workers else _Prefetch(loader)

    def train_dataloader(self) -> DataLoader | _Prefetch:
        return self._loader(self.train_masked, shuffle=True)

    def val_dataloader(self) -> DataLoader | _Prefetch:
        return self._loader(self.val_masked, shuffle=False)

    def test_dataloader(self) -> DataLoader | _Prefetch:
        if self.test_masked is None:
            raise RuntimeError("No test set was given to this datamodule.")
        return self._loader(self.test_masked, shuffle=False)
