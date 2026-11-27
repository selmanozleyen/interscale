from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from interscale.data import GraphDataset, InMemoryGraphDataset, MaskedGraphDataset
from interscale.tl.masking import MASK_STRATEGIES

VALID_STAGE = {"fit", "test", "validate", None}
VALID_SPLIT = {"node", "graph"}


def _as_graph_dataset(datas: GraphDataset | Sequence[Data]) -> GraphDataset:
    """Accept a contract dataset as is, or wrap a bare list of graphs (tests, notebooks)."""
    if isinstance(datas, GraphDataset):
        return datas
    datas = list(datas)
    n_features = datas[0].x.shape[1] if datas else 0
    return InMemoryGraphDataset(datas, var_names=[str(i) for i in range(n_features)])


class GraphAnnDataModule(pl.LightningDataModule):
    """Lightning DataModule over InterScale contract datasets.

    Parameters
    ----------
    datas
        ``[train, val]`` or ``[train, val, test]``, each a :class:`~interscale.data.GraphDataset`
        (or a plain list of :class:`~torch_geometric.data.Data`, wrapped in an
        :class:`~interscale.data.InMemoryGraphDataset`).
    batch_size
        Number of graphs per batch.
    num_workers
        DataLoader worker processes. Workers are re-created every epoch (no persistent workers),
        which is how the epoch reaches the mask draw; see :class:`~interscale.data.MaskedGraphDataset`.
    mask_percentage
        Bernoulli masking probability, per cell under ``mask_strategy="node"`` and per
        (cell, gene) entry under ``"gene"``.
    mask_strategy
        Granularity of the corruption. ``"node"`` blanks whole cells (loss over all G genes of the
        masked cells); ``"gene"`` blanks individual (cell, gene) entries in every cell (loss over
        those entries only). See :mod:`interscale.tl.masking`.
    learning_type
        ``"node"`` or ``"graph"``. For node-level learning the masked nodes *are* the
        supervision/evaluation targets, so every split is masked. For graph-level learning masking
        is pure input augmentation, so only train is masked.
    seed
        Base seed of the mask draws. Train masks are a function of ``(seed, epoch, graph)``,
        val/test masks of ``(seed, graph)``.
    """

    def __init__(
        self,
        datas: Sequence[GraphDataset | Sequence[Data]],
        batch_size: int = 1,
        num_workers: int = 1,
        mask_percentage: float = 0.5,
        mask_strategy: Literal["node", "gene"] = "node",
        learning_type: Literal["node", "graph"] = "node",
        seed: int = 0,
    ):
        super().__init__()
        if datas is None or len(datas) not in (2, 3):
            raise ValueError("datas must hold a train and a validation set, and optionally a test set.")
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError("mask_strategy must be one of %r." % (MASK_STRATEGIES,))

        self.train_data = _as_graph_dataset(datas[0])
        self.val_data = _as_graph_dataset(datas[1])
        self.test_data = _as_graph_dataset(datas[2]) if len(datas) == 3 else None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask_percentage = mask_percentage
        self.mask_strategy = mask_strategy
        self.learning_type = learning_type
        self.seed = seed

        eval_mask = learning_type == "node"
        self.train_masked = self._masked(self.train_data, enabled=True, resample=True)
        self.val_masked = self._masked(self.val_data, enabled=eval_mask, resample=False)
        self.test_masked = (
            None if self.test_data is None else self._masked(self.test_data, enabled=eval_mask, resample=False)
        )

    def _masked(self, dataset: GraphDataset, *, enabled: bool, resample: bool) -> MaskedGraphDataset:
        return MaskedGraphDataset(
            dataset,
            mask_strategy=self.mask_strategy,
            mask_percentage=self.mask_percentage,
            enabled=enabled,
            resample=resample,
            seed=self.seed,
        )

    def setup(self, stage: str | None = None):
        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

    def set_epoch(self, epoch: int) -> None:
        """Select the training masks for ``epoch``. Takes effect at the next train iterator."""
        self.train_masked.epoch = int(epoch)

    def _loader(self, dataset: MaskedGraphDataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_masked, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_masked, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_masked is None:
            raise RuntimeError("No test set was given to this datamodule.")
        return self._loader(self.test_masked, shuffle=False)
