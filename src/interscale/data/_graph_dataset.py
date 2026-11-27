"""The data contract between InterScale's storage and its models.

A model needs exactly three things from its data, and all of them are on the dataset:

- ``num_node_features`` -- the width of ``data.x`` (the model's input and, for regression, output),
- ``class_names`` -- the column order of the one-hot ``data.y`` (classification only),
- ``label_counts(level)`` -- per-class counts of the training units, for ``WeightedCE``.

Everything else is an ordinary :class:`torch_geometric.data.Dataset`: ``len()`` graphs, and
``get(i)`` returning one :class:`~torch_geometric.data.Data` per sample/window with ``x [N, G]``,
``edge_index [2, E]`` and, for classification, a one-hot ``y [N, C]``. Any storage that satisfies
this -- :class:`InMemoryGraphDataset` or the on-disk :class:`~interscale.data.ZarrGraphDataset` --
can train any InterScale model, which never sees where the graphs came from.

Masking is deliberately *not* part of the contract. :class:`MaskedGraphDataset` wraps any
contract dataset and draws the corruption per item from a seeded generator, so a stored graph is
never mutated and the draw does not depend on which worker process loads it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PygDataset

from interscale.tl.masking import MASK_STRATEGIES, sample_gene_mask, sample_node_mask


class GraphDataset(PygDataset):
    """Base class for the datasets InterScale models train on.

    Subclasses implement :meth:`len`, :meth:`get` and :meth:`_class_counts`.

    Parameters
    ----------
    var_names
        Names of the ``G`` columns of ``data.x``, in order. Saved with a checkpoint and checked
        against the dataset it is loaded with.
    class_names
        Names of the ``C`` columns of the one-hot ``data.y``, in order. ``None`` for regression.
    """

    def __init__(self, var_names: Sequence[str], class_names: Sequence[str] | None = None):
        super().__init__(root=None)
        self.var_names = [str(v) for v in var_names]
        self.class_names = None if class_names is None else [str(c) for c in class_names]

    # PyG's defaults read data to answer these: ``num_node_features`` loads ``self[0]`` and
    # ``num_classes`` loads *every* graph. Both are metadata here.
    @property
    def num_node_features(self) -> int:
        return len(self.var_names)

    @property
    def num_classes(self) -> int:
        return 0 if self.class_names is None else len(self.class_names)

    def _class_counts(self) -> np.ndarray:
        """Cells per class per graph, ``[len(self), num_classes]``."""
        raise NotImplementedError

    def label_counts(self, level: Literal["node", "graph"]) -> np.ndarray:
        """Per-class number of training units, ordered like :attr:`class_names`.

        Parameters
        ----------
        level
            ``"node"`` counts cells; ``"graph"`` counts graphs, each by its majority label.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``[num_classes]``.
        """
        if self.class_names is None:
            raise ValueError("label_counts needs class_names; this is a regression dataset.")
        per_graph = np.asarray(self._class_counts(), dtype=np.float64).reshape(-1, self.num_classes)
        if level == "node":
            return per_graph.sum(axis=0)
        if level == "graph":
            return np.bincount(per_graph.argmax(axis=1), minlength=self.num_classes).astype(np.float64)
        raise ValueError(f"level must be 'node' or 'graph', got {level!r}")


class InMemoryGraphDataset(GraphDataset):
    """A contract dataset over a list of graphs held in memory.

    Parameters
    ----------
    data_list
        One :class:`~torch_geometric.data.Data` per graph. Never mutated.
    var_names
        See :class:`GraphDataset`.
    class_names
        See :class:`GraphDataset`.
    """

    def __init__(
        self,
        data_list: Sequence[Data],
        var_names: Sequence[str],
        class_names: Sequence[str] | None = None,
    ):
        super().__init__(var_names=var_names, class_names=class_names)
        self.data_list = list(data_list)
        for i, data in enumerate(self.data_list):
            if data.x.shape[1] != self.num_node_features:
                raise ValueError(
                    f"graph {i} has {data.x.shape[1]} features but {self.num_node_features} var_names were given"
                )
            y = data.y
            if self.class_names is not None and (y is None or y.dim() != 2 or y.shape[1] != self.num_classes):
                shape = None if y is None else tuple(y.shape)
                raise ValueError(
                    f"graph {i}: expected a one-hot y of shape [N, {self.num_classes}] matching class_names, got {shape}"
                )

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def _class_counts(self) -> np.ndarray:
        return [data.y.sum(dim=0).double().numpy() for data in self.data_list]


class MaskedGraphDataset(TorchDataset):
    """Attaches a freshly drawn reconstruction mask to every graph of a contract dataset.

    The draw for graph ``i`` is a pure function of ``(seed, epoch, i)`` when ``resample`` is
    true and of ``(seed, i)`` otherwise, so

    - training masks change every epoch (set :attr:`epoch`, see
      :class:`~interscale.train._utils.NodeMaskResampleCallback`) while val/test masks stay fixed
      and monitored metrics remain comparable across epochs;
    - the result does not depend on ``num_workers``: each worker receives a pickled copy of this
      object when the epoch's iterator is created, ``epoch`` included. This is why InterScale's
      loaders never use ``persistent_workers=True``, which would keep the first epoch's copy.

    Each item is a shallow copy of the stored graph: tensors are shared, the stored
    :class:`~torch_geometric.data.Data` is never written to.

    Parameters
    ----------
    dataset
        Any :class:`GraphDataset`.
    mask_strategy
        ``"node"`` writes ``data.mask`` ``[N]``; ``"gene"`` writes ``data.gene_mask`` ``[N, G]``
        and ``data.mask`` as its row-wise OR. See :mod:`interscale.tl.masking`.
    mask_percentage
        Bernoulli rate, per cell under ``"node"`` and per entry under ``"gene"``.
    enabled
        If false, every item gets an all-false ``data.mask`` and no ``gene_mask`` -- used for
        graph-level eval splits, whose inputs must not be corrupted.
    resample
        Whether the draw depends on :attr:`epoch`.
    seed
        Base seed of the per-item generators.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        mask_strategy: Literal["node", "gene"] = "node",
        mask_percentage: float = 0.5,
        enabled: bool = True,
        resample: bool = False,
        seed: int = 0,
    ):
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError(f"mask_strategy must be one of {MASK_STRATEGIES!r}.")
        self.dataset = dataset
        self.mask_strategy = mask_strategy
        self.mask_percentage = mask_percentage
        self.enabled = enabled
        self.resample = resample
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def _generator(self, idx: int) -> torch.Generator:
        # SeedSequence mixes the entropy words properly, so neighbouring (epoch, idx) pairs give
        # unrelated streams instead of the correlated ones a seed + idx would.
        words = [self.seed, self.epoch, idx] if self.resample else [self.seed, idx]
        seed = int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> Data:
        return self._masked(idx, self.dataset[idx])

    def __getitems__(self, indices) -> list[Data]:
        # Pass whole batches through, so a dataset that reads them concurrently (the zarr store) can.
        fetch = getattr(self.dataset, "__getitems__", None)
        items = fetch(indices) if fetch is not None else [self.dataset[i] for i in indices]
        return [self._masked(i, data) for i, data in zip(indices, items, strict=True)]

    def _masked(self, idx: int, data: Data) -> Data:
        data = copy.copy(data)
        if "gene_mask" in data:
            del data.gene_mask
        if not self.enabled:
            data.mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        elif self.mask_strategy == "gene":
            data.gene_mask = sample_gene_mask(
                data.num_nodes, data.x.shape[1], self.mask_percentage, self._generator(idx)
            )
            data.mask = data.gene_mask.any(dim=1)
        else:
            data.mask = sample_node_mask(data.num_nodes, self.mask_percentage, self._generator(idx))
        return data
