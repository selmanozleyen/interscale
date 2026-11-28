"""PyG datasets over a grouped AnnData (:mod:`interscale.geome.io`): one graph per group.

Staged in InterScale until it moves to ``geome.data``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PygDataset

from interscale.geome._refs import Field, to_path
from interscale.geome.io import CONNECTIVITIES, GroupedAnnData, _column


class GroupDataset(PygDataset):
    """The groups of a grouped AnnData as graphs: group ``i`` is rows ``group_ptr[i]:group_ptr[i+1]``.

    Each item is a :class:`~torch_geometric.data.Data` with ``x`` (``[N, G]`` float32, dense), the group's
    ``edge_index`` from ``obsp["spatial_connectivities"]``, ``source_row`` (row in the original file),
    a one-hot ``y`` if ``y`` is given, and one attribute per ``extra`` entry. :meth:`__getitems__`, which
    torch's DataLoader calls per batch, reads every array once for the whole batch.

    Parameters
    ----------
    source
        A ``.zarr`` path written by :func:`~interscale.geome.io.write_grouped`, a
        :class:`~interscale.geome.io.GroupedAnnData`, or an AnnData object read from one (in memory,
        backed or lazy).
    x
        Node features: ``A.X`` / ``A.layers["<name>"]``, or the same as a ``.`` string (``"layers.<name>"``) or
        ``/`` path (``"layers/<name>"``); see :mod:`interscale.geome._refs`.
    y
        Optional categorical ``obs`` column (``A.obs["<col>"]``, ``"obs.<col>"``, ``"obs/<col>"`` or the bare
        name), one-hot encoded over all of its categories.
    extra
        ``{attribute: field}`` further per-cell arrays, e.g. ``{"emb": A.obsm["latent"]}``.
    split
        Keep only the groups of this split (``uns["geome"]["group_split"]``); None keeps all.
    """

    def __init__(
        self,
        source: GroupedAnnData | AnnData | str | os.PathLike,
        *,
        x: Field = "X",
        y: Field | None = None,
        extra: Mapping[str, Field] | None = None,
        split: str | None = None,
    ):
        super().__init__(root=None)
        self.reader = source if isinstance(source, GroupedAnnData) else GroupedAnnData(source)
        self.x_key = to_path(x)
        self.y_key = None if y is None else _column(y)
        self.extra = {attr: to_path(field) for attr, field in (extra or {}).items()}
        groups = self.reader.groups
        self.group_ids = np.flatnonzero(groups["split"].to_numpy() == split) if split else np.arange(len(groups))
        self.var_names = self.reader.var_names
        self.categories = None
        self._codes = None
        if y is not None:
            column = self.reader.obs[self.y_key].astype("category")
            self.categories = [str(c) for c in column.cat.categories]
            self._codes = column.cat.codes.to_numpy()
        self._rows = self.reader.obs["source_row"].to_numpy()

    def len(self) -> int:
        return len(self.group_ids)

    def get(self, idx: int) -> Data:
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[Data]:
        groups = self.group_ids[np.asarray([self.indices()[i] for i in indices], dtype=np.int64)]
        order = np.argsort(groups, kind="stable")
        keys = [self.x_key, f"obsp/{CONNECTIVITIES}", *self.extra.values()]
        blocks = self.reader.read(list(dict.fromkeys(keys)), groups[order])
        out = [None] * len(indices)
        for k, j in enumerate(order):
            g = groups[order][k]
            lo, hi = self.reader.group_ptr[g], self.reader.group_ptr[g + 1]
            x = blocks[self.x_key][k]
            data = Data(
                x=torch.from_numpy(np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)),
                edge_index=torch.from_numpy(blocks[f"obsp/{CONNECTIVITIES}"][k]).long(),
                source_row=torch.from_numpy(self._rows[lo:hi].astype(np.int64)),
            )
            if self._codes is not None:
                codes = torch.from_numpy(self._codes[lo:hi].astype(np.int64))
                data.y = torch.nn.functional.one_hot(codes, len(self.categories)).float()
            for attr, key in self.extra.items():
                block = blocks[key][k]
                data[attr] = torch.from_numpy(np.asarray(block.toarray() if sp.issparse(block) else block))
            out[j] = data
        return out if self.transform is None else [self.transform(d) for d in out]

    def label_counts(self, level: str = "node") -> np.ndarray:
        """Per-category counts over this dataset's groups: cells (``"node"``) or groups by majority (``"graph"``)."""
        if self._codes is None:
            raise ValueError("label_counts needs y.")
        ptr, c = self.reader.group_ptr, len(self.categories)
        per_group = np.stack([np.bincount(self._codes[ptr[g] : ptr[g + 1]], minlength=c) for g in self.group_ids])
        if level == "node":
            return per_group.sum(axis=0).astype(np.float64)
        return np.bincount(per_group.argmax(axis=1), minlength=c).astype(np.float64)
