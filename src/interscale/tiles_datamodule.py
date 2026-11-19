"""InterScale windows as geome spatial tiles: a tile plus its neighbour margin, read one tile at a time.

`GraphAnnDataModule` turns every window into a PyG graph up front, with edges cut at the window border. Here a
window is a :class:`geome.tiles.TileLoader` tile: at most ``max_seq_len`` cells (the *core*, what the transformer
sees), plus every cell within ``num_layers + 1`` hops of it (the *margin*, which only the local GNN sees). The
graph is built once for the whole sample, so the local embeddings of the core cells are the ones the whole-sample
graph would give: the GCN's own degree normalization needs the extra hop to see each margin cell's full degree.

Cells are only masked, and so only scored, in the core. :func:`drop_halo` removes the margin before the
transformer.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from yacs.config import CfgNode as CN

__all__ = ["TileDataModule", "drop_halo"]


def drop_halo(batch):
    """``batch`` restricted to its core cells; unchanged if it has no margin."""
    core = getattr(batch, "core_mask", None)
    if core is None or bool(core.all()):
        return batch
    return batch.subgraph(core)


class _Tiles(Dataset):
    """Tiles ``index`` of ``loader``, each with a mask drawn on its core cells only."""

    def __init__(self, loader, index: list[int], mask_percentage: float, mask_strategy: str, fixed_seed: int | None):
        self.loader, self.index = loader, index
        self.p, self.strategy, self.fixed_seed = mask_percentage, mask_strategy, fixed_seed

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, j: int):
        data = self.loader.tile(self.index[j])
        del data.edge_weight  # InterScale's GCN normalizes its own edges
        data.obs_names = data.rows + 1  # InterScale names cells "1".."n" (prepare_geome_dataset)
        # val/test: the same mask every epoch, so monitored metrics stay comparable; train: a fresh one
        gen = None if self.fixed_seed is None else torch.Generator().manual_seed(self.fixed_seed + self.index[j])
        core = data.core_mask
        if self.strategy == "gene":
            data.gene_mask = (torch.rand(data.x.shape, generator=gen) < self.p) & core[:, None]
            data.mask = data.gene_mask.any(dim=1)
        else:
            data.mask = (torch.rand(data.num_nodes, generator=gen) < self.p) & core
        return data


class TileDataModule(pl.LightningDataModule):
    """Drop-in for :class:`~interscale.geome_dataloader.GraphAnnDataModule` (node-level regression).

    Parameters
    ----------
    adata
        Data with ``cfg.dataset.layer_key``, ``obsm["spatial"]``, the ``sample_key`` columns and ``split_key``.
    cfg
        The run config.
    graph
        Adjacency over all cells, e.g. ``sq.gr.spatial_neighbors(...)``'s connectivities computed per sample and
        split. Defaults to geome's kNN graph within each sample and split.
    tile_size
        Core cells per tile; defaults to the transformer's ``max_seq_len``.
    num_workers
        Workers per dataloader; tiles are read from ``adata`` (in memory or backed) in them.
    """

    def __init__(self, adata, cfg: CN, *, graph=None, tile_size: int | None = None, num_workers: int = 0):
        super().__init__()
        from anndata.acc import A
        from geome.tiles import TileLoader

        if "regression" not in cfg.dataset.prediction_task or cfg.dataset.prediction_level != "node":
            raise NotImplementedError("TileDataModule covers node-level regression (self-supervised) only.")
        rep = A.layers[cfg.dataset.layer_key] if cfg.dataset.layer_key else A.X
        groups = [*(A.obs[k] for k in cfg.dataset.sample_key), A.obs[cfg.dataset.split_key]]
        self.loader = TileLoader(
            adata,
            rep=rep,
            groups=groups,
            graph=graph,
            n_hops=int(cfg.model.local_component.parameters.num_layers) + 1,
            tile_size=int(tile_size or cfg.model.global_component.parameters.max_seq_len),
            shuffle=False,
        )
        split = np.array([key[-1] for key in self.loader.tile_groups])
        kw = {"mask_percentage": cfg.dataset.mask_percentage, "mask_strategy": cfg.dataset.mask_strategy}
        seed = int(cfg.optim.seed)
        self.train_data = _Tiles(self.loader, np.flatnonzero(split == "train").tolist(), fixed_seed=None, **kw)
        self.val_data = _Tiles(self.loader, np.flatnonzero(split == "val").tolist(), fixed_seed=seed, **kw)
        self.test_data = _Tiles(self.loader, np.flatnonzero(split == "test").tolist(), fixed_seed=seed, **kw)
        self.batch_size = int(cfg.dataset.batch_size)
        self.num_workers = num_workers
        self.learning_type = "node"
        self.setup_called = False

    def setup(self, stage: str | None = None) -> None:
        self.setup_called = True

    def resample_train_mask(self) -> None:
        """No-op: training tiles draw a fresh mask every time they are read."""

    def _loader(self, data, shuffle: bool) -> DataLoader:
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_data, shuffle=False)
