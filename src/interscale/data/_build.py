from __future__ import annotations

from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data._graph_dataset import InMemoryGraphDataset
from interscale.tl.geome_utils import prepare_geome_dataset


def build_graph_datasets(adata: AnnData, cfg: CN) -> list[InMemoryGraphDataset]:
    """Build the train/val(/test) contract datasets from an in-memory AnnData.

    Thin wrapper over :func:`interscale.tl.prepare_geome_dataset` that attaches the metadata the
    models read: ``var_names`` from ``adata.var_names`` and, for classification, ``class_names``
    from the categories of ``adata.obs[cfg.dataset.prediction_obs]`` -- the column order of the
    one-hot ``data.y`` geome writes, which :class:`InMemoryGraphDataset` checks graph by graph.

    Like ``prepare_geome_dataset`` it renames ``adata.obs_names`` to ``"1".."N"`` and casts the
    sample keys to categorical, in place.

    Parameters
    ----------
    adata
        The AnnData to build graphs from; must hold ``cfg.dataset.split_key`` in ``obs``.
    cfg
        The run config.

    Returns
    -------
    list[InMemoryGraphDataset]
        ``[train, val]``, or ``[train, val, test]`` when the split has a ``"test"`` category.
    """
    class_names = None
    if "classification" in cfg.dataset.prediction_task:
        class_names = adata.obs[cfg.dataset.prediction_obs].astype("category").cat.categories

    splits, _ = prepare_geome_dataset(adata, cfg)
    return [InMemoryGraphDataset(data_list, var_names=adata.var_names, class_names=class_names) for data_list in splits]
