from __future__ import annotations

from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data._graph_dataset import InMemoryGraphDataset
from interscale.data._zarr_store import graph_store_path, open_graph_store, write_graph_store
from interscale.tl.geome_utils import prepare_geome_dataset


def build_graph_datasets(adata: AnnData, cfg: CN) -> list[InMemoryGraphDataset]:
    """:func:`~interscale.tl.prepare_geome_dataset`'s splits as contract datasets, ``[train, val(, test)]``.

    Adds ``var_names`` and, for classification, ``class_names`` (the one-hot column order). Like
    ``prepare_geome_dataset``, renames ``obs_names`` to ``"1".."N"`` and casts sample keys to
    categorical in place.
    """
    class_names = None
    if "classification" in cfg.dataset.prediction_task:
        class_names = adata.obs[cfg.dataset.prediction_obs].astype("category").cat.categories

    splits, _ = prepare_geome_dataset(adata, cfg)
    return [InMemoryGraphDataset(data_list, var_names=adata.var_names, class_names=class_names) for data_list in splits]


def load_training_adata(cfg: CN) -> AnnData:
    """Read ``cfg.dataset.h5ad_data``, drop cells with no expression and apply
    ``cfg.dataset.segmentation_robustness`` (``[node_fraction, overflow_fraction]``) if set.
    """
    import scanpy as sc

    from interscale.tl import remove_zero_expression_cells

    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata = remove_zero_expression_cells(adata)
    print(adata)
    if cfg.dataset.segmentation_robustness is not None:
        # Imported only when needed: importing interscale.pp currently fails, because its
        # __init__ imports modules (background, donor_scale) that are not in the repository.
        import squidpy as sq

        from interscale.pp.segmentation_noise import apply_segmentation_noise

        node_fraction, overflow_fraction = cfg.dataset.segmentation_robustness
        print(f"Applying segmentation noise: node fraction {node_fraction}, overflow fraction {overflow_fraction}")
        sq.gr.spatial_neighbors(adata, **cfg.dataset.spatial_neigbors_kwargs)
        adata = apply_segmentation_noise(adata, node_fraction, overflow_fraction)
    return adata


def load_graph_datasets(cfg: CN) -> list:
    """The ``[train, val(, test)]`` contract datasets of a run.

    With ``cfg.dataset.graph_store`` set they come from the store at
    :func:`~interscale.data.graph_store_path`, written on first use straight from ``h5ad_data``
    (segmentation noise needs the AnnData in memory); otherwise they are built in memory.
    """
    if not cfg.dataset.graph_store:
        return build_graph_datasets(load_training_adata(cfg), cfg)
    path = graph_store_path(cfg.dataset.graph_store, cfg)
    if path.exists():
        print(f"Reusing graph store {path}")
    else:
        print(f"Writing graph store {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        noise = cfg.dataset.segmentation_robustness is not None  # load_training_adata drops zero cells itself
        source = load_training_adata(cfg) if noise else cfg.dataset.h5ad_data
        write_graph_store(
            source, cfg, path, drop_zero_expression=not noise, shard_rows=cfg.dataset.graph_store_shard_rows
        )
    return open_graph_store(path, cfg)
