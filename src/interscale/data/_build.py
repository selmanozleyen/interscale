from __future__ import annotations

from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data._graph_dataset import InMemoryGraphDataset
from interscale.data._zarr_store import graph_store_path, open_graph_store, write_graph_store
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


def load_training_adata(cfg: CN) -> AnnData:
    """Read ``cfg.dataset.h5ad_data`` into memory, drop cells with no expression and, if
    ``cfg.dataset.segmentation_robustness`` is ``[node_fraction, overflow_fraction]``, simulate
    segmentation errors.

    Parameters
    ----------
    cfg
        The run config.

    Returns
    -------
    AnnData
        The preprocessed, in-memory AnnData.
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
    """The train/val(/test) datasets for a training run.

    With ``cfg.dataset.graph_store`` (a directory) set, the graphs come from
    :func:`~interscale.data.graph_store_path`, written on first use. The store is built straight
    from ``h5ad_data`` without loading its expression matrix, except under segmentation noise,
    which needs the whole AnnData in memory (:func:`load_training_adata`). Without a graph store
    the graphs are built in memory.

    Parameters
    ----------
    cfg
        The run config.

    Returns
    -------
    list
        Contract datasets, ``[train, val]`` or ``[train, val, test]``.
    """
    if not cfg.dataset.graph_store:
        return build_graph_datasets(load_training_adata(cfg), cfg)
    path = graph_store_path(cfg.dataset.graph_store, cfg)
    if path.exists():
        print(f"Reusing graph store {path}")
    else:
        print(f"Writing graph store {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.dataset.segmentation_robustness is None:
            write_graph_store(cfg.dataset.h5ad_data, cfg, path, drop_zero_expression=True)
        else:
            write_graph_store(load_training_adata(cfg), cfg, path)
    return open_graph_store(path, cfg)
