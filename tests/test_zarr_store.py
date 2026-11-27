"""The zarr graph store is a drop-in, out-of-core implementation of the data contract."""

import anndata as ad
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from interscale.data import (
    ZarrGraphDataset,
    _build,
    build_graph_datasets,
    graph_store_path,
    load_graph_datasets,
    open_graph_store,
    write_graph_store,
)
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.model.local_model import LocalModel
from interscale.tl import remove_zero_expression_cells

PRECOMPUTED = [
    "model.global_component.parameters.type_gex_embedding", "Precomputed",
    "model.global_component.latent_obsm_key", "latent",
]  # fmt: skip


def _assert_same_graphs(store_sets, memory_sets, keys=("x", "edge_index", "y", "obs_names", "embeddings")):
    assert [len(d) for d in store_sets] == [len(d) for d in memory_sets]
    for store, memory in zip(store_sets, memory_sets, strict=True):
        assert (store.var_names, store.class_names) == (memory.var_names, memory.class_names)
        for i in range(len(store)):
            a, b = store[i], memory[i]
            for key in keys:
                assert (key in a) == (key in b), f"graph {i}: {key} present in only one"
                if key in b:
                    assert torch.equal(a[key], b[key]), f"graph {i}: {key} differs"


def _source(kind, adata, tmp_path):
    """The same AnnData as an in-memory object, a backed object, or a path the writer opens."""
    if kind == "memory":
        return adata
    path = tmp_path / ("in.zarr" if kind == "zarr" else "in.h5ad")
    adata.write_zarr(path) if kind == "zarr" else adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r") if kind == "backed" else path


@pytest.mark.parametrize(
    ("task", "sample_key", "sparse", "extra", "source"),
    [
        ("classification", ("sample",), False, [], "memory"),
        ("classification", ("sample", "fov"), True, ["dataset.layer_key", "log1p"], "h5ad"),
        ("regression", ("sample",), True, ["dataset.layer_key", "log1p"], "backed"),
        ("regression", ("sample",), False, PRECOMPUTED, "zarr"),
        ("classification", ("sample",), False, ["dataset.layer_key", "log1p"], "h5ad"),
    ],
)
def test_store_reproduces_prepare_geome_dataset(
    make_cfg, make_spatial_adata, tmp_path, task, sample_key, sparse, extra, source
):
    cfg = make_cfg(task=task, sample_key=sample_key, extra=extra)

    path = write_graph_store(_source(source, make_spatial_adata(sparse=sparse), tmp_path), cfg, tmp_path / "g.zarr")

    _assert_same_graphs(open_graph_store(path, cfg), build_graph_datasets(make_spatial_adata(sparse=sparse), cfg))


def test_zero_expression_filter_matches_the_in_memory_one(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    adata = make_spatial_adata(sparse=True)
    empty = np.flatnonzero(adata.obs["sample"].isin(["s1", "s4"]))[::3]
    adata.X[empty] = 0
    adata.X.eliminate_zeros()

    path = write_graph_store(_source("h5ad", adata, tmp_path), cfg, tmp_path / "g.zarr", drop_zero_expression=True)

    store = open_graph_store(path, cfg)
    filtered = remove_zero_expression_cells(adata.copy())
    _assert_same_graphs(store, build_graph_datasets(filtered.copy(), cfg), keys=("x", "edge_index", "y"))
    kept = np.flatnonzero(np.asarray(adata.X.sum(axis=1)).ravel() != 0)
    for dataset in store:  # obs_index points at rows of the file, not of the filtered AnnData
        for i in range(len(dataset)):
            assert not np.isin(dataset[i].obs_index.numpy(), empty).any()
            assert np.isin(dataset[i].obs_index.numpy(), kept).all()


def test_graphs_are_contiguous_and_map_back_to_their_rows(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    adata = make_spatial_adata()
    path = write_graph_store(adata, cfg, tmp_path / "g.zarr")

    for dataset in open_graph_store(path, cfg):
        for i in range(len(dataset)):
            rows = dataset[i].obs_index.numpy()
            np.testing.assert_array_equal(dataset[i].x.numpy(), adata.X[rows])
            assert adata.obs["sample"].iloc[rows].nunique() == 1
            assert set(adata.obs["split"].iloc[rows]) == {dataset.split}


def test_label_counts_match_the_in_memory_dataset(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    store = open_graph_store(write_graph_store(make_spatial_adata(), cfg, tmp_path / "g.zarr"), cfg)
    memory = build_graph_datasets(make_spatial_adata(), cfg)

    for level in ("node", "graph"):
        np.testing.assert_array_equal(store[0].label_counts(level), memory[0].label_counts(level))


def test_store_is_tied_to_its_config(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    other = make_cfg(extra=["dataset.spatial_neigbors_kwargs.radius", 10])
    path = write_graph_store(make_spatial_adata(), cfg, tmp_path / "g.zarr")

    assert graph_store_path(tmp_path, cfg) != graph_store_path(tmp_path, other)
    with pytest.raises(ValueError, match="different config"):
        open_graph_store(path, other)
    with pytest.raises(FileExistsError):
        write_graph_store(make_spatial_adata(), cfg, path)
    write_graph_store(make_spatial_adata(), cfg, path, overwrite=True)


def test_failed_write_leaves_no_store(make_cfg, make_spatial_adata, tmp_path):
    adata = make_spatial_adata()
    del adata.obsm["spatial"]

    with pytest.raises(ValueError, match="spatial"):
        write_graph_store(adata, make_cfg(), tmp_path / "g.zarr")
    assert list(tmp_path.glob("*g.zarr*")) == []


def test_workers_open_their_own_handle_and_read_the_same_graphs(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    train = ZarrGraphDataset(write_graph_store(make_spatial_adata(), cfg, tmp_path / "g.zarr"), "train")
    _ = train[0]  # open the handle in the parent so pickling has to drop it

    for i, batch in enumerate(DataLoader(train, batch_size=1, shuffle=False, num_workers=2)):
        assert torch.equal(batch.x, train[i].x)


def test_load_graph_datasets_trains_out_of_core(make_cfg, make_spatial_adata, tmp_path, monkeypatch):
    """With a graph store configured, the h5ad is never loaded into memory: the first run builds
    the store from the file row by row, later runs only open it."""
    make_spatial_adata().write_h5ad(tmp_path / "in.h5ad")
    epochs = ["optim.n_epochs", 2, "optim.min_epochs", 2, "optim.lr_warmup", 1, "optim.accelerator", "cpu"]
    cfg = make_cfg(task="regression", extra=["dataset.h5ad_data", str(tmp_path / "in.h5ad"), *epochs])
    monkeypatch.setattr(_build, "load_training_adata", lambda cfg: pytest.fail("loaded the AnnData into memory"))

    datasets = load_graph_datasets(cfg)
    monkeypatch.setattr(_build, "write_graph_store", lambda *a, **k: pytest.fail("rebuilt an existing store"))
    assert load_graph_datasets(cfg)[0].path == datasets[0].path == graph_store_path(cfg.dataset.graph_store, cfg)

    model = LocalModel(datasets[0], cfg)
    model.train(
        max_epochs=2, datamodule=GraphAnnDataModule.from_cfg(datasets, cfg, num_workers=0), enable_progress_bar=False
    )
    assert model.is_trained_


def test_segmentation_noise_builds_the_store_from_memory(make_cfg, make_spatial_adata, monkeypatch):
    """Segmentation noise needs the whole AnnData, so that one case is not out of core."""
    cfg = make_cfg(extra=["dataset.segmentation_robustness", [0.1, 0.5]])
    loaded = []
    monkeypatch.setattr(_build, "load_training_adata", lambda cfg: loaded.append(1) or make_spatial_adata())

    load_graph_datasets(cfg)

    assert loaded == [1]
