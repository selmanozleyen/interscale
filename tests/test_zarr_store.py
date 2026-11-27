"""The zarr graph store is a drop-in implementation of the data contract."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch_geometric.loader import DataLoader

from interscale.config import load_config
from interscale.data import (
    ZarrGraphDataset,
    build_graph_datasets,
    graph_store_datasets,
    graph_store_path,
    open_graph_store,
    write_graph_store,
)
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.model.local_model import LocalModel

GENES = [f"g{i}" for i in range(6)]
CLASSES = ["A", "B", "C"]


@pytest.fixture
def components_yaml(tmp_path):
    path = tmp_path / "components.yaml"
    path.write_text("model:\n  local_component:\n    name: GCN\n  global_component:\n    name: self-attn-transformer\n")
    return path


def _cfg(components_yaml, tmp_path, *, task="classification", sample_key=("sample",), extra=()):
    overrides = [
        "dataset.sample_key", list(sample_key),
        "dataset.prediction_task", task,
        "dataset.name", "unit",
        "dataset.graph_store", str(tmp_path / "stores"),
        "optim.loss", "CrossEntropy" if task == "classification" else "MSELoss",
        "model.n_embed", 4,
        "model.save", str(tmp_path / "models"),
        "wandb.use", False,
        *extra,
    ]  # fmt: skip
    if task == "classification":
        overrides += ["dataset.prediction_obs", "ct"]
    return load_config(components_yaml, overrides=overrides)


def _adata(seed: int = 0, sparse: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    samples = ["s1", "s2", "s3", "s4", "s5"]
    n_per = 12
    n = n_per * len(samples)
    x = rng.poisson(1.0, size=(n, len(GENES))).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(x) if sparse else x)
    adata.var_names = GENES
    adata.layers["log1p"] = sp.csr_matrix(np.log1p(x)) if sparse else np.log1p(x)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obsm["latent"] = rng.normal(size=(n, 3)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, n_per))
    adata.obs["fov"] = pd.Categorical(rng.choice(["f1", "f2"], size=n))
    # Shuffle the rows so the store, not the input order, has to make each graph contiguous.
    order = rng.permutation(n)
    adata = adata[order].copy()
    adata.obs["split"] = pd.Categorical(
        adata.obs["sample"].map({"s1": "train", "s2": "train", "s3": "train", "s4": "val", "s5": "test"}).astype(str)
    )
    adata.obs["ct"] = pd.Categorical(rng.choice(CLASSES, size=n))
    return adata


def _assert_same_graphs(store_sets, memory_sets):
    assert [len(d) for d in store_sets] == [len(d) for d in memory_sets]
    for store, memory in zip(store_sets, memory_sets, strict=True):
        assert store.var_names == memory.var_names
        assert store.class_names == memory.class_names
        for i in range(len(store)):
            a, b = store[i], memory[i]
            for key in ("x", "edge_index", "y", "obs_names", "embeddings"):
                if key in b:
                    assert torch.equal(a[key], b[key]), f"split graph {i}: {key} differs"
                else:
                    assert key not in a


@pytest.mark.parametrize(
    ("task", "sample_key", "sparse", "layer"),
    [
        ("classification", ("sample",), False, None),
        ("classification", ("sample", "fov"), True, "log1p"),
        ("regression", ("sample",), True, "log1p"),
    ],
)
def test_store_reproduces_prepare_geome_dataset(components_yaml, tmp_path, task, sample_key, sparse, layer):
    cfg = _cfg(components_yaml, tmp_path, task=task, sample_key=sample_key, extra=["dataset.layer_key", layer])

    path = write_graph_store(_adata(sparse=sparse), cfg, tmp_path / "g.zarr")

    _assert_same_graphs(open_graph_store(path, cfg), build_graph_datasets(_adata(sparse=sparse), cfg))


def test_store_carries_precomputed_embeddings(components_yaml, tmp_path):
    cfg = _cfg(
        components_yaml,
        tmp_path,
        task="regression",
        extra=[
            "model.global_component.parameters.type_gex_embedding", "Precomputed",
            "model.global_component.latent_obsm_key", "latent",
        ],
    )  # fmt: skip

    path = write_graph_store(_adata(), cfg, tmp_path / "g.zarr")

    store = open_graph_store(path, cfg)
    assert store[0][0].embeddings.shape[1] == 3
    _assert_same_graphs(store, build_graph_datasets(_adata(), cfg))


def test_backed_input_writes_the_same_store(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path, extra=["dataset.layer_key", "log1p"])
    _adata(sparse=True).write_h5ad(tmp_path / "in.h5ad")
    backed = ad.read_h5ad(tmp_path / "in.h5ad", backed="r")

    path = write_graph_store(backed, cfg, tmp_path / "g.zarr")

    _assert_same_graphs(open_graph_store(path, cfg), build_graph_datasets(_adata(sparse=True), cfg))


def test_graphs_are_contiguous_and_map_back_to_their_rows(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    adata = _adata()
    path = write_graph_store(adata, cfg, tmp_path / "g.zarr")

    for dataset in open_graph_store(path, cfg):
        for i in range(len(dataset)):
            graph = dataset[i]
            rows = graph.obs_index.numpy()
            np.testing.assert_array_equal(graph.x.numpy(), adata.X[rows])
            assert adata.obs["sample"].iloc[rows].nunique() == 1
            assert set(adata.obs["split"].iloc[rows]) == {dataset.split}


def test_label_counts_and_model_metadata_match_the_in_memory_dataset(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path, extra=["optim.loss", "WeightedCE"])
    path = write_graph_store(_adata(), cfg, tmp_path / "g.zarr")
    store, memory = open_graph_store(path, cfg), build_graph_datasets(_adata(), cfg)

    for level in ("node", "graph"):
        np.testing.assert_array_equal(store[0].label_counts(level), memory[0].label_counts(level))
    from_store, from_memory = LocalModel(store[0], cfg), LocalModel(memory[0], cfg)
    assert (from_store.n_input, from_store.n_output) == (from_memory.n_input, from_memory.n_output)
    assert torch.equal(from_store.class_weights, from_memory.class_weights)


def test_store_built_for_another_config_is_rejected(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    path = write_graph_store(_adata(), cfg, tmp_path / "g.zarr")
    other = _cfg(components_yaml, tmp_path, extra=["dataset.spatial_neigbors_kwargs.radius", 10])

    with pytest.raises(ValueError, match="different config"):
        open_graph_store(path, other)


def test_existing_store_is_not_overwritten_by_default(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    path = write_graph_store(_adata(), cfg, tmp_path / "g.zarr")

    with pytest.raises(FileExistsError):
        write_graph_store(_adata(), cfg, path)
    write_graph_store(_adata(), cfg, path, overwrite=True)


def test_graph_store_datasets_builds_once_then_reuses_without_the_adata(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    calls = []

    def load():
        calls.append(1)
        return _adata()

    first = graph_store_datasets(cfg, load)
    second = graph_store_datasets(cfg, load)

    assert calls == [1], "a matching store must be reused without loading the AnnData"
    assert first[0].path == second[0].path == graph_store_path(cfg.dataset.graph_store, cfg)
    other = _cfg(components_yaml, tmp_path, extra=["dataset.spatial_neigbors_kwargs.radius", 10])
    assert graph_store_path(other.dataset.graph_store, other) != first[0].path


def test_failed_write_leaves_no_store(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    adata = _adata()
    del adata.obsm["spatial"]

    with pytest.raises(ValueError, match="spatial"):
        write_graph_store(adata, cfg, tmp_path / "g.zarr")
    assert list(tmp_path.glob("*g.zarr*")) == []


def test_workers_open_their_own_handle_and_read_the_same_graphs(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)
    path = write_graph_store(_adata(), cfg, tmp_path / "g.zarr")
    train = ZarrGraphDataset(path, "train")
    _ = train[0]  # open the handle in the parent so pickling has to drop it

    received = list(DataLoader(train, batch_size=1, shuffle=False, num_workers=2))

    for i, batch in enumerate(received):
        assert torch.equal(batch.x, train[i].x)


def test_model_trains_from_the_store(components_yaml, tmp_path):
    cfg = _cfg(
        components_yaml,
        tmp_path,
        task="regression",
        extra=["optim.n_epochs", 2, "optim.min_epochs", 2, "optim.lr_warmup", 1, "optim.accelerator", "cpu"],
    )
    datasets = graph_store_datasets(cfg, _adata)
    model = LocalModel(datasets[0], cfg)
    dm = GraphAnnDataModule(datas=datasets, batch_size=2, num_workers=0, mask_percentage=0.3, seed=0)

    model.train(max_epochs=2, datamodule=dm, early_stopping=False, enable_progress_bar=False)

    assert model.is_trained_
