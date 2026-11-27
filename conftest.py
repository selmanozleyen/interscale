"""Shared test factories. Fixtures return builders, since tests need several variants each."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule

GENES = [f"g{i}" for i in range(6)]
CLASSES = ["A", "B", "C"]


@pytest.fixture
def make_graph():
    """``make_graph(n, n_features=6, labels=None)``: an edgeless graph, one-hot ``y`` if labelled."""

    def make(num_nodes: int, n_features: int = len(GENES), labels=None) -> Data:
        data = Data(x=torch.randn(num_nodes, n_features), edge_index=torch.zeros((2, 0), dtype=torch.long))
        if labels is not None:
            data.y = torch.nn.functional.one_hot(torch.tensor(labels), len(CLASSES)).float()
        return data

    return make


@pytest.fixture
def make_datamodule(make_graph):
    """``make_datamodule(**kwargs)``: 3 train / 1 val / 1 test graphs, kwargs to the datamodule."""

    def make(n_features: int = 3, **kwargs) -> GraphAnnDataModule:
        sizes = [[20, 20, 20], [10], [10]]
        datas = [[make_graph(n, n_features) for n in split] for split in sizes]
        return GraphAnnDataModule(datas, **{"batch_size": 1, "num_workers": 0, **kwargs})

    return make


@pytest.fixture
def make_cfg(tmp_path):
    """``make_cfg(task=, level=, loss=, sample_key=, extra=)``: a GCN/transformer test config."""
    # Component names select which component defaults load_config merges, so they come from a file.
    components = tmp_path / "components.yaml"
    components.write_text(
        "model:\n  local_component:\n    name: GCN\n  global_component:\n    name: self-attn-transformer\n"
    )

    def make(task="classification", level="node", loss=None, sample_key=("sample",), extra=()):
        overrides = [
            "dataset.sample_key", list(sample_key),
            "dataset.prediction_task", task,
            "dataset.prediction_level", level,
            "dataset.name", "unit",
            "dataset.graph_store", str(tmp_path / "stores"),
            "optim.loss", loss or ("CrossEntropy" if task == "classification" else "MSELoss"),
            "model.n_embed", 4,
            "model.save", str(tmp_path / "models"),
            "wandb.use", False,
        ]  # fmt: skip
        if task == "classification":
            overrides += ["dataset.prediction_obs", "ct"]
        if level == "graph":
            overrides += ["dataset.mask_percentage", 0.0]  # _validate_masking: graph tasks are unmasked
        return load_config(components, overrides=[*overrides, *extra])

    return make


@pytest.fixture
def make_spatial_adata():
    """``make_spatial_adata(seed=0, sparse=False)``: 5 samples x 12 cells, split 3/1/1 by sample.

    Rows are shuffled so graph order differs from row order. Has a ``log1p`` layer, a ``latent``
    obsm, a ``fov`` second sample key and a ``ct`` label.
    """

    def make(seed: int = 0, sparse: bool = False) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        samples, n_per = ["s1", "s2", "s3", "s4", "s5"], 12
        n = n_per * len(samples)
        x = rng.poisson(1.0, size=(n, len(GENES))).astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(x) if sparse else x)
        adata.var_names = GENES
        adata.layers["log1p"] = sp.csr_matrix(np.log1p(x)) if sparse else np.log1p(x)
        adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
        adata.obsm["latent"] = rng.normal(size=(n, 3)).astype(np.float32)
        adata.obs["sample"] = pd.Categorical(np.repeat(samples, n_per))
        adata.obs["fov"] = pd.Categorical(rng.choice(["f1", "f2"], size=n))
        adata = adata[rng.permutation(n)].copy()
        split = {"s1": "train", "s2": "train", "s3": "train", "s4": "val", "s5": "test"}
        adata.obs["split"] = pd.Categorical(adata.obs["sample"].map(split).astype(str))
        adata.obs["ct"] = pd.Categorical(rng.choice(CLASSES, size=n))
        return adata

    return make
