"""Models are built from the PyG data contract (`interscale.data`), not from an AnnData registry."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from torch_geometric.data import Data

from interscale.config import load_config
from interscale.data import InMemoryGraphDataset, build_graph_datasets
from interscale.model.local_model import LocalModel
from interscale.tl import prepare_geome_dataset

GENES = [f"g{i}" for i in range(5)]
CLASSES = ["A", "B", "C"]


@pytest.fixture
def components_yaml(tmp_path):
    # Component names select which component defaults load_config merges, so they have to come
    # from a file rather than from overrides.
    path = tmp_path / "components.yaml"
    path.write_text("model:\n  local_component:\n    name: GCN\n  global_component:\n    name: self-attn-transformer\n")
    return path


def _cfg(components_yaml, tmp_path, *, task="classification", level="node", loss="WeightedCE"):
    overrides = [
        "dataset.sample_key", ["sample"],
        "dataset.prediction_task", task,
        "dataset.prediction_level", level,
        "dataset.name", "unit",
        "optim.loss", loss,
        "model.n_embed", 4,
        "model.save", str(tmp_path / "models"),
        "wandb.use", False,
    ]  # fmt: skip
    if task == "classification":
        overrides += ["dataset.prediction_obs", "ct"]
    if level == "graph":
        overrides += ["dataset.mask_percentage", 0.0]  # _validate_masking: graph tasks are unmasked
    return load_config(components_yaml, overrides=overrides)


def _graph(labels: list[int], n_genes: int = len(GENES), n_classes: int = len(CLASSES)) -> Data:
    n = len(labels)
    return Data(
        x=torch.randn(n, n_genes),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        y=torch.nn.functional.one_hot(torch.tensor(labels), n_classes).float(),
    )


def _classification_dataset() -> InMemoryGraphDataset:
    # graph 0: majority A; graph 1: majority C; graph 2: majority A
    graphs = [_graph([0, 0, 1]), _graph([2, 2, 2, 1]), _graph([0, 1, 0, 0, 2])]
    return InMemoryGraphDataset(graphs, var_names=GENES, class_names=CLASSES)


def _spatial_adata(seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    samples = ["s1", "s2", "s3", "s4"]
    n_per = 15
    n = n_per * len(samples)
    adata = AnnData(X=rng.random((n, len(GENES)), dtype=np.float32))
    adata.var_names = GENES
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, n_per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "train", "val", "test"], n_per))
    adata.obs["ct"] = pd.Categorical(rng.choice(CLASSES, size=n))
    return adata


# ----------------------------------------------------------------------- contract metadata


def test_model_reads_dimensions_and_classes_from_the_dataset(components_yaml, tmp_path):
    model = LocalModel(_classification_dataset(), _cfg(components_yaml, tmp_path, loss="CrossEntropy"))

    assert model.n_input == len(GENES)
    assert model.n_output == len(CLASSES)
    assert model.class_labels == CLASSES
    assert model.var_names == GENES
    assert model.class_weights is None


def test_regression_model_outputs_every_gene(components_yaml, tmp_path):
    graphs = [Data(x=torch.randn(4, len(GENES)), edge_index=torch.zeros((2, 0), dtype=torch.long))]
    dataset = InMemoryGraphDataset(graphs, var_names=GENES)
    model = LocalModel(dataset, _cfg(components_yaml, tmp_path, task="regression", loss="MSELoss"))

    assert model.n_input == model.n_output == len(GENES)
    assert model.class_labels is None


def test_classification_needs_class_names(components_yaml, tmp_path):
    dataset = InMemoryGraphDataset([_graph([0, 1])], var_names=GENES)  # no class_names

    with pytest.raises(ValueError, match="class_names"):
        LocalModel(dataset, _cfg(components_yaml, tmp_path, loss="CrossEntropy"))


def test_num_classes_does_not_load_the_graphs():
    """PyG's default `num_classes` loads every graph -- a full pass over an on-disk dataset."""

    class _Exploding(InMemoryGraphDataset):
        def get(self, idx):
            raise AssertionError("metadata must not touch the data")

    dataset = _Exploding([_graph([0])], var_names=GENES, class_names=CLASSES)

    assert dataset.num_classes == len(CLASSES)
    assert dataset.num_node_features == len(GENES)


def test_in_memory_dataset_rejects_a_y_that_does_not_match_class_names():
    with pytest.raises(ValueError, match="one-hot y"):
        InMemoryGraphDataset([_graph([0, 1], n_classes=2)], var_names=GENES, class_names=CLASSES)


def test_in_memory_dataset_rejects_mismatched_var_names():
    with pytest.raises(ValueError, match="features"):
        InMemoryGraphDataset([_graph([0])], var_names=GENES[:-1], class_names=CLASSES)


# --------------------------------------------------------------------------- class weights


def _balanced(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return counts.sum() / (len(counts) * counts)  # sklearn's compute_class_weight("balanced")


def test_node_level_weights_count_cells(components_yaml, tmp_path):
    model = LocalModel(_classification_dataset(), _cfg(components_yaml, tmp_path, level="node"))

    # cells: A = 2 + 0 + 3, B = 1 + 1 + 1, C = 0 + 3 + 1
    np.testing.assert_allclose(model.class_weights.numpy(), _balanced([5, 3, 4]), rtol=1e-6)


def test_graph_level_weights_count_graphs_by_majority_label(components_yaml, tmp_path):
    dataset = InMemoryGraphDataset(
        [_graph([0, 0, 1]), _graph([2, 2, 1]), _graph([0, 0, 2]), _graph([1, 1, 0])],  # A, C, A, B
        var_names=GENES,
        class_names=CLASSES,
    )
    model = LocalModel(dataset, _cfg(components_yaml, tmp_path, level="graph"))

    np.testing.assert_allclose(model.class_weights.numpy(), _balanced([2, 1, 1]), rtol=1e-6)


def test_weights_fail_loudly_when_a_class_has_no_training_unit(components_yaml, tmp_path):
    # Graph majorities are A, C, A: no graph is a B, so graph-level WeightedCE is undefined.
    with pytest.raises(ValueError, match="no training units"):
        LocalModel(_classification_dataset(), _cfg(components_yaml, tmp_path, level="graph"))


def test_weights_match_the_previous_obs_based_computation(components_yaml, tmp_path):
    """Same numbers as the AnnData-based weights this replaces: train-split cells per class."""
    adata = _spatial_adata()
    cfg = _cfg(components_yaml, tmp_path)
    train_ct = adata.obs.loc[adata.obs["split"] == "train", "ct"].astype(str)
    counts = train_ct.value_counts().reindex(CLASSES).to_numpy(dtype=np.float64)
    expected = counts.sum() / (len(CLASSES) * counts)

    datasets = build_graph_datasets(adata, cfg)
    model = LocalModel(datasets[0], cfg)

    np.testing.assert_allclose(model.class_weights.numpy(), expected, rtol=1e-6)


# --------------------------------------------------------------------------- build from AnnData


def test_build_graph_datasets_wraps_prepare_geome_dataset(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path)

    datasets = build_graph_datasets(_spatial_adata(), cfg)
    reference, _ = prepare_geome_dataset(_spatial_adata(), cfg)

    assert [len(d) for d in datasets] == [len(r) for r in reference] == [2, 1, 1]
    for dataset, graphs in zip(datasets, reference, strict=True):
        assert dataset.var_names == GENES
        assert dataset.class_names == CLASSES
        for i, graph in enumerate(graphs):
            assert torch.equal(dataset[i].x, graph.x)
            assert torch.equal(dataset[i].edge_index, graph.edge_index)
            assert torch.equal(dataset[i].y, graph.y)


# ----------------------------------------------------------------------------- save / load


def test_save_load_round_trip_and_metadata_check(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path, loss="CrossEntropy")
    dataset = _classification_dataset()
    model = LocalModel(dataset, cfg)
    model.save()

    loaded = LocalModel.load(cfg.model.save, dataset, cfg, local_component=True)
    for key, value in model.module.state_dict().items():
        assert torch.equal(value, loaded.module.state_dict()[key])

    reordered = InMemoryGraphDataset(dataset.data_list, var_names=GENES[::-1], class_names=CLASSES)
    with pytest.raises(ValueError, match="var_names"):
        LocalModel.load(cfg.model.save, reordered, cfg, local_component=True)

    relabelled = InMemoryGraphDataset(dataset.data_list, var_names=GENES, class_names=["A", "C", "B"])
    with pytest.raises(ValueError, match="class_labels"):
        LocalModel.load(cfg.model.save, relabelled, cfg, local_component=True)


def test_checkpoint_without_saved_metadata_still_loads(components_yaml, tmp_path):
    """Checkpoints written before the metadata was saved carry only the state dict."""
    from interscale.model.base._base_model import SAVE_KEYS
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _cfg(components_yaml, tmp_path, loss="CrossEntropy")
    dataset = _classification_dataset()
    model = LocalModel(dataset, cfg)
    model.save()
    path = tmp_path / "models" / f"{get_model_filename_prefix(cfg, True, False)}{SAVE_KEYS.MODEL_FNAME}"
    checkpoint = torch.load(path)
    torch.save({SAVE_KEYS.MODEL_STATE_DICT_KEY: checkpoint[SAVE_KEYS.MODEL_STATE_DICT_KEY]}, path)

    LocalModel.load(cfg.model.save, dataset, cfg, local_component=True)


def test_get_model_output_rejects_an_adata_with_other_genes(components_yaml, tmp_path):
    cfg = _cfg(components_yaml, tmp_path, loss="CrossEntropy")
    model = LocalModel(_classification_dataset(), cfg)
    model.is_trained_ = True
    adata = _spatial_adata()
    adata.var_names = [f"other{i}" for i in range(len(GENES))]

    with pytest.raises(ValueError, match="var_names"):
        model.get_model_output(adata)
