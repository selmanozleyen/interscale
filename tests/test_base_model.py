"""Models are built from the PyG data contract (`interscale.data`), not from an AnnData registry."""

import numpy as np
import pytest
import torch

from interscale.data import InMemoryGraphDataset, build_graph_datasets
from interscale.model.local_model import LocalModel
from interscale.tl import prepare_geome_dataset

GENES = [f"g{i}" for i in range(6)]
CLASSES = ["A", "B", "C"]


@pytest.fixture
def classification_dataset(make_graph):
    # Graph majorities: A, C, A.
    graphs = [
        make_graph(3, labels=[0, 0, 1]),
        make_graph(4, labels=[2, 2, 2, 1]),
        make_graph(5, labels=[0, 1, 0, 0, 2]),
    ]
    return InMemoryGraphDataset(graphs, var_names=GENES, class_names=CLASSES)


def _balanced(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return counts.sum() / (len(counts) * counts)  # sklearn's compute_class_weight("balanced")


# ----------------------------------------------------------------------- contract metadata


def test_model_reads_dimensions_and_classes_from_the_dataset(make_cfg, classification_dataset):
    model = LocalModel(classification_dataset, make_cfg())

    assert (model.n_input, model.n_output) == (len(GENES), len(CLASSES))
    assert model.class_labels == CLASSES
    assert model.var_names == GENES
    assert model.class_weights is None


def test_regression_model_outputs_every_gene(make_cfg, make_graph):
    model = LocalModel(InMemoryGraphDataset([make_graph(4)], var_names=GENES), make_cfg(task="regression"))

    assert model.n_input == model.n_output == len(GENES)
    assert model.class_labels is None


def test_classification_needs_class_names(make_cfg, make_graph):
    with pytest.raises(ValueError, match="class_names"):
        LocalModel(InMemoryGraphDataset([make_graph(2, labels=[0, 1])], var_names=GENES), make_cfg())


def test_num_classes_does_not_load_the_graphs(make_graph):
    """PyG's default `num_classes` loads every graph -- a full pass over an on-disk dataset."""

    class _Exploding(InMemoryGraphDataset):
        def get(self, idx):
            raise AssertionError("metadata must not touch the data")

    dataset = _Exploding([make_graph(1, labels=[0])], var_names=GENES, class_names=CLASSES)

    assert (dataset.num_classes, dataset.num_node_features) == (len(CLASSES), len(GENES))


def test_in_memory_dataset_validates_widths_against_metadata(make_graph):
    with pytest.raises(ValueError, match="one-hot y"):
        InMemoryGraphDataset([make_graph(2, labels=[0, 1])], var_names=GENES, class_names=CLASSES[:2])
    with pytest.raises(ValueError, match="features"):
        InMemoryGraphDataset([make_graph(1, labels=[0])], var_names=GENES[:-1], class_names=CLASSES)


# --------------------------------------------------------------------------- class weights


def test_node_level_weights_count_cells(make_cfg, classification_dataset):
    model = LocalModel(classification_dataset, make_cfg(loss="WeightedCE"))

    # cells: A = 2 + 0 + 3, B = 1 + 1 + 1, C = 0 + 3 + 1
    np.testing.assert_allclose(model.class_weights.numpy(), _balanced([5, 3, 4]), rtol=1e-6)


def test_graph_level_weights_count_graphs_by_majority_label(make_cfg, make_graph, classification_dataset):
    labels = [[0, 0, 1], [2, 2, 1], [0, 0, 2], [1, 1, 0]]  # majorities A, C, A, B
    dataset = InMemoryGraphDataset([make_graph(3, labels=y) for y in labels], var_names=GENES, class_names=CLASSES)
    cfg = make_cfg(level="graph", loss="WeightedCE")

    np.testing.assert_allclose(LocalModel(dataset, cfg).class_weights.numpy(), _balanced([2, 1, 1]), rtol=1e-6)
    # classification_dataset's majorities are A, C, A: no B graph, so the weights are undefined.
    with pytest.raises(ValueError, match="no training units"):
        LocalModel(classification_dataset, cfg)


def test_weights_match_the_previous_obs_based_computation(make_cfg, make_spatial_adata):
    """Same numbers as the AnnData-based weights this replaces: train-split cells per class."""
    adata = make_spatial_adata()
    cfg = make_cfg(loss="WeightedCE")
    counts = adata.obs.loc[adata.obs["split"] == "train", "ct"].astype(str).value_counts().reindex(CLASSES)

    model = LocalModel(build_graph_datasets(adata, cfg)[0], cfg)

    np.testing.assert_allclose(model.class_weights.numpy(), _balanced(counts), rtol=1e-6)


# --------------------------------------------------------------------------- build from AnnData


def test_build_graph_datasets_wraps_prepare_geome_dataset(make_cfg, make_spatial_adata):
    cfg = make_cfg()

    datasets = build_graph_datasets(make_spatial_adata(), cfg)
    reference, _ = prepare_geome_dataset(make_spatial_adata(), cfg)

    assert [len(d) for d in datasets] == [len(r) for r in reference] == [3, 1, 1]
    for dataset, graphs in zip(datasets, reference, strict=True):
        assert (dataset.var_names, dataset.class_names) == (GENES, CLASSES)
        for i, graph in enumerate(graphs):
            for key in ("x", "edge_index", "y"):
                assert torch.equal(dataset[i][key], graph[key])


# ----------------------------------------------------------------------------- save / load


def test_save_load_round_trip_and_metadata_check(make_cfg, classification_dataset):
    cfg = make_cfg()
    model = LocalModel(classification_dataset, cfg)
    model.save()

    loaded = LocalModel.load(cfg.model.save, classification_dataset, cfg)
    for key, value in model.module.state_dict().items():
        assert torch.equal(value, loaded.module.state_dict()[key])

    graphs = classification_dataset.data_list
    with pytest.raises(ValueError, match="var_names"):
        LocalModel.load(cfg.model.save, InMemoryGraphDataset(graphs, GENES[::-1], CLASSES), cfg)
    with pytest.raises(ValueError, match="class_labels"):
        LocalModel.load(cfg.model.save, InMemoryGraphDataset(graphs, GENES, ["A", "C", "B"]), cfg)


def test_checkpoint_without_saved_metadata_still_loads(make_cfg, classification_dataset, tmp_path):
    """Checkpoints written before the metadata was saved carry only the state dict."""
    from interscale.model.base._base_model import SAVE_KEYS

    cfg = make_cfg()
    LocalModel(classification_dataset, cfg).save()
    (path,) = (tmp_path / "models").glob(f"*{SAVE_KEYS.MODEL_FNAME}")
    state = torch.load(path)[SAVE_KEYS.MODEL_STATE_DICT_KEY]
    torch.save({SAVE_KEYS.MODEL_STATE_DICT_KEY: state}, path)

    LocalModel.load(cfg.model.save, classification_dataset, cfg)


def test_get_model_output_rejects_an_adata_with_other_genes(make_cfg, classification_dataset, make_spatial_adata):
    model = LocalModel(classification_dataset, make_cfg())
    model.is_trained_ = True
    adata = make_spatial_adata()
    adata.var_names = [f"other{i}" for i in range(len(GENES))]

    with pytest.raises(ValueError, match="var_names"):
        model.get_model_output(adata)
