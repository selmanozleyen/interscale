import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from interscale.data import InMemoryGraphDataset, MaskedGraphDataset
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.train._utils import NodeMaskResampleCallback


def _make_data(num_nodes: int, num_features: int = 3) -> Data:
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def _build_datamodule(mask_percentage: float = 0.5, learning_type: str = "node") -> GraphAnnDataModule:
    train_data = [_make_data(20), _make_data(20), _make_data(20)]
    val_data = [_make_data(10)]
    test_data = [_make_data(10)]
    return GraphAnnDataModule(
        datas=[train_data, val_data, test_data],
        batch_size=1,
        num_workers=0,
        mask_percentage=mask_percentage,
        learning_type=learning_type,
    )


def _masks(dataset) -> list[torch.Tensor]:
    return [dataset[i].mask for i in range(len(dataset))]


def test_plain_lists_are_wrapped_in_the_contract():
    dm = _build_datamodule()

    assert isinstance(dm.train_data, InMemoryGraphDataset)
    assert dm.train_data.num_node_features == 3


def test_train_mask_is_stable_within_an_epoch_and_changes_across_epochs():
    dm = _build_datamodule()

    epoch0 = _masks(dm.train_masked)
    assert all(torch.equal(a, b) for a, b in zip(epoch0, _masks(dm.train_masked), strict=True))

    dm.set_epoch(1)
    epoch1 = _masks(dm.train_masked)
    assert any(not torch.equal(a, b) for a, b in zip(epoch0, epoch1, strict=True))

    dm.set_epoch(0)
    assert all(torch.equal(a, b) for a, b in zip(epoch0, _masks(dm.train_masked), strict=True))


def test_val_and_test_masks_do_not_depend_on_the_epoch():
    dm = _build_datamodule()
    val, test = _masks(dm.val_masked), _masks(dm.test_masked)

    for epoch in range(1, 5):
        dm.set_epoch(epoch)
        assert all(torch.equal(a, b) for a, b in zip(val, _masks(dm.val_masked), strict=True))
        assert all(torch.equal(a, b) for a, b in zip(test, _masks(dm.test_masked), strict=True))


def test_stored_graphs_are_never_mutated():
    dm = _build_datamodule()
    stored = dm.train_data.data_list[0]
    x_before = stored.x.clone()

    item = dm.train_masked[0]

    assert "mask" not in stored
    assert item is not stored
    assert item.x.data_ptr() == stored.x.data_ptr(), "items must share the stored tensors, not copy them"
    assert torch.equal(stored.x, x_before)


def test_graph_level_eval_splits_are_not_masked():
    dm = _build_datamodule(mask_percentage=0.5, learning_type="graph")

    assert dm.train_masked[0].mask.any()
    assert not dm.val_masked[0].mask.any()
    assert not dm.test_masked[0].mask.any()


def test_the_epoch_reaches_worker_processes():
    """Workers get a pickled copy of the dataset when the epoch's iterator is created, so the
    masks they draw must equal the ones drawn in the main process for the same epoch."""
    graphs = [_make_data(30) for _ in range(6)]
    masked = MaskedGraphDataset(InMemoryGraphDataset(graphs, var_names=["a", "b", "c"]), resample=True, seed=7)
    masked.epoch = 3
    expected = _masks(masked)

    loader = DataLoader(masked, batch_size=1, shuffle=False, num_workers=2)
    received = [batch.mask for batch in loader]

    assert all(torch.equal(a, b) for a, b in zip(expected, received, strict=True))


def test_node_mask_resample_callback_sets_the_trainer_epoch():
    dm = _build_datamodule()

    class _StubTrainer:
        datamodule = dm
        current_epoch = 4

    NodeMaskResampleCallback().on_train_epoch_start(_StubTrainer(), pl_module=None)

    assert dm.train_masked.epoch == 4
