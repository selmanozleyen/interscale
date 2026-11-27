import torch
from torch_geometric.loader import DataLoader

from interscale.data import InMemoryGraphDataset, MaskedGraphDataset
from interscale.train._utils import NodeMaskResampleCallback


def _masks(dataset) -> list[torch.Tensor]:
    return [dataset[i].mask for i in range(len(dataset))]


def test_plain_lists_are_wrapped_in_the_contract(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5)

    assert isinstance(dm.train_data, InMemoryGraphDataset)
    assert dm.train_data.num_node_features == 3


def test_train_mask_is_stable_within_an_epoch_and_changes_across_epochs(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5)

    epoch0 = _masks(dm.train_masked)
    assert all(torch.equal(a, b) for a, b in zip(epoch0, _masks(dm.train_masked), strict=True))

    dm.set_epoch(1)
    epoch1 = _masks(dm.train_masked)
    assert any(not torch.equal(a, b) for a, b in zip(epoch0, epoch1, strict=True))

    dm.set_epoch(0)
    assert all(torch.equal(a, b) for a, b in zip(epoch0, _masks(dm.train_masked), strict=True))


def test_val_and_test_masks_do_not_depend_on_the_epoch(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5)
    val, test = _masks(dm.val_masked), _masks(dm.test_masked)

    for epoch in range(1, 5):
        dm.set_epoch(epoch)
        assert all(torch.equal(a, b) for a, b in zip(val, _masks(dm.val_masked), strict=True))
        assert all(torch.equal(a, b) for a, b in zip(test, _masks(dm.test_masked), strict=True))


def test_stored_graphs_are_never_mutated(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5)
    stored = dm.train_data.data_list[0]
    x_before = stored.x.clone()

    item = dm.train_masked[0]

    assert "mask" not in stored
    assert item is not stored
    assert item.x.data_ptr() == stored.x.data_ptr(), "items must share the stored tensors, not copy them"
    assert torch.equal(stored.x, x_before)


def test_graph_level_eval_splits_are_not_masked(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5, learning_type="graph")

    assert dm.train_masked[0].mask.any()
    assert not dm.val_masked[0].mask.any()
    assert not dm.test_masked[0].mask.any()


def test_the_epoch_reaches_worker_processes(make_graph):
    """Workers get a pickled copy of the dataset when the epoch's iterator is created, so the
    masks they draw must equal the ones drawn in the main process for the same epoch."""
    graphs = [make_graph(30, 3) for _ in range(6)]
    masked = MaskedGraphDataset(InMemoryGraphDataset(graphs, var_names=["a", "b", "c"]), resample=True, seed=7)
    masked.epoch = 3
    expected = _masks(masked)

    loader = DataLoader(masked, batch_size=1, shuffle=False, num_workers=2)
    received = [batch.mask for batch in loader]

    assert all(torch.equal(a, b) for a, b in zip(expected, received, strict=True))


def test_node_mask_resample_callback_sets_the_trainer_epoch(make_datamodule):
    dm = make_datamodule(mask_percentage=0.5)

    class _StubTrainer:
        datamodule = dm
        current_epoch = 4

    NodeMaskResampleCallback().on_train_epoch_start(_StubTrainer(), pl_module=None)

    assert dm.train_masked.epoch == 4
