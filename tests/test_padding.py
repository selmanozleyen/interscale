"""pad_batch must give exactly what the per-graph loop it replaced gave: tensors, lists and gradients."""

import random

import pytest
import torch

from interscale.tl.padding import _select_masked_nodes, pad_batch


def _pad_batch_loop(x, batch_idx, max_seq_len, get_mask=False, keep_indices=None):
    """The previous implementation, kept verbatim as the reference."""
    num_batch = int(batch_idx[-1].item() + 1)
    num_nodes, masks = [], []
    for i in range(num_batch):
        mask = batch_idx.eq(i)
        num_nodes.append(mask.sum().item())
        masks.append(mask)
    max_num_nodes = min(max(num_nodes), max_seq_len)
    padded_x = x.data.new(max_num_nodes, num_batch, x.size(-1)).fill_(0)
    src_padding_mask = x.data.new(num_batch, max_num_nodes).fill_(0).bool()
    index_nodes = []
    for i, mask in enumerate(masks):
        num_nodes_i = num_nodes[i]
        if num_nodes_i > max_num_nodes:
            if get_mask:
                idx_nodes = _select_masked_nodes(keep_indices, mask, num_nodes_i, max_seq_len)
            else:
                idx_nodes = random.sample(range(0, num_nodes_i), max_seq_len)
            idx_nodes.sort()
            num_nodes_i = max_num_nodes
            padded_x[-num_nodes_i:, i] = x[mask][idx_nodes]
        else:
            idx_nodes = list(range(0, num_nodes_i))
            padded_x[-num_nodes_i:, i] = x[mask][-num_nodes_i:]
        src_padding_mask[i, : max_num_nodes - num_nodes_i] = True
        index_nodes.append(idx_nodes)
    if get_mask:
        return padded_x, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes
    return padded_x, src_padding_mask, index_nodes, num_nodes, None, max_num_nodes


@pytest.mark.parametrize("get_mask", [False, True])
@pytest.mark.parametrize("max_seq_len", [5, 12, 64])
@pytest.mark.parametrize("seed", range(4))
def test_matches_the_per_graph_loop(get_mask, max_seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1, 20, (7,), generator=g).tolist()  # some above max_seq_len 5 and 12
    batch_idx = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    keep = torch.rand(batch_idx.numel(), generator=g) < 0.3 if get_mask else None
    x = torch.randn(batch_idx.numel(), 4, generator=g)
    w = torch.randn(max(min(max(sizes), max_seq_len), 1), len(sizes), 4, generator=g)

    outs = []
    for fn in (_pad_batch_loop, pad_batch):
        xi = x.clone().requires_grad_(True)
        random.seed(seed)  # oversized graphs subsample with `random`: same draws in the same order
        out = fn(xi, batch_idx, max_seq_len, get_mask=get_mask, keep_indices=keep)
        (out[0] * w[: out[0].shape[0]]).sum().backward()
        outs.append((out, xi.grad))

    (ref, ref_grad), (new, new_grad) = outs
    assert torch.equal(ref[0], new[0]) and torch.equal(ref[1], new[1])
    assert ref[2] == new[2] and ref[3] == new[3] and ref[5] == new[5]
    assert (ref[4] is None) == (new[4] is None)
    if get_mask:
        assert all(torch.equal(a, b) for a, b in zip(ref[4], new[4], strict=True))
    assert torch.equal(ref_grad, new_grad)


def test_keeps_node_order_for_unsorted_batch_index():
    batch_idx = torch.tensor([1, 0, 1, 0, 1])
    x = torch.arange(5.0)[:, None]
    ref, new = _pad_batch_loop(x, batch_idx, 8), pad_batch(x, batch_idx, 8)
    assert torch.equal(ref[0], new[0]) and torch.equal(ref[1], new[1])
