"""zarr helpers shared by the graph store and the grouped AnnData: whole-unit appends, batched slice reads.

``read_runs`` uses zarr internals (``BasicIndexer``, ``Array._chunk_grid``, ``AsyncArray._get_selection``),
which is why zarr is pinned to the tested series.
"""

from __future__ import annotations

import asyncio

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import BasicIndexer, Indexer


class Appender:
    """Appends to a zarr array along ``axis`` in whole write units (shards, else chunks).

    Every unit is written exactly once: a partial write to a shard rewrites the whole shard file,
    so writing graph by graph into a sharded array would cost O(shard size) per graph.
    """

    def __init__(self, array: zarr.Array, axis: int = 0):
        self.array, self.axis, self.parts, self.size = array, axis, [], 0
        self.unit = (array.shards or array.chunks)[axis]

    def add(self, block: np.ndarray) -> None:
        self.parts.append(block)
        self.size += block.shape[self.axis]
        if self.size >= self.unit:
            self.flush(final=False)

    def flush(self, final: bool = True) -> None:
        if not self.parts:
            return
        data = np.concatenate(self.parts, axis=self.axis)
        n = data.shape[self.axis] if final else data.shape[self.axis] // self.unit * self.unit
        head, tail = np.split(data, [n], axis=self.axis)
        if n:
            self.array.append(head, axis=self.axis)
        self.parts, self.size = ([tail], tail.shape[self.axis]) if tail.shape[self.axis] else ([], 0)


def runs(lo: np.ndarray, hi: np.ndarray) -> list[tuple[int, int]]:
    """Non-empty ``[lo, hi)`` ranges in store order, with adjacent ones merged (consecutive graphs touch)."""
    if not len(lo):
        return []
    first = np.flatnonzero(np.r_[True, lo[1:] != hi[:-1]])
    last = np.r_[first[1:] - 1, len(lo) - 1]
    return [(int(a), int(b)) for a, b in zip(lo[first], hi[last], strict=True) if b > a]


class _Runs(Indexer):
    """Several slices along ``axis`` read back to back as one selection.

    annbatch's ``MultiBasicIndexer``, along any axis. Every chunk piece of it is a slice, which the
    zarrs codec pipeline serves; an integer selection with gaps inside a chunk makes zarrs fall back
    to zarr-python for the whole read (``UnsupportedVIndexingError``).
    """

    def __init__(self, array: zarr.Array, runs: list[tuple[int, int]], axis: int):
        self.axis, self.drop_axes = axis, ()
        self.indexers = []
        for lo, hi in runs:
            selection = [slice(None)] * array.ndim
            selection[axis] = slice(lo, hi)
            self.indexers.append(BasicIndexer(tuple(selection), shape=array.shape, chunk_grid=array._chunk_grid))
        shape = list(array.shape)
        shape[axis] = sum(hi - lo for lo, hi in runs)
        self.shape = tuple(shape)

    def __iter__(self):
        offset = 0
        for indexer in self.indexers:
            for projection in indexer:
                out = list(projection.out_selection)
                out[self.axis] = slice(offset + out[self.axis].start, offset + out[self.axis].stop)
                yield projection._replace(out_selection=tuple(out))
            offset += indexer.shape[self.axis]


async def gather(coroutines: list) -> list:
    # gather inside the coroutine sync() runs, so its future belongs to zarr's event loop
    return await asyncio.gather(*coroutines)


async def read_runs(array: zarr.Array, runs: list[tuple[int, int]], axis: int) -> np.ndarray:
    indexer = _Runs(array, runs, axis)
    out = np.empty(indexer.shape, dtype=array.dtype)
    if runs:
        prototype = default_buffer_prototype()
        await array._async_array._get_selection(indexer, prototype=prototype, out=prototype.nd_buffer(out))
    return out
