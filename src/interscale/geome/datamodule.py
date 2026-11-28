"""A Lightning DataModule over a grouped AnnData: one graph per group, one split per loader.

Staged in InterScale until it moves to ``geome.data``.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Iterator, Mapping

import lightning.pytorch as pl
import numpy as np
import torch
from anndata import AnnData
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader

from interscale.geome.data import GroupDataset
from interscale.geome.io import SPLITS, GroupedAnnData


def item_generator(seed: int, idx: int, epoch: int | None = None) -> torch.Generator:
    """A generator that depends only on ``(seed, epoch, idx)`` -- or ``(seed, idx)`` for ``epoch=None``,
    i.e. the same draw every epoch -- whichever process draws from it.
    """
    words = [int(seed), int(idx)] if epoch is None else [int(seed), int(epoch), int(idx)]
    return torch.Generator().manual_seed(int(np.random.SeedSequence(words).generate_state(1, np.uint64)[0] >> 1))


class _Transformed(TorchDataset):
    """``dataset`` with ``transform(data, idx=, epoch=, split=)`` applied to every item."""

    def __init__(self, dataset: GroupDataset, transform: Callable | None, split: str):
        self.dataset, self.transform, self.split, self.epoch = dataset, transform, split, 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitems__(self, indices):
        items = self.dataset.__getitems__(indices)
        if self.transform is None:
            return items
        return [
            self.transform(d, idx=int(i), epoch=self.epoch, split=self.split)
            for d, i in zip(items, indices, strict=True)
        ]

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]


class _Prefetch:
    """Iterate ``loader`` on a background thread, ``depth`` batches ahead.

    Reading a batch mostly waits on zarr I/O and decoding, which release the GIL, so the next batch loads
    while the GPU runs the current step -- without worker processes, which are re-forked every epoch and
    which the zarrs codec pipeline cannot use (zarrs-python#171). Batches come out in ``loader``'s order.
    """

    def __init__(self, loader: DataLoader, depth: int = 2):
        self.loader, self.depth = loader, depth

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator:
        q, stop, end = queue.Queue(self.depth), threading.Event(), object()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def fill() -> None:
            try:
                for batch in self.loader:
                    if not put(batch):
                        return
                put(end)
            except BaseException as e:  # noqa: BLE001 -- handed to the consumer and re-raised there
                put(e)

        thread = threading.Thread(target=fill, daemon=True)
        thread.start()
        try:
            while (item := q.get()) is not end:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:  # the consumer stopped (sanity check, break, error): release the thread
            stop.set()
            thread.join()


class GroupDataModule(pl.LightningDataModule):
    """Train/val/test loaders over the groups of a grouped AnnData.

    Parameters
    ----------
    source
        A grouped ``.zarr`` path, a :class:`~interscale.geome.io.GroupedAnnData`, or an AnnData read from one.
    x, y, extra
        Passed to :class:`~interscale.geome.data.GroupDataset`.
    batch_size
        Graphs per batch.
    transform
        ``transform(data, *, idx, epoch, split) -> data`` applied to every graph, e.g. masking. Use
        :func:`item_generator` for randomness, so draws depend on ``(seed, epoch, idx)`` only.
    prefetch
        Batches loaded ahead on a background thread when ``num_workers == 0``; 0 disables it.
    pin_memory
        ``"auto"`` pins batches when CUDA is available.
    num_workers
        DataLoader worker processes; the default 0 uses the prefetch thread instead.
    """

    def __init__(
        self,
        source: GroupedAnnData | AnnData | str | os.PathLike,
        *,
        x: str = "X",
        y: str | None = None,
        extra: Mapping[str, str] | None = None,
        batch_size: int = 1,
        transform: Callable | None = None,
        prefetch: int = 2,
        pin_memory: bool | str = "auto",
        num_workers: int = 0,
    ):
        super().__init__()
        reader = source if isinstance(source, GroupedAnnData) else GroupedAnnData(source)
        present = [s for s in SPLITS if (reader.groups["split"] == s).any()]
        self.datasets = {s: GroupDataset(reader, x=x, y=y, extra=extra, split=s) for s in present}
        self._wrapped = {s: _Transformed(ds, transform, s) for s, ds in self.datasets.items()}
        self.batch_size, self.prefetch, self.num_workers = batch_size, prefetch, num_workers
        self.pin_memory = torch.cuda.is_available() if pin_memory == "auto" else bool(pin_memory)

    def set_epoch(self, epoch: int) -> None:
        """The epoch passed to ``transform``; takes effect at the next iterator."""
        for wrapped in self._wrapped.values():
            wrapped.epoch = int(epoch)

    def _loader(self, split: str, shuffle: bool):
        if split not in self._wrapped:
            raise RuntimeError(f"The grouped AnnData has no {split!r} groups.")
        loader = DataLoader(
            self._wrapped[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return _Prefetch(loader, self.prefetch) if self.num_workers == 0 and self.prefetch else loader

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)


class EpochCallback(pl.Callback):
    """Calls ``trainer.datamodule.set_epoch`` at every epoch start, before the epoch's iterator exists."""

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.set_epoch(trainer.current_epoch)
