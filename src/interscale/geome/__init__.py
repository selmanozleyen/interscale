"""Staging area for geome: grouped AnnData I/O, a graph dataset and a DataModule over it.

Kept inside InterScale until it moves to the geome package (``geome.io`` / ``geome.data``); nothing
here depends on InterScale's config, models or masking.
"""

from .data import GroupDataset
from .datamodule import EpochCallback, GroupDataModule, item_generator
from .io import GroupedAnnData, read_lazy, write_grouped

__all__ = [
    "EpochCallback",
    "GroupDataModule",
    "GroupDataset",
    "GroupedAnnData",
    "item_generator",
    "read_lazy",
    "write_grouped",
]
