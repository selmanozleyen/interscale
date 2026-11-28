"""One field, three spellings: anndata accessors, their ``.`` strings, and ``/`` storage paths.

The staged API takes a field as any of

- an :mod:`anndata.acc` accessor: ``A.X``, ``A.layers["counts"]``, ``A.obsm["spatial"]``,
  ``A.obsp["spatial_connectivities"]``, ``A.obs["cell_type"]``;
- its ``.`` string (:meth:`anndata.acc.AdAcc.resolve`): ``"X"``, ``"layers.counts"``, ``"obs.cell_type"``;
- its ``/`` path, the element's location inside a ``.zarr``/``.h5ad`` store: ``"X"``, ``"layers/counts"``,
  ``"obs/cell_type"``.

Readers of in-memory AnnData objects use the accessor (``acc.get(adata)``); readers of stores use the path.
:func:`as_acc` and :func:`to_path` convert either way; :func:`to_dotted` gives the ``.`` string.
"""

from __future__ import annotations

from anndata import AnnData
from anndata.acc import A, AdRef, GraphAcc, LayerAcc, MultiAcc

type Field = AdRef | LayerAcc | MultiAcc | GraphAcc | str
"""An accessor, a ``.`` string or a ``/`` path."""


def as_acc(field: Field) -> AdRef | LayerAcc | MultiAcc | GraphAcc:
    """The accessor for ``field``, whichever of the three spellings it uses."""
    if not isinstance(field, str):
        return field
    if "/" in field:
        return A.from_json([None if part == "" else part for part in field.split("/", 1)])
    return A.X if field == "X" else A.resolve(field)


def to_path(field: Field) -> str:
    """The ``/`` storage path of ``field``: ``"X"``, ``"layers/<k>"``, ``"obsm/<k>"``, ``"obsp/<k>"``, ``"obs/<col>"``."""
    parts = A.to_json(as_acc(field))
    if parts[:2] == ["layers", None]:
        return "X"
    if len(parts) != 2:
        raise ValueError(f"{field!r} refers to part of an array ({parts}); a whole array or obs column is needed.")
    return "/".join(map(str, parts))


def to_dotted(field: Field) -> str:
    """The ``.`` string of ``field``, as :meth:`anndata.acc.AdAcc.resolve` reads it."""
    path = to_path(field)
    return path if path == "X" else path.replace("/", ".", 1)


def get(adata: AnnData, field: Field):
    """The whole array (or obs column) ``field`` refers to in ``adata``."""
    return as_acc(field).get(adata)
