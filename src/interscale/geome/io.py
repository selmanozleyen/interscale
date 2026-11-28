"""Grouped AnnData: an ordinary AnnData whose rows are sorted so that every group is one contiguous block.

Nothing here is a new file format. :func:`write_grouped` writes a regular zarr AnnData -- ``ad.read_zarr``,
scanpy and squidpy open it -- with three conventions on top:

- rows are ordered by ``(split, group)``; within a group they keep their source order;
- ``obsp["spatial_connectivities"]`` is squidpy's neighbour graph, computed per group, so it is
  block-diagonal and a group's edges are the same rows of the CSR;
- ``uns["geome"]`` records the group boundaries (``group_ptr``), names and splits.

:class:`GroupedAnnData` reads row ranges of such an AnnData. From a ``.zarr`` path it reads each array
once per batch (slice runs, :func:`interscale.geome._zarr.read_runs`); given an AnnData object --
in memory, backed (``.h5ad``) or lazy (``anndata.experimental.read_lazy``) -- it slices that object.

Staged in InterScale until it moves to ``geome.io``.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq
import zarr
from anndata import AnnData
from anndata.io import read_elem, sparse_dataset, write_elem
from torch_geometric.utils import from_scipy_sparse_matrix
from zarr.core.sync import sync

from interscale.geome._zarr import Appender, gather, read_runs, runs

FORMAT_VERSION = 1
SPLITS = ("train", "val", "test")
CONNECTIVITIES = "spatial_connectivities"


# ---------------------------------------------------------------------------------- source


def _resolve(key: str) -> tuple[str, str | None]:
    """``"X"`` -> ("X", None); ``"layers/name"`` / ``"obsm/name"`` -> (slot, name)."""
    if key == "X":
        return "X", None
    slot, _, name = key.partition("/")
    if slot not in ("layers", "obsm", "obsp") or not name:
        raise ValueError(f"expected 'X', 'layers/<name>', 'obsm/<name>' or 'obsp/<name>', got {key!r}")
    return slot, name


class _Source:
    """Row access to an AnnData object, or to an ``.h5ad``/``.zarr`` file without loading its matrices."""

    def __init__(self, source: AnnData | str | os.PathLike):
        if isinstance(source, AnnData):
            self.adata, self.group = source, None
            self.obs, self.var = source.obs, source.var
        else:
            path = str(source)
            self.adata = None
            self.group = zarr.open_group(path, mode="r") if path.endswith(".zarr") else h5py.File(path, "r")
            self.obs, self.var = read_elem(self.group["obs"]), read_elem(self.group["var"])

    def element(self, key: str):
        slot, name = _resolve(key)
        if self.adata is not None:
            return self.adata.X if slot == "X" else getattr(self.adata, slot)[name]
        elem = self.group[key]
        return sparse_dataset(elem) if elem.attrs.get("encoding-type") in ("csr_matrix", "csc_matrix") else elem

    @staticmethod
    def is_sparse(elem) -> bool:
        return sp.issparse(elem) or type(elem).__name__ in ("_CSRDataset", "_CSCDataset")

    @staticmethod
    def rows(elem, rows: np.ndarray):
        """``elem[rows]`` as scipy CSR or numpy; ascending ``rows``, read as slices where they are contiguous."""
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        if len(breaks) < max(1, len(rows) // 64):  # mostly contiguous: one slice per run
            parts = [elem[int(r[0]) : int(r[-1]) + 1] for r in np.split(rows, breaks)]
            stack = sp.vstack if sp.issparse(parts[0]) else np.concatenate
            out = parts[0] if len(parts) == 1 else stack(parts)
        elif isinstance(elem, zarr.Array):
            out = elem.get_orthogonal_selection((rows, slice(None)))
        else:
            out = elem[rows]
        return out.tocsr() if sp.issparse(out) else np.asarray(out)

    def nonzero_rows(self, chunk: int = 65536) -> np.ndarray:
        """Rows whose ``X`` sums to non-zero, read ``chunk`` rows at a time."""
        x, n = self.element("X"), len(self.obs)
        return np.concatenate([np.asarray(x[a : a + chunk].sum(axis=1)).ravel() != 0 for a in range(0, n, chunk)])


def _groups(obs: pd.DataFrame, groupby: str, split: str | None, keep: np.ndarray):
    """``(split, group name, ascending rows)`` per group, ordered by split then category."""
    codes = obs[groupby].astype("category").cat.codes.to_numpy()
    names = obs[groupby].astype("category").cat.categories
    splits = obs[split].astype(str).to_numpy() if split else None
    for split_name in SPLITS if split else (None,):
        sel = keep & (codes >= 0) & (splits == split_name if split else True)
        rows = np.flatnonzero(sel)
        rows = rows[np.argsort(codes[rows], kind="stable")]
        for g in np.split(rows, np.flatnonzero(np.diff(codes[rows])) + 1):
            if len(g):
                yield split_name, str(names[codes[g[0]]]), g


# ---------------------------------------------------------------------------------- writer


class _CSRWriter:
    """A CSR matrix written row block by row block, in whole chunks, with AnnData's encoding."""

    def __init__(self, group: zarr.Group, name: str, shape: tuple[int, int], dtype, chunk: int):
        self.group = group.create_group(name)
        self.group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": list(shape)})
        index_dtype = np.int64  # scipy needs indices and indptr of one dtype, and nnz is unknown up front
        self.data = Appender(self.group.create_array("data", shape=(0,), chunks=(chunk,), dtype=dtype))
        self.indices = Appender(self.group.create_array("indices", shape=(0,), chunks=(chunk,), dtype=index_dtype))
        self.indptr, self.dtype, self.index_dtype = [np.zeros(1, np.int64)], dtype, index_dtype

    def add(self, block: sp.csr_matrix, col_offset: int = 0) -> None:
        self.data.add(block.data.astype(self.dtype, copy=False))
        self.indices.add((block.indices + col_offset).astype(self.index_dtype, copy=False))
        self.indptr.append(block.indptr[1:].astype(np.int64) + self.indptr[-1][-1])

    def close(self) -> None:
        self.data.flush()
        self.indices.flush()
        self.group.create_array("indptr", data=np.concatenate(self.indptr))


def _dense_writer(group: zarr.Group, name: str, n: int, width: int, dtype, chunk_rows: int, shard_rows):
    shards = None if shard_rows is None else (shard_rows, width)
    array = group.create_array(name, shape=(0, width), chunks=(chunk_rows, width), shards=shards, dtype=dtype)
    array.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    return Appender(array)


def _mapping(group: zarr.Group, name: str) -> zarr.Group:
    sub = group.require_group(name)
    sub.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
    return sub


def write_grouped(
    source: AnnData | str | os.PathLike,
    out: str | os.PathLike,
    *,
    groupby: str,
    neighbors: dict,
    split: str | None = None,
    matrices: Sequence[str] = ("X",),
    obs: Sequence[str] = (),
    obsm: Sequence[str] = (),
    row_filter: str | np.ndarray | None = None,
    chunk_rows: int = 1024,
    shard_rows: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``source`` as a grouped zarr AnnData: one contiguous block of rows per group.

    One pass. From a path, only ``obs``, ``var`` and ``obsm["spatial"]`` are loaded; every matrix is read
    a group's rows at a time, as slices where those rows are already contiguous in the source. Sparse
    matrices stay sparse. The AnnData is written to a temporary sibling directory and renamed into place.

    Parameters
    ----------
    source
        An AnnData (in memory or backed) or an ``.h5ad``/``.zarr`` path with ``obsm["spatial"]``.
    out
        Target ``.zarr`` directory.
    groupby
        ``obs`` column; each category becomes a group, i.e. one graph.
    neighbors
        :func:`squidpy.gr.spatial_neighbors` keyword arguments, applied to each group's coordinates.
    split
        Optional ``obs`` column with ``train``/``val``/``test``; groups are ordered by it first, and a
        group is the rows of one category within one split.
    matrices
        Matrices to keep: ``"X"`` and/or ``"layers/<name>"``.
    obs, obsm
        ``obs`` columns (beyond ``groupby``/``split``) and ``obsm`` keys (beyond ``spatial``) to keep.
    row_filter
        ``"nonzero"`` drops cells whose ``X`` row sums to zero (computed in chunks), or a boolean mask.
    chunk_rows, shard_rows
        Rows per chunk, and optionally per shard, of the dense arrays (CSR arrays are chunked by entries).
    overwrite
        Replace an existing ``out``.

    Returns
    -------
    Path
        ``out``.
    """
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it.")
    src = _Source(source)
    coords = np.asarray(src.element("obsm/spatial")) if src.adata is None else np.asarray(src.adata.obsm["spatial"])
    keep = (
        src.nonzero_rows()
        if isinstance(row_filter, str) and row_filter == "nonzero"
        else np.ones(len(src.obs), bool)
        if row_filter is None
        else np.asarray(row_filter, bool)
    )
    groups = list(_groups(src.obs, groupby, split, keep))
    order = np.concatenate([rows for *_, rows in groups])
    n = len(order)
    group_ptr = np.r_[0, np.cumsum([len(rows) for *_, rows in groups])]

    tmp = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        root = zarr.open_group(tmp, mode="w")
        root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        columns = list(dict.fromkeys([groupby, *([split] if split else []), *obs]))
        obs_out = src.obs.iloc[order][columns].copy()
        obs_out["source_row"] = order
        write_elem(root, "obs", obs_out)
        write_elem(root, "var", src.var)
        for slot in ("layers", "obsm", "obsp", "varm", "varp"):
            _mapping(root, slot)

        writers = {}
        for key in matrices:
            slot, name = _resolve(key)
            elem = src.element(key)
            parent, leaf = (root, "X") if slot == "X" else (root[slot], name)
            if src.is_sparse(elem):
                writers[key] = ("csr", elem, _CSRWriter(parent, leaf, (n, elem.shape[1]), elem.dtype, chunk_rows * 64))
            else:
                writers[key] = (
                    "dense",
                    elem,
                    _dense_writer(parent, leaf, n, elem.shape[1], elem.dtype, chunk_rows, shard_rows),
                )
        dense_obsm = {"spatial": coords, **{k: src.element(f"obsm/{k}") for k in obsm}}
        obsm_writers = {
            k: (v, _dense_writer(root["obsm"], k, n, v.shape[1], v.dtype, chunk_rows, shard_rows))
            for k, v in dense_obsm.items()
        }
        conn = _CSRWriter(root["obsp"], CONNECTIVITIES, (n, n), np.float32, chunk_rows * 64)

        for (_, _, rows), start in zip(groups, group_ptr[:-1], strict=True):
            for kind, elem, writer in writers.values():
                block = _Source.rows(elem, rows)
                writer.add(sp.csr_matrix(block) if kind == "csr" else block)
            for elem, writer in obsm_writers.values():
                writer.add(_Source.rows(elem, rows) if not isinstance(elem, np.ndarray) else elem[rows])
            block, _ = sq.gr.spatial_neighbors(AnnData(obsm={"spatial": coords[rows]}), copy=True, **neighbors)
            conn.add(block.tocsr().astype(np.float32), col_offset=int(start))
        for kind, _, writer in writers.values():
            writer.close() if kind == "csr" else writer.flush()
        for _, writer in obsm_writers.values():
            writer.flush()
        conn.close()

        write_elem(
            root,
            "uns",
            {
                "geome": {
                    "format_version": FORMAT_VERSION,
                    "groupby": groupby,
                    "split": split or "",
                    "group_ptr": group_ptr.astype(np.int64),
                    "group_names": np.asarray([name for _, name, _ in groups], dtype=object),
                    "group_split": np.asarray([s or "" for s, _, _ in groups], dtype=object),
                },
                "spatial_neighbors": {"connectivities_key": CONNECTIVITIES, "params": dict(neighbors)},
            },
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if overwrite and out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------------- reader


class _ZarrCSR:
    """CSR rows of a zarr AnnData: ``indptr`` held in memory, ``data``/``indices`` read as slice runs."""

    def __init__(self, group: zarr.Group):
        self.data, self.indices = group["data"], group["indices"]
        self.indptr = group["indptr"][:]
        self.width = int(group.attrs["shape"][1])

    def nnz_spans(self, spans):
        return [(int(self.indptr[a]), int(self.indptr[b])) for a, b in spans]


def read_lazy(path: str | os.PathLike) -> AnnData:
    """Open a grouped ``.zarr`` AnnData with the small parts in memory and the matrices lazy.

    ``obs``, ``var``, ``uns`` and ``obsm`` are read into memory -- pandas, dicts and numpy, as
    ``ad.read_zarr`` gives them. ``X``, ``layers`` and ``obsp`` are lazy dask arrays
    (:func:`anndata.experimental.read_lazy`), so opening costs memory for the small parts only.
    The object remembers its store (``uns["geome"]["store"]``, in memory only), and
    :class:`GroupedAnnData` / :class:`~interscale.geome.data.GroupDataset` read its matrices from the
    zarr arrays directly -- as fast as passing the path -- rather than through dask.
    """
    from anndata.experimental import read_lazy as anndata_read_lazy

    group = zarr.open_group(path, mode="r")
    lazy = anndata_read_lazy(group)
    uns = read_elem(group["uns"])
    uns["geome"]["store"] = str(Path(path).resolve())
    return AnnData(
        X=lazy.X,
        obs=read_elem(group["obs"]),
        var=read_elem(group["var"]),
        uns=uns,
        obsm={key: read_elem(group["obsm"][key]) for key in group["obsm"]},
        layers={key: lazy.layers[key] for key in lazy.layers},
        obsp={key: lazy.obsp[key] for key in lazy.obsp},
    )


class GroupedAnnData:
    """Row-range reads from an AnnData written by :func:`write_grouped`.

    Parameters
    ----------
    source
        A ``.zarr`` path or an AnnData from :func:`read_lazy`: matrices are read from the zarr arrays
        directly, each array once per batch. Any other AnnData object -- in memory, backed (``.h5ad``,
        where anndata keeps only ``X`` on disk) or from anndata's own ``read_lazy`` (slow: every
        slice goes through dask) -- is sliced.
    """

    def __init__(self, source: AnnData | str | os.PathLike):
        self.adata = self.group = None
        if isinstance(source, AnnData):
            obs = source.obs
            self.obs = obs.to_memory() if hasattr(obs, "to_memory") else obs
            self.var_names = list(map(str, source.var_names))
            # anndata's read_lazy keeps even uns arrays lazy (dask): materialise the small metadata
            meta = {k: v.compute() if hasattr(v, "compute") else v for k, v in source.uns["geome"].items()}
            if meta.get("store"):  # from read_lazy: read the matrices from the store itself
                self.group = zarr.open_group(meta["store"], mode="r")
            else:
                self.adata = source
        elif Path(source).suffix == ".zarr":
            self.group = zarr.open_group(source, mode="r")
            self.obs = read_elem(self.group["obs"])
            self.var_names = list(map(str, read_elem(self.group["var"]).index))
            meta = read_elem(self.group["uns/geome"])
        else:  # an .h5ad: slice a backed AnnData
            import anndata as ad

            self.adata = ad.read_h5ad(source, backed="r")
            self.obs, self.var_names = self.adata.obs, list(map(str, self.adata.var_names))
            meta = dict(self.adata.uns["geome"])
        if int(meta["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"grouped AnnData format {meta['format_version']}, expected {FORMAT_VERSION}.")
        self.group_ptr = np.asarray(meta["group_ptr"], dtype=np.int64)
        self.groups = pd.DataFrame(
            {
                "name": [str(v) for v in meta["group_names"]],
                "split": [str(v) for v in meta["group_split"]],
                "start": self.group_ptr[:-1],
                "stop": self.group_ptr[1:],
            }
        )
        self._cache = {}

    def _zarr(self, key: str):
        if key not in self._cache:
            elem = self.group[key]
            sparse = elem.attrs.get("encoding-type") == "csr_matrix"
            self._cache[key] = _ZarrCSR(elem) if sparse else elem
        return self._cache[key]

    def read(self, keys: Sequence[str], group_ids: np.ndarray) -> dict[str, list]:
        """Each key's rows for each group in ``group_ids`` (ascending): numpy or CSR blocks per group.

        ``"obsp/<name>"`` keys come back as group-local ``edge_index`` arrays ``[2, E]``.
        """
        lo, hi = self.group_ptr[group_ids], self.group_ptr[group_ids + 1]
        lengths = hi - lo
        if self.adata is not None:
            return {key: [self._slice(key, int(a), int(b)) for a, b in zip(lo, hi, strict=True)] for key in keys}
        spans = runs(lo, hi)
        coroutines, plan = [], []
        for key in keys:
            elem = self._zarr(key)
            if isinstance(elem, zarr.Array):
                plan.append((key, "dense", len(coroutines)))
                coroutines.append(read_runs(elem, spans, 0))
            else:
                nnz = elem.nnz_spans(spans)
                plan.append((key, "csr", len(coroutines)))
                coroutines += [read_runs(elem.data, nnz, 0), read_runs(elem.indices, nnz, 0)]
        results = sync(gather(coroutines))
        out = {}
        for key, kind, i in plan:
            if kind == "dense":
                out[key] = np.split(results[i], np.cumsum(lengths)[:-1])
                continue
            elem, data, indices = self._zarr(key), results[i], results[i + 1]
            blocks, offset = [], 0
            for a, b in zip(lo, hi, strict=True):
                ptr = elem.indptr[a : b + 1] - elem.indptr[a]
                size = int(ptr[-1])
                width = int(b - a) if key.startswith("obsp/") else elem.width
                cols = indices[offset : offset + size] - (a if key.startswith("obsp/") else 0)
                blocks.append(sp.csr_matrix((data[offset : offset + size], cols, ptr), shape=(int(b - a), width)))
                offset += size
            out[key] = [self._edges(m) for m in blocks] if key.startswith("obsp/") else blocks
        return out

    def _slice(self, key: str, a: int, b: int):
        slot, name = _resolve(key)
        elem = self.adata.X if slot == "X" else getattr(self.adata, slot)[name]
        block = elem[a:b]
        block = block.compute() if hasattr(block, "compute") else block  # lazy (dask)
        if slot == "obsp":  # block-diagonal: shift columns to group-local ids, keeping the edge order
            m = sp.csr_matrix(block)
            return self._edges(sp.csr_matrix((m.data, m.indices - a, m.indptr), shape=(b - a, b - a)))
        return block.tocsr() if sp.issparse(block) else np.asarray(block)

    @staticmethod
    def _edges(block: sp.csr_matrix) -> np.ndarray:
        return from_scipy_sparse_matrix(block)[0].numpy()
