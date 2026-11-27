"""An on-disk implementation of the graph-dataset contract, backed by a zarr store.

Every graph of every split is concatenated into flat arrays, with pointer arrays recording
where each graph starts -- the layout PyG's ``InMemoryDataset`` uses in memory, on disk::

    <store>.zarr/
      x            [N_total, G]  float32   cells; each graph is one contiguous block of rows
      label        [N_total]     int32     class code per cell (classification only)
      embeddings   [N_total, D]  float32   precomputed embeddings (only if configured)
      obs_index    [N_total]     int64     row of the cell in the source the store was built from
      edge_index   [2, E_total]  int32     node ids local to each graph
      node_ptr     [n_graphs+1]  int64     graph i owns rows node_ptr[i]:node_ptr[i+1]
      edge_ptr     [n_graphs+1]  int64     graph i owns edges edge_ptr[i]:edge_ptr[i+1]
      split        [n_graphs]    int8      index into SPLITS
      class_counts [n_graphs, C] int64     cells per class per graph (classification only)
      .attrs       var_names, class_names, fingerprint, complete

Reading a graph is one contiguous slice per array, and the model-facing metadata lives in
``attrs`` and ``class_counts``, so building a model reads no expression data.

The graphs are built as :func:`interscale.tl.prepare_geome_dataset` builds them -- one per
(sample key, split, category), neighbours from :func:`squidpy.gr.spatial_neighbors` on that
graph's cells, edges via :func:`torch_geometric.utils.from_scipy_sparse_matrix` -- but one graph at
a time, reading only that graph's rows of the expression matrix. Building from a file path never
loads the matrix: only ``obs``, ``var`` and the coordinates are held in memory.
``tests/test_zarr_store.py`` checks the result graph for graph against ``prepare_geome_dataset``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import scipy.sparse as sp
import squidpy as sq
import torch
import zarr
from anndata import AnnData
from anndata.io import read_elem, sparse_dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from yacs.config import CfgNode as CN

from interscale.data._graph_dataset import GraphDataset

#: Bump when the layout or the way graphs are built changes, so old stores stop matching.
STORE_VERSION = 2
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- fingerprint


def graph_store_fingerprint(cfg: CN) -> dict:
    """The config values the graphs depend on: equal fingerprints build identical stores.

    Training-only settings (masking, batch size, model, optimiser) are deliberately absent.
    """
    neighbors = {k: v for k, v in dict(cfg.dataset.spatial_neigbors_kwargs).items() if k != "library_key"}
    precomputed = cfg.model.global_component.get("parameters", CN()).get("type_gex_embedding") == "Precomputed"
    source = str(cfg.dataset.h5ad_data)
    stat = os.stat(source) if source and os.path.exists(source) else None
    fingerprint = {
        "store_version": STORE_VERSION,
        "source": source,
        # A changed input file must not silently reuse graphs built from the old one.
        "source_size": None if stat is None else stat.st_size,
        "source_mtime_ns": None if stat is None else stat.st_mtime_ns,
        "layer_key": cfg.dataset.layer_key,
        "sample_key": list(cfg.dataset.sample_key),
        "split_key": cfg.dataset.split_key,
        "prediction_task": cfg.dataset.prediction_task,
        "prediction_obs": cfg.dataset.prediction_obs if "classification" in cfg.dataset.prediction_task else None,
        "spatial_neighbors": neighbors,
        "embeddings_obsm_key": cfg.model.global_component.get("latent_obsm_key") if precomputed else None,
        "segmentation_robustness": None
        if cfg.dataset.segmentation_robustness is None
        else list(cfg.dataset.segmentation_robustness),
    }
    # Segmentation noise is random, so the graphs also depend on the seed that drew it.
    if cfg.dataset.segmentation_robustness is not None:
        fingerprint["seed"] = int(cfg.optim.seed)
    return fingerprint


def graph_store_path(store_dir: str | os.PathLike, cfg: CN) -> Path:
    """``<store_dir>/<dataset.name>-<fingerprint hash>.zarr``: one store per distinct graph config."""
    digest = hashlib.sha256(json.dumps(graph_store_fingerprint(cfg), sort_keys=True).encode()).hexdigest()[:16]
    return Path(store_dir) / f"{cfg.dataset.name or 'graphs'}-{digest}.zarr"


# --------------------------------------------------------------------------------- sources


class _Source:
    """Row access to an AnnData, or to an ``.h5ad``/``.zarr`` file without loading its matrices."""

    def __init__(self, source: AnnData | str | os.PathLike, layer_key: str | None, embeddings_key: str | None):
        if isinstance(source, AnnData):
            self.obs, self.var_names = source.obs, source.var_names
            self.coords = np.asarray(source.obsm["spatial"]) if "spatial" in source.obsm else None
            self._x = source.X if layer_key is None else source.layers[layer_key]
            self._counts = source.X
            self._embeddings = None if embeddings_key is None else source.obsm.get(embeddings_key)
            return
        path = str(source)
        group = zarr.open_group(path, mode="r") if path.endswith(".zarr") else h5py.File(path, "r")
        self.obs, self.var_names = read_elem(group["obs"]), read_elem(group["var"]).index
        self.coords = read_elem(group["obsm/spatial"]) if "obsm/spatial" in group else None
        self._x = self._matrix(group["X" if layer_key is None else f"layers/{layer_key}"])
        self._counts = self._matrix(group["X"])
        key = f"obsm/{embeddings_key}"
        self._embeddings = None if embeddings_key is None or key not in group else self._matrix(group[key])

    @staticmethod
    def _matrix(elem):
        # Sparse groups become row-indexable without loading; dense datasets/arrays already are.
        return sparse_dataset(elem) if elem.attrs.get("encoding-type") in ("csr_matrix", "csc_matrix") else elem

    @staticmethod
    def _rows(matrix, rows: np.ndarray) -> np.ndarray:
        if isinstance(matrix, zarr.Array):
            out = matrix.get_orthogonal_selection((rows, slice(None)))
        else:
            out = matrix[rows]
        return np.asarray(out.toarray() if sp.issparse(out) else out, dtype=np.float32)

    def x(self, rows: np.ndarray) -> np.ndarray:
        return self._rows(self._x, rows)

    def embeddings(self, rows: np.ndarray) -> np.ndarray | None:
        return None if self._embeddings is None else self._rows(self._embeddings, rows)

    def expressed(self, chunk: int = 65536) -> np.ndarray:
        """Cells whose ``X`` row sum is non-zero, read ``chunk`` rows at a time."""
        n = len(self.obs)
        keep = np.empty(n, dtype=bool)
        for start in range(0, n, chunk):
            block = self._counts[start : start + chunk]
            keep[start : start + chunk] = np.asarray(block.sum(axis=1)).ravel() != 0
        return keep


def _edge_index(coords: np.ndarray, neighbors: dict) -> np.ndarray:
    connectivities, _ = sq.gr.spatial_neighbors(AnnData(obsm={"spatial": coords}), copy=True, **neighbors)
    return from_scipy_sparse_matrix(connectivities)[0].numpy().astype(np.int32)


def _graph_groups(obs, cfg: CN, keep: np.ndarray):
    """``(split, row positions)`` per graph, in ``prepare_geome_dataset``'s order."""
    split = obs[cfg.dataset.split_key].astype(str).to_numpy()
    for key in cfg.dataset.sample_key:
        codes = obs[key].astype("category").cat.codes.to_numpy()
        for split_name in SPLITS:
            # One stable sort per split, not one pass over every cell per graph: the per-graph
            # scan was O(cells x graphs), hours at ~50M cells / ~80k windows. Stable keeps each
            # graph's rows ascending and graphs in code order, as before.
            rows = np.flatnonzero(keep & (split == split_name) & (codes >= 0))
            rows = rows[np.argsort(codes[rows], kind="stable")]
            yield from ((split_name, g) for g in np.split(rows, np.flatnonzero(np.diff(codes[rows])) + 1) if len(g))


class _Appender:
    """Appends along the last axis of a zarr array in whole chunks, so no chunk is rewritten."""

    def __init__(self, array: zarr.Array):
        self.array, self.parts, self.size = array, [], 0

    def add(self, block: np.ndarray) -> None:
        self.parts.append(block)
        self.size += block.shape[-1]
        if self.size >= self.array.chunks[-1]:
            self.flush()

    def flush(self) -> None:
        if self.parts:
            self.array.append(np.concatenate(self.parts, axis=-1), axis=self.array.ndim - 1)
            self.parts, self.size = [], 0


# ---------------------------------------------------------------------------------- writer


def write_graph_store(
    source: AnnData | str | os.PathLike,
    cfg: CN,
    path: str | os.PathLike,
    *,
    drop_zero_expression: bool = False,
    overwrite: bool = False,
    chunk_rows: int = 1024,
    chunk_edges: int = 65536,
) -> Path:
    """Build every graph of ``source`` and write them to a zarr graph store.

    Graphs are built and written one at a time. From a file path, only ``obs``, ``var`` and the
    spatial coordinates are loaded; each graph's rows of the expression matrix are read when it
    is written, so memory is bounded by the largest graph. The store is written to a temporary
    sibling directory and renamed into place once complete.

    Parameters
    ----------
    source
        An AnnData (in memory or backed) or the path of an ``.h5ad``/``.zarr`` AnnData, with
        ``obsm["spatial"]``, ``obs[cfg.dataset.split_key]`` (``train``, ``val``, optionally
        ``test``) and every key of ``cfg.dataset.sample_key``.
    cfg
        The run config; its :func:`graph_store_fingerprint` is saved with the store.
    path
        Target ``.zarr`` directory.
    drop_zero_expression
        Skip cells whose ``X`` row sums to zero, as
        :func:`interscale.tl.remove_zero_expression_cells` does in memory.
    overwrite
        Replace an existing store at ``path`` instead of raising.
    chunk_rows
        Rows per chunk of the per-cell arrays; about one graph is a good size.
    chunk_edges
        Edges per chunk of ``edge_index``.

    Returns
    -------
    Path
        ``path``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    fingerprint = graph_store_fingerprint(cfg)
    src = _Source(source, cfg.dataset.layer_key, fingerprint["embeddings_obsm_key"])
    if src.coords is None:
        raise ValueError("obsm['spatial'] is required to build the neighbour graphs.")
    if not {"train", "val"} <= set(src.obs[cfg.dataset.split_key].astype(str).unique()):
        raise ValueError(f"obs['{cfg.dataset.split_key}'] must contain 'train' and 'val'.")
    if fingerprint["embeddings_obsm_key"] is not None and src._embeddings is None:
        raise ValueError(f"Precomputed embeddings key '{fingerprint['embeddings_obsm_key']}' not found in obsm")

    class_names = labels = None
    if "classification" in cfg.dataset.prediction_task:
        column = src.obs[cfg.dataset.prediction_obs].astype("category")
        class_names = [str(c) for c in column.cat.categories]
        labels = column.cat.codes.to_numpy().astype(np.int32)

    keep = src.expressed() if drop_zero_expression else np.ones(len(src.obs), dtype=bool)
    groups = list(_graph_groups(src.obs, cfg, keep))
    n_total = sum(len(rows) for _, rows in groups)
    n_genes = len(src.var_names)

    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    root = zarr.open_group(tmp, mode="w")

    def per_cell(name, width, dtype):
        shape, chunks = ((n_total, width), (chunk_rows, width)) if width else ((n_total,), (chunk_rows,))
        return root.create_array(name, shape=shape, chunks=chunks, dtype=dtype)

    x = per_cell("x", n_genes, "float32")
    obs_index = per_cell("obs_index", 0, "int64")
    label = per_cell("label", 0, "int32") if labels is not None else None
    first = src.embeddings(groups[0][1][:1]) if groups else None
    embed = per_cell("embeddings", first.shape[1], "float32") if first is not None else None
    edges = _Appender(root.create_array("edge_index", shape=(2, 0), chunks=(2, chunk_edges), dtype="int32"))

    node_ptr, edge_ptr, graph_split, class_counts = [0], [0], [], []
    try:
        for split_name, rows in groups:
            start, stop = node_ptr[-1], node_ptr[-1] + len(rows)
            x[start:stop] = src.x(rows)
            obs_index[start:stop] = rows
            if label is not None:
                label[start:stop] = labels[rows]
                class_counts.append(np.bincount(labels[rows], minlength=len(class_names)))
            if embed is not None:
                embed[start:stop] = src.embeddings(rows)
            graph_edges = _edge_index(src.coords[rows], fingerprint["spatial_neighbors"])
            edges.add(graph_edges)
            node_ptr.append(stop)
            edge_ptr.append(edge_ptr[-1] + graph_edges.shape[1])
            graph_split.append(SPLITS.index(split_name))
        edges.flush()

        root.create_array("node_ptr", data=np.asarray(node_ptr, dtype=np.int64))
        root.create_array("edge_ptr", data=np.asarray(edge_ptr, dtype=np.int64))
        root.create_array("split", data=np.asarray(graph_split, dtype=np.int8))
        if class_counts:
            root.create_array("class_counts", data=np.stack(class_counts).astype(np.int64))
        root.attrs.update(
            {
                "var_names": [str(v) for v in src.var_names],
                "class_names": class_names,
                "fingerprint": fingerprint,
                "complete": True,
            }
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if overwrite and path.exists():
        shutil.rmtree(path)
    try:
        os.replace(tmp, path)
    except OSError:
        # Another process with the same config renamed its identical store into place first;
        # keep that one rather than pull it from under a reader.
        shutil.rmtree(tmp, ignore_errors=True)
        if not path.exists():
            raise
    return path


# ---------------------------------------------------------------------------------- reader


class ZarrGraphDataset(GraphDataset):
    """One split of a zarr graph store, as a contract dataset.

    The zarr group is opened lazily and dropped on pickling, so each DataLoader worker opens its
    own handle.

    Parameters
    ----------
    path
        A store written by :func:`write_graph_store`.
    split
        ``"train"``, ``"val"`` or ``"test"``.
    fingerprint
        If given, the store's fingerprint must equal it (see :func:`graph_store_fingerprint`).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        split: Literal["train", "val", "test"],
        fingerprint: dict | None = None,
    ):
        self.path = Path(path)
        self.split = split
        self._group = None
        attrs = self.group.attrs.asdict()
        if not attrs.get("complete"):
            raise ValueError(f"{self.path} is not a complete graph store.")
        if fingerprint is not None and attrs["fingerprint"] != json.loads(json.dumps(fingerprint)):
            raise ValueError(
                f"{self.path} was built for a different config.\n"
                f"  store:  {attrs['fingerprint']}\n  wanted: {fingerprint}"
            )
        super().__init__(var_names=attrs["var_names"], class_names=attrs["class_names"])
        self.graph_ids = np.flatnonzero(self.group["split"][:] == SPLITS.index(split))
        self.node_ptr = self.group["node_ptr"][:]
        self.edge_ptr = self.group["edge_ptr"][:]
        self._has_embeddings = "embeddings" in self.group

    @property
    def group(self) -> zarr.Group:
        """The store's zarr group, opened read-only on first use in each process."""
        if self._group is None:
            self._group = zarr.open_group(self.path, mode="r")
        return self._group

    def __getstate__(self):
        return {**self.__dict__, "_group": None}

    def len(self) -> int:
        return len(self.graph_ids)

    def get(self, idx: int) -> Data:
        graph = self.graph_ids[idx]
        rows = slice(int(self.node_ptr[graph]), int(self.node_ptr[graph + 1]))
        edges = slice(int(self.edge_ptr[graph]), int(self.edge_ptr[graph + 1]))
        obs_index = torch.from_numpy(self.group["obs_index"][rows])
        data = Data(
            x=torch.from_numpy(self.group["x"][rows]),
            edge_index=torch.from_numpy(self.group["edge_index"][:, edges]).long(),
            obs_index=obs_index,
            # The modules read len(obs_names) as the node count; geome stores the 1-based row
            # name as a float, which this reproduces.
            obs_names=(obs_index + 1).float(),
        )
        if self.class_names is not None:
            codes = torch.from_numpy(self.group["label"][rows]).long()
            data.y = torch.nn.functional.one_hot(codes, self.num_classes).float()
        if self._has_embeddings:
            data.embeddings = torch.from_numpy(self.group["embeddings"][rows])
        return data

    def _class_counts(self) -> np.ndarray:
        return self.group["class_counts"][:][self.graph_ids]


def open_graph_store(path: str | os.PathLike, cfg: CN | None = None) -> list[ZarrGraphDataset]:
    """Open every split of a graph store: ``[train, val]`` or ``[train, val, test]``.

    With ``cfg``, the store must have been built for the same :func:`graph_store_fingerprint`.
    """
    fingerprint = None if cfg is None else graph_store_fingerprint(cfg)
    datasets = [ZarrGraphDataset(path, split, fingerprint) for split in SPLITS]
    return datasets if len(datasets[2]) else datasets[:2]
