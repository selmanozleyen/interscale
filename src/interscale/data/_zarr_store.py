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
from collections.abc import Sequence
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
from zarr.core.sync import sync

from interscale.data._graph_dataset import GraphDataset
from interscale.geome._zarr import Appender as _Appender
from interscale.geome._zarr import gather as _gather
from interscale.geome._zarr import read_runs as _read_runs
from interscale.geome._zarr import runs as _runs

#: Bump when the layout or the way graphs are built changes, so old stores stop matching.
STORE_VERSION = 2
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- fingerprint


def graph_store_fingerprint(cfg: CN) -> dict:
    """The config values the graphs depend on: equal fingerprints build identical stores.

    Training-only settings (masking, batch size, model, optimiser) are deliberately absent. The
    source enters as its path string plus size and mtime, so moving or copying it (new mtime)
    forces a rebuild, and for a ``.zarr`` directory an in-place rewrite of its arrays does not
    change the directory's own stat and is not detected.
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
            # one stable sort per split: rows stay ascending, graphs in code order
            rows = np.flatnonzero(keep & (split == split_name) & (codes >= 0))
            rows = rows[np.argsort(codes[rows], kind="stable")]
            yield from ((split_name, g) for g in np.split(rows, np.flatnonzero(np.diff(codes[rows])) + 1) if len(g))


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
    shard_rows: int | None = None,
) -> Path:
    """Build every graph of ``source`` and write them to a zarr graph store.

    Graphs are built and written one at a time. From a file path, only ``obs``, ``var`` and the
    spatial coordinates are loaded; each graph's rows of the expression matrix are read when it
    is written, so memory is bounded by the largest graph -- or, with ``shard_rows``, by one shard
    per array (the writer buffers whole shards, e.g. 65536 x 5000 genes x 4 B = 1.3 GB for ``x``).
    The store is written to a temporary sibling directory and renamed into place once complete.

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
    shard_rows
        Rows per shard (a multiple of ``chunk_rows``), or None for one file per chunk. Sharding
        packs many chunks into one file -- far fewer files on a parallel file system such as
        Lustre -- and ``edge_index`` gets as many chunks per shard as the per-cell arrays.

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
    if shard_rows is not None and shard_rows % chunk_rows:
        raise ValueError(f"shard_rows ({shard_rows}) must be a multiple of chunk_rows ({chunk_rows}).")
    groups = list(_graph_groups(src.obs, cfg, keep))
    n_genes = len(src.var_names)

    first = src.embeddings(groups[0][1][:1]) if groups else None
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")

    def per_cell(name, width, dtype):  # root is opened inside the try below
        tail = (width,) if width else ()
        shards = None if shard_rows is None else (shard_rows, *tail)
        array = root.create_array(name, shape=(0, *tail), chunks=(chunk_rows, *tail), shards=shards, dtype=dtype)
        return _Appender(array)

    node_ptr, edge_ptr, graph_split, class_counts = [0], [0], [], []
    try:  # from the first write on, any failure removes the temporary store
        root = zarr.open_group(tmp, mode="w")
        x = per_cell("x", n_genes, "float32")
        obs_index = per_cell("obs_index", 0, "int64")
        label = per_cell("label", 0, "int32") if labels is not None else None
        embed = per_cell("embeddings", first.shape[1], "float32") if first is not None else None
        edge_shards = None if shard_rows is None else (2, chunk_edges * (shard_rows // chunk_rows))
        edges = _Appender(
            root.create_array("edge_index", shape=(2, 0), chunks=(2, chunk_edges), shards=edge_shards, dtype="int32"),
            axis=1,
        )
        for split_name, rows in groups:
            stop = node_ptr[-1] + len(rows)
            x.add(src.x(rows))
            obs_index.add(rows.astype(np.int64))
            if label is not None:
                label.add(labels[rows])
                class_counts.append(np.bincount(labels[rows], minlength=len(class_names)))
            if embed is not None:
                embed.add(src.embeddings(rows))
            graph_edges = _edge_index(src.coords[rows], fingerprint["spatial_neighbors"])
            edges.add(graph_edges)
            node_ptr.append(stop)
            edge_ptr.append(edge_ptr[-1] + graph_edges.shape[1])
            graph_split.append(SPLITS.index(split_name))
        for appender in (x, obs_index, label, embed, edges):
            if appender is not None:
                appender.flush()

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

    The zarr arrays are opened lazily and dropped on pickling, so each DataLoader worker
    opens its own handle. :meth:`__getitems__` reads a whole batch with one zarr call per array.

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
        self._arrays = None
        group = zarr.open_group(self.path, mode="r")
        attrs = group.attrs.asdict()
        if not attrs.get("complete"):
            raise ValueError(f"{self.path} is not a complete graph store.")
        if attrs["fingerprint"].get("store_version") != STORE_VERSION:
            raise ValueError(
                f"{self.path} has store layout {attrs['fingerprint'].get('store_version')}, not {STORE_VERSION}."
            )
        if fingerprint is not None and attrs["fingerprint"] != json.loads(json.dumps(fingerprint)):
            raise ValueError(
                f"{self.path} was built for a different config.\n"
                f"  store:  {attrs['fingerprint']}\n  wanted: {fingerprint}"
            )
        super().__init__(var_names=attrs["var_names"], class_names=attrs["class_names"])
        self.graph_ids = np.flatnonzero(group["split"][:] == SPLITS.index(split))
        self.node_ptr = group["node_ptr"][:]
        self.edge_ptr = group["edge_ptr"][:]
        self._has_embeddings = "embeddings" in group

    @property
    def arrays(self) -> dict[str, zarr.Array]:
        """The store's arrays, opened once per process (``group[name]`` re-reads the metadata)."""
        if self._arrays is None:
            self._arrays = dict(zarr.open_group(self.path, mode="r").arrays())
        return self._arrays

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    @property
    def _cell_keys(self) -> list[str]:
        keys = ["obs_index", "x"]
        return keys + ["label"] * (self.class_names is not None) + ["embeddings"] * self._has_embeddings

    def __getitems__(self, indices: Sequence[int]) -> list[Data]:
        """The graphs at ``indices``, with one read per array; torch's DataLoader calls this per batch.

        Reading graph by graph costs ~3 zarr calls per graph, and on a warm store nearly all of that
        is zarr-python's per-call overhead, run on its single event-loop thread. Instead each array is
        read once for the whole batch, as slice runs (:class:`_Runs`), all arrays concurrently.
        """
        graphs = self.graph_ids[np.asarray([self.indices()[i] for i in indices], dtype=np.int64)]
        order = np.argsort(graphs, kind="stable")  # increasing store offsets
        g = graphs[order]
        node_lo, node_hi, edge_lo, edge_hi = (
            self.node_ptr[g],
            self.node_ptr[g + 1],
            self.edge_ptr[g],
            self.edge_ptr[g + 1],
        )
        arrays, keys = self.arrays, self._cell_keys
        node_runs, edge_runs = _runs(node_lo, node_hi), _runs(edge_lo, edge_hi)
        reads = [_read_runs(arrays[k], node_runs, 0) for k in keys] + [_read_runs(arrays["edge_index"], edge_runs, 1)]
        *per_cell, edges = sync(_gather(reads))  # every array of the batch at once, on zarr's loop
        cut = np.cumsum(node_hi - node_lo)[:-1]
        cells = {k: np.split(v, cut) for k, v in zip(keys, per_cell, strict=True)}
        edges = np.split(edges, np.cumsum(edge_hi - edge_lo)[:-1], axis=1)
        out = [None] * len(indices)
        for k, j in enumerate(order):
            out[j] = self._graph(edge_index=edges[k], **{key: parts[k] for key, parts in cells.items()})
        return out if self.transform is None else [self.transform(d) for d in out]

    def len(self) -> int:
        return len(self.graph_ids)

    def get(self, idx: int) -> Data:
        graph = self.graph_ids[idx]
        rows = slice(int(self.node_ptr[graph]), int(self.node_ptr[graph + 1]))
        edges = slice(int(self.edge_ptr[graph]), int(self.edge_ptr[graph + 1]))
        arrays = self.arrays
        return self._graph(edge_index=arrays["edge_index"][:, edges], **{k: arrays[k][rows] for k in self._cell_keys})

    def _graph(self, x, obs_index, edge_index, label=None, embeddings=None) -> Data:
        obs_index = torch.from_numpy(obs_index)
        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index).long(),
            # not "obs_index": PyG's Batch offsets every attribute named *index* by the node count
            source_row=obs_index,
            # The modules read len(obs_names) as the node count; geome stores the 1-based row
            # name as a float, which this reproduces.
            obs_names=(obs_index + 1).float(),
        )
        if label is not None:
            data.y = torch.nn.functional.one_hot(torch.from_numpy(label).long(), self.num_classes).float()
        if embeddings is not None:
            data.embeddings = torch.from_numpy(embeddings)
        return data

    def _class_counts(self) -> np.ndarray:
        return self.arrays["class_counts"][:][self.graph_ids]


def open_graph_store(path: str | os.PathLike, cfg: CN | None = None) -> list[ZarrGraphDataset]:
    """Open every split of a graph store: ``[train, val]`` or ``[train, val, test]``.

    With ``cfg``, the store must have been built for the same :func:`graph_store_fingerprint`.
    """
    fingerprint = None if cfg is None else graph_store_fingerprint(cfg)
    datasets = [ZarrGraphDataset(path, split, fingerprint) for split in SPLITS]
    return datasets if len(datasets[2]) else datasets[:2]
