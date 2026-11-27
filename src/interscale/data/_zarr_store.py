"""An on-disk implementation of the graph-dataset contract, backed by a zarr store.

Every graph of every split is concatenated into flat arrays, with pointer arrays recording
where each graph starts -- the layout PyG's ``InMemoryDataset`` uses in memory, on disk::

    <store>.zarr/
      x            [N_total, G]  float32   cells; each graph is one contiguous block of rows
      label        [N_total]     int32     class code per cell (classification only)
      embeddings   [N_total, D]  float32   precomputed embeddings (only if configured)
      obs_index    [N_total]     int64     row of the cell in the AnnData the store was built from
      edge_index   [2, E_total]  int32     node ids local to each graph
      node_ptr     [n_graphs+1]  int64     graph i owns rows node_ptr[i]:node_ptr[i+1]
      edge_ptr     [n_graphs+1]  int64     graph i owns edges edge_ptr[i]:edge_ptr[i+1]
      split        [n_graphs]    int8      index into attrs["splits"]
      class_counts [n_graphs, C] int64     cells per class per graph (classification only)
      .attrs       var_names, class_names, splits, fingerprint, complete

Reading graph ``i`` is one contiguous slice per array, so a training step touches only the rows
of the graphs in its batch. The model-facing metadata (``var_names``, ``class_names``,
``label_counts``) comes from ``attrs`` and ``class_counts``, so building a model reads no
expression data at all.

The graphs are built exactly as :func:`interscale.tl.prepare_geome_dataset` builds them -- one
per (split, sample category), neighbours from :func:`squidpy.gr.spatial_neighbors` on that
graph's cells, edges via :func:`torch_geometric.utils.from_scipy_sparse_matrix` -- but one graph at
a time, so writing holds at most one graph in memory on top of the input AnnData (which may be
backed). ``tests/test_zarr_store.py`` checks the two agree graph for graph.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq
import torch
import zarr
from anndata import AnnData
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from yacs.config import CfgNode as CN

from interscale.data._graph_dataset import GraphDataset

#: Bump when the layout or the way graphs are built changes, so old stores stop matching.
STORE_VERSION = 1
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- fingerprint


def graph_store_fingerprint(cfg: CN) -> dict:
    """The config values a graph store depends on.

    Two configs with equal fingerprints build identical graphs, so a store written for one can
    be reused by the other. Training-only settings (masking, batch size, model, optimiser) are
    deliberately absent: they do not change the graphs.

    Parameters
    ----------
    cfg
        The run config.

    Returns
    -------
    dict
        JSON-serialisable description of the graphs.
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


def _fingerprint_hash(fingerprint: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]


def graph_store_path(store_dir: str | os.PathLike, cfg: CN) -> Path:
    """Where the store for ``cfg`` lives inside ``store_dir``: ``<name>-<fingerprint hash>.zarr``.

    Configs that build different graphs get different paths, so sweep trials never overwrite each
    other's store and every trial with the same graphs reuses one.
    """
    name = cfg.dataset.name or "graphs"
    return Path(store_dir) / f"{name}-{_fingerprint_hash(graph_store_fingerprint(cfg))}.zarr"


# ---------------------------------------------------------------------------------- writer


def _layer_rows(adata: AnnData, layer_key: str | None) -> np.ndarray:
    matrix = adata.X if layer_key is None else adata.layers[layer_key]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _edge_index(sub: AnnData, neighbors: dict) -> np.ndarray:
    # Only coordinates are needed, so build the neighbour graph on an obs/obsm-only AnnData and
    # leave the expression matrix alone. One library: the graph's own cells.
    coords = AnnData(obs=pd.DataFrame(index=sub.obs_names), obsm={"spatial": np.asarray(sub.obsm["spatial"])})
    sq.gr.spatial_neighbors(coords, spatial_key="spatial", key_added="adj", **neighbors)
    edge_index, _ = from_scipy_sparse_matrix(coords.obsp["adj_connectivities"])
    return edge_index.numpy().astype(np.int32)


def _graph_groups(adata: AnnData, cfg: CN):
    """Yield ``(split, row positions)`` per graph, in ``prepare_geome_dataset``'s order."""
    split = adata.obs[cfg.dataset.split_key].astype(str).to_numpy()
    for key in cfg.dataset.sample_key:
        column = adata.obs[key].astype("category")
        codes = column.cat.codes.to_numpy()
        for split_name in SPLITS:
            in_split = split == split_name
            for code in range(len(column.cat.categories)):
                rows = np.flatnonzero(in_split & (codes == code))
                if len(rows):
                    yield split_name, rows


def write_graph_store(
    adata: AnnData,
    cfg: CN,
    path: str | os.PathLike,
    *,
    overwrite: bool = False,
    chunk_rows: int = 4096,
    chunk_edges: int = 65536,
) -> Path:
    """Build every graph of ``adata`` and write them to a zarr graph store.

    Graphs are built and appended one at a time, so memory is bounded by the largest graph on top
    of ``adata`` itself, which may be backed (``read_h5ad(..., backed="r")``). The store is written
    to a temporary sibling directory and renamed into place only once complete, so an interrupted
    write never leaves a store that looks valid.

    Parameters
    ----------
    adata
        Preprocessed AnnData with ``obsm["spatial"]``, ``obs[cfg.dataset.split_key]`` (``train``,
        ``val``, optionally ``test``) and every key of ``cfg.dataset.sample_key``.
    cfg
        The run config. Its :func:`graph_store_fingerprint` is saved with the store.
    path
        Target ``.zarr`` directory.
    overwrite
        Replace an existing store at ``path``. Otherwise an existing store raises.
    chunk_rows
        Rows per chunk of the per-cell arrays.
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
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required to build the neighbour graphs.")
    split_values = set(adata.obs[cfg.dataset.split_key].astype(str).unique())
    if not {"train", "val"} <= split_values:
        raise ValueError(f"adata.obs['{cfg.dataset.split_key}'] must contain 'train' and 'val'.")

    fingerprint = graph_store_fingerprint(cfg)
    classification = "classification" in cfg.dataset.prediction_task
    class_names = None
    labels = None
    if classification:
        label_column = adata.obs[cfg.dataset.prediction_obs].astype("category")
        class_names = [str(c) for c in label_column.cat.categories]
        labels = label_column.cat.codes.to_numpy().astype(np.int32)
    embeddings_key = fingerprint["embeddings_obsm_key"]
    if embeddings_key is not None and embeddings_key not in adata.obsm:
        raise ValueError(f"Precomputed embeddings key '{embeddings_key}' not found in adata.obsm")
    neighbors = fingerprint["spatial_neighbors"]
    n_genes = adata.n_vars if cfg.dataset.layer_key is None else adata.layers[cfg.dataset.layer_key].shape[1]

    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    root = zarr.open_group(tmp, mode="w")
    x = root.create_array("x", shape=(0, n_genes), chunks=(chunk_rows, n_genes), dtype="float32")
    obs_index = root.create_array("obs_index", shape=(0,), chunks=(chunk_rows,), dtype="int64")
    edges = root.create_array("edge_index", shape=(2, 0), chunks=(2, chunk_edges), dtype="int32")
    label = root.create_array("label", shape=(0,), chunks=(chunk_rows,), dtype="int32") if classification else None
    embed = None
    if embeddings_key is not None:
        dim = adata.obsm[embeddings_key].shape[1]
        embed = root.create_array("embeddings", shape=(0, dim), chunks=(chunk_rows, dim), dtype="float32")

    node_ptr, edge_ptr, graph_split, class_counts = [0], [0], [], []
    try:
        for split_name, rows in _graph_groups(adata, cfg):
            sub = adata[rows]
            if sub.isbacked:
                sub = sub.to_memory()
            x.append(_layer_rows(sub, cfg.dataset.layer_key), axis=0)
            obs_index.append(rows.astype(np.int64), axis=0)
            graph_edges = _edge_index(sub, neighbors)
            edges.append(graph_edges, axis=1)
            if label is not None:
                codes = labels[rows]
                label.append(codes, axis=0)
                class_counts.append(np.bincount(codes, minlength=len(class_names)))
            if embed is not None:
                embed.append(np.asarray(sub.obsm[embeddings_key], dtype=np.float32), axis=0)
            node_ptr.append(node_ptr[-1] + len(rows))
            edge_ptr.append(edge_ptr[-1] + graph_edges.shape[1])
            graph_split.append(SPLITS.index(split_name))

        root.create_array("node_ptr", data=np.asarray(node_ptr, dtype=np.int64))
        root.create_array("edge_ptr", data=np.asarray(edge_ptr, dtype=np.int64))
        root.create_array("split", data=np.asarray(graph_split, dtype=np.int8))
        if class_counts:
            root.create_array("class_counts", data=np.stack(class_counts).astype(np.int64))
        root.attrs.update(
            {
                "var_names": [str(v) for v in adata.var_names],
                "class_names": class_names,
                "splits": list(SPLITS),
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
        # Another process with the same config renamed its store into place first. The graphs
        # are identical, so keep that one: deleting it could pull it from under a reader.
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
        self.fingerprint = attrs["fingerprint"]
        self.graph_ids = np.flatnonzero(self.group["split"][:] == attrs["splits"].index(split))
        self.node_ptr = self.group["node_ptr"][:]
        self.edge_ptr = self.group["edge_ptr"][:]
        self._has_labels = "label" in self.group
        self._has_embeddings = "embeddings" in self.group

    @property
    def group(self) -> zarr.Group:
        """The store's zarr group, opened read-only on first use in each process."""
        if self._group is None:
            self._group = zarr.open_group(self.path, mode="r")
        return self._group

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_group"] = None
        return state

    def len(self) -> int:
        return len(self.graph_ids)

    def get(self, idx: int) -> Data:
        graph = self.graph_ids[idx]
        start, stop = int(self.node_ptr[graph]), int(self.node_ptr[graph + 1])
        e_start, e_stop = int(self.edge_ptr[graph]), int(self.edge_ptr[graph + 1])
        obs_index = torch.from_numpy(self.group["obs_index"][start:stop])
        data = Data(
            x=torch.from_numpy(self.group["x"][start:stop]),
            edge_index=torch.from_numpy(self.group["edge_index"][:, e_start:e_stop]).long(),
            obs_index=obs_index,
            # The modules read len(obs_names) as the node count; geome stores the 1-based row
            # name as a float, which this reproduces.
            obs_names=(obs_index + 1).float(),
        )
        if self._has_labels:
            codes = torch.from_numpy(self.group["label"][start:stop]).long()
            data.y = torch.nn.functional.one_hot(codes, self.num_classes).float()
        if self._has_embeddings:
            data.embeddings = torch.from_numpy(self.group["embeddings"][start:stop])
        return data

    def label_counts(self, level: Literal["node", "graph"]) -> np.ndarray:
        if self.class_names is None:
            raise ValueError("label_counts needs class_names; this is a regression store.")
        per_graph = self.group["class_counts"][:][self.graph_ids].astype(np.float64)
        if level == "node":
            return per_graph.sum(axis=0)
        if level == "graph":
            return np.bincount(per_graph.argmax(axis=1), minlength=self.num_classes).astype(np.float64)
        raise ValueError(f"level must be 'node' or 'graph', got {level!r}")


def open_graph_store(path: str | os.PathLike, cfg: CN | None = None) -> list[ZarrGraphDataset]:
    """Open every split of a graph store, ``[train, val]`` or ``[train, val, test]``.

    Parameters
    ----------
    path
        A store written by :func:`write_graph_store`.
    cfg
        If given, the store must have been built for a config with the same
        :func:`graph_store_fingerprint`.
    """
    fingerprint = None if cfg is None else graph_store_fingerprint(cfg)
    datasets = [ZarrGraphDataset(path, split, fingerprint) for split in ("train", "val")]
    test = ZarrGraphDataset(path, "test", fingerprint)
    return datasets + ([test] if len(test) else [])


def graph_store_datasets(cfg: CN, load_adata: Callable[[], AnnData]) -> list[ZarrGraphDataset]:
    """Open the graph store for ``cfg``, building it first if it does not exist yet.

    The store lives at :func:`graph_store_path` ``(cfg.dataset.graph_store, cfg)``. If it is
    there, ``load_adata`` is never called -- training then never reads the h5ad. Otherwise
    ``load_adata()`` must return the preprocessed AnnData to build it from.

    Parameters
    ----------
    cfg
        The run config, with ``cfg.dataset.graph_store`` set to a directory.
    load_adata
        Zero-argument callable returning the preprocessed AnnData.
    """
    path = graph_store_path(cfg.dataset.graph_store, cfg)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing graph store {path}")
        write_graph_store(load_adata(), cfg, path)
    else:
        print(f"Reusing graph store {path}")
    return open_graph_store(path, cfg)
