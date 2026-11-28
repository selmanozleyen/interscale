"""The staged geome API (interscale.geome): a grouped AnnData, a dataset and a DataModule over it."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import zarr
from anndata.acc import A
from anndata.experimental import read_lazy as anndata_read_lazy

from interscale.data import ZarrGraphDataset, write_graph_store
from interscale.geome import GroupDataModule, GroupDataset, item_generator, read_lazy, write_grouped
from interscale.geome.io import GroupedAnnData


def _neighbors(cfg):
    return {k: v for k, v in dict(cfg.dataset.spatial_neigbors_kwargs).items() if k != "library_key"}


def _write(adata, cfg, path, x="X", **kwargs):
    return write_grouped(
        adata, path, groupby="sample", split="split", neighbors=_neighbors(cfg), matrices=(x,), obs=("ct",), **kwargs
    )


@pytest.mark.parametrize(("sparse", "layer"), [(False, None), (True, "log1p")])
def test_graphs_match_the_graph_store(make_cfg, make_spatial_adata, tmp_path, sparse, layer):
    cfg = make_cfg(extra=["dataset.layer_key", layer])
    store = write_graph_store(make_spatial_adata(sparse=sparse), cfg, tmp_path / "store.zarr")
    x = "X" if layer is None else f"layers/{layer}"
    grouped = _write(make_spatial_adata(sparse=sparse), cfg, tmp_path / "grouped.zarr", x=x)

    for split in ("train", "val", "test"):
        reference, dataset = ZarrGraphDataset(store, split), GroupDataset(grouped, x=x, y="ct", split=split)
        assert len(reference) == len(dataset)
        for a, b in zip(reference.__getitems__(range(len(reference))), dataset.__getitems__(range(len(dataset)))):
            for key in ("x", "edge_index", "y", "source_row"):
                assert torch.equal(a[key], b[key]), key


def test_any_anndata_object_gives_the_same_graphs(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    path = _write(make_spatial_adata(sparse=True), cfg, tmp_path / "g.zarr")
    ad.read_zarr(path).write_h5ad(tmp_path / "g.h5ad")
    sources = {
        "path": path,
        "in memory": ad.read_zarr(path),
        "anndata read_lazy": anndata_read_lazy(zarr.open_group(path, mode="r")),
        "geome read_lazy": read_lazy(path),
        "backed h5ad": tmp_path / "g.h5ad",
    }
    graphs = {name: GroupDataset(s, y="ct").__getitems__(range(5)) for name, s in sources.items()}
    for name, items in graphs.items():
        for a, b in zip(graphs["path"], items, strict=True):
            for key in ("x", "edge_index", "y", "source_row"):
                assert torch.equal(a[key], b[key]), (name, key)


def test_output_is_an_ordinary_grouped_anndata(make_cfg, make_spatial_adata, tmp_path):
    source = make_spatial_adata(sparse=True)
    adata = ad.read_zarr(_write(source, make_cfg(), tmp_path / "g.zarr"))

    meta = adata.uns["geome"]
    ptr = np.asarray(meta["group_ptr"])
    assert sp.issparse(adata.X), "sparse input stays sparse"
    assert adata.n_obs == source.n_obs and list(adata.var_names) == list(source.var_names)
    for g in range(len(ptr) - 1):
        block = adata.obs.iloc[ptr[g] : ptr[g + 1]]
        assert block["sample"].nunique() == 1 and block["split"].nunique() == 1
        assert block["sample"].iloc[0] == meta["group_names"][g]
    conn = adata.obsp["spatial_connectivities"].tocoo()
    group_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    assert (group_of[conn.row] == group_of[conn.col]).all(), "edges stay inside their group"
    np.testing.assert_array_equal(adata.X.toarray(), source.X[adata.obs["source_row"].to_numpy()].toarray())
    assert adata.uns["spatial_neighbors"]["connectivities_key"] == "spatial_connectivities"


def test_nonzero_filter_drops_empty_cells(make_cfg, make_spatial_adata, tmp_path):
    adata = make_spatial_adata(sparse=True)
    empty = np.arange(0, adata.n_obs, 7)
    adata.X[empty] = 0
    adata.X.eliminate_zeros()

    out = ad.read_zarr(_write(adata, make_cfg(), tmp_path / "g.zarr", row_filter="nonzero"))

    assert out.n_obs == adata.n_obs - len(empty)
    assert not np.isin(out.obs["source_row"], empty).any()


def test_datamodule_passes_idx_epoch_and_split_to_the_transform(make_cfg, make_spatial_adata, tmp_path):
    path = _write(make_spatial_adata(), make_cfg(), tmp_path / "g.zarr")
    seen = []

    def transform(data, *, idx, epoch, split):
        seen.append((split, epoch))
        data.noise = torch.rand(1, generator=item_generator(0, idx, epoch if split == "train" else None))
        return data

    dm = GroupDataModule(path, y="ct", batch_size=2, transform=transform)
    first = [b.noise for b in dm.train_dataloader()]
    val = [b.noise for b in dm.val_dataloader()]
    dm.set_epoch(1)
    assert sum(len(b) for b in first) == len(dm.datasets["train"])
    assert [b.noise for b in dm.val_dataloader()][0].tolist() == val[0].tolist()
    assert ("train", 1) not in seen
    list(dm.train_dataloader())
    assert ("train", 1) in seen and {s for s, _ in seen} == {"train", "val"}


def test_read_lazy_holds_the_small_parts_in_memory_and_reads_matrices_from_the_store(
    make_cfg, make_spatial_adata, tmp_path
):
    path = _write(make_spatial_adata(sparse=True), make_cfg(), tmp_path / "g.zarr", obsm=("latent",))

    adata = read_lazy(path)

    assert type(adata.obs).__name__ == "DataFrame" and isinstance(
        adata.obs["sample"].dtype, __import__("pandas").CategoricalDtype
    )
    assert isinstance(adata.obsm["spatial"], np.ndarray) and isinstance(adata.obsm["latent"], np.ndarray)
    assert isinstance(adata.uns["geome"]["group_ptr"], np.ndarray)
    assert all(hasattr(m, "compute") for m in (adata.X, adata.obsp["spatial_connectivities"])), "matrices stay lazy"
    reader = GroupedAnnData(adata)
    assert reader.adata is None and reader.group is not None, "matrices are read from the store, not through dask"


@pytest.mark.parametrize(
    ("spellings", "path"),
    [
        ((A.X, "X"), "X"),
        ((A.layers["log1p"], "layers.log1p", "layers/log1p"), "layers/log1p"),
        ((A.obsm["latent"], "obsm.latent", "obsm/latent"), "obsm/latent"),
        ((A.obsp["spatial_connectivities"], "obsp.spatial_connectivities"), "obsp/spatial_connectivities"),
        ((A.obs["ct"], "obs.ct", "obs/ct"), "obs/ct"),
    ],
)
def test_accessors_dotted_strings_and_paths_name_the_same_field(spellings, path):
    from interscale.geome._refs import as_acc, to_dotted, to_path

    for field in spellings:
        assert to_path(field) == path
        assert as_acc(field) == as_acc(path)
        assert as_acc(to_dotted(field)) == as_acc(field)


def test_every_spelling_gives_the_same_graphs(make_cfg, make_spatial_adata, tmp_path):
    cfg = make_cfg()
    by_accessor = write_grouped(
        make_spatial_adata(sparse=True), tmp_path / "a.zarr", groupby=A.obs["sample"], split="obs.split",
        neighbors=_neighbors(cfg), matrices=[A.layers["log1p"]], obs=[A.obs["ct"]], obsm=[A.obsm["latent"]],
    )  # fmt: skip
    by_path = write_grouped(
        make_spatial_adata(sparse=True), tmp_path / "p.zarr", groupby="obs/sample", split="split",
        neighbors=_neighbors(cfg), matrices=["layers/log1p"], obs=["ct"], obsm=["obsm/latent"],
    )  # fmt: skip
    datasets = [
        GroupDataset(by_accessor, x=A.layers["log1p"], y=A.obs["ct"], extra={"emb": A.obsm["latent"]}),
        GroupDataset(by_path, x="layers/log1p", y="ct", extra={"emb": "obsm/latent"}),
        GroupDataset(read_lazy(by_path), x="layers.log1p", y="obs.ct", extra={"emb": "obsm.latent"}),
    ]
    reference = datasets[0].__getitems__(range(len(datasets[0])))
    for dataset in datasets[1:]:
        for a, b in zip(reference, dataset.__getitems__(range(len(dataset))), strict=True):
            for key in ("x", "edge_index", "y", "source_row", "emb"):
                assert torch.equal(a[key], b[key]), key
