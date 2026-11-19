"""Train InterScale's CombinedModel on Legnini with its window pipeline or with geome tiles, and report both.

    python scripts/tiles_vs_windows.py --pipeline windows --epochs 30 --out windows.json
    python scripts/tiles_vs_windows.py --pipeline tiles   --epochs 30 --out tiles.json

Data and split follow docs/notebooks/0_setup_legnini.ipynb: the log1p of counts normalized to 1e4, a radius-300 graph
per sample, and the tutorial's per-sample train/val/test assignment. ``windows`` is `prepare_geome_dataset` +
`GraphAnnDataModule` (one graph per sample, in memory); ``tiles`` is `TileDataModule` (tiles of at most
``max_seq_len`` cells plus a margin, read per tile), on the same graph.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import yaml

import interscale
from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.tiles_datamodule import TileDataModule
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells, set_full_reproducibility

VAL = ["slide1_B2-1", "slide1_B2-3", "slide1_A2-2", "slide1_C2-3", "slide4_B2-3"]
TEST = ["slide4_A2-3", "slide4_B2-2"]
RADIUS = 300


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def legnini(cache: str | None):
    adata = interscale.datasets.legnini(cache)
    adata = adata[~np.isnan(adata.obsm["spatial"]).any(axis=1)].copy()
    adata.layers["log1p_norm"] = sc.pp.log1p(sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)["X"])
    adata = remove_zero_expression_cells(adata)
    adata.obs["split"] = "train"
    adata.obs.loc[adata.obs["sample"].isin(VAL), "split"] = "val"
    adata.obs.loc[adata.obs["sample"].isin(TEST), "split"] = "test"
    return adata


def config(args, h5ad: str):
    overrides = {
        "model": {"local_component": {"name": "GCN"}, "global_component": {"name": "self-attn-transformer"}},
        "dataset": {
            "h5ad_data": h5ad,
            "name": f"legnini_{args.pipeline}",
            "layer_key": "log1p_norm",
            "sample_key": ["sample"],
            "split_key": "split",
            "prediction_task": "regression",
            "prediction_level": "node",
            "batch_size": args.batch_size,
            "spatial_neigbors_kwargs": {"radius": RADIUS},
        },
        "optim": {"n_epochs": args.epochs, "early_stopping": False, "seed": args.seed},
        "wandb": {"use": False},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(overrides, f)
    return load_config(f.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=["windows", "tiles", "tiles-backed"], default="tiles")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--cache", default=None, help="where legnini() caches its .h5ad")
    p.add_argument("--zarr", default=None, help="tiles-backed: the zarr store (written by --write-zarr)")
    p.add_argument("--write-zarr", action="store_true", help="write --zarr from Legnini and exit")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.write_zarr:
        write_zarr(args)
        return
    cfg = config(args, h5ad=str(args.cache or ""))
    set_full_reproducibility(cfg.optim.seed)
    if args.pipeline == "tiles-backed":
        return run(args, cfg, *backed(args, cfg))

    adata = legnini(args.cache)
    sizes = adata.obs["sample"].value_counts()
    print(f"{adata.n_obs} cells, {adata.n_vars} genes, {sizes.size} samples ({sizes.min()}-{sizes.max()} cells)")

    interscale.model.CombinedModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        split_key="split",
    )
    model = interscale.model.CombinedModel(adata, cfg=cfg)

    t0 = time.perf_counter()
    if args.pipeline == "windows":
        datas, _ = prepare_geome_dataset(adata, cfg)
        dm = GraphAnnDataModule(
            datas=datas,
            num_workers=1,
            batch_size=args.batch_size,
            mask_percentage=cfg.dataset.mask_percentage,
            mask_strategy=cfg.dataset.mask_strategy,
            learning_type="node",
        )
    else:
        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=RADIUS, library_key="sample")
        dm = TileDataModule(adata, cfg, graph=adata.obsp["spatial_connectivities"])
    run(args, cfg, model, dm, time.perf_counter() - t0)


def write_zarr(args) -> None:
    """Legnini's layer, coordinates, sample and split, rows sorted by sample and then along a Z-order curve."""
    import anndata as ad
    from geome.tiles import morton_order

    adata = legnini(args.cache)
    rank = np.empty(adata.n_obs, dtype=np.int64)
    rank[morton_order(adata.obsm["spatial"])] = np.arange(adata.n_obs)
    order = np.lexsort((rank, pd.factorize(adata.obs["sample"])[0]))
    slim = ad.AnnData(
        obs=adata.obs[["sample", "split"]].astype("category"),
        var=adata.var[[]],
        layers={"log1p_norm": adata.layers["log1p_norm"]},
        obsm={"spatial": adata.obsm["spatial"]},
    )[order].copy()
    slim.write_zarr(args.zarr)
    print(f"wrote {args.zarr}: {slim.n_obs} cells x {slim.n_vars} genes")


def backed(args, cfg):
    """Model and datamodule from the zarr store: expression stays on disk, the model is built from shapes."""
    import anndata as ad
    import zarr

    t0 = time.perf_counter()
    g = zarr.open_group(args.zarr, mode="r")  # only obs and coordinates, for the radius graph
    light = ad.AnnData(obs=ad.io.read_elem(g["obs"])[["sample"]], obsm={"spatial": np.asarray(g["obsm/spatial"][:])})
    sq.gr.spatial_neighbors(light, coord_type="generic", radius=RADIUS, library_key="sample")
    dm = TileDataModule.from_path(args.zarr, cfg, graph=light.obsp["spatial_connectivities"])
    model = interscale.model.CombinedModel(None, cfg=cfg, registry=dm.registry)
    return model, dm, time.perf_counter() - t0


def run(args, cfg, model, dm, prep_s) -> None:
    if isinstance(dm, TileDataModule):
        n = [t.size for t in dm.loader.tiles]
        print(f"{len(n)} tiles ({min(n)}-{max(n)} core cells), margin of {dm.loader.n_hops} hops")
    t0 = time.perf_counter()
    model.train(max_epochs=args.epochs, datamodule=dm, early_stopping=False, wandb_use=False)
    train_s = time.perf_counter() - t0

    history = [
        {k: float(v) if isinstance(v, int | float) else str(v) for k, v in h.items()} for h in model.history_.history
    ]
    val = [h for h in history if any(k.startswith("val_") for k in h)]
    result = {
        "pipeline": args.pipeline,
        "epochs": args.epochs,
        "prep_seconds": prep_s,
        "train_seconds": train_s,
        "peak_rss_gb": peak_rss_gb(),
        "last": val[-1] if val else (history[-1] if history else {}),
        "best_val_loss": min((h["val_loss"] for h in val if "val_loss" in h), default=None),
    }
    Path(args.out).write_text(json.dumps({"result": result, "history": history}, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
