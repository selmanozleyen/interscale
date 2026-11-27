import argparse
import warnings
from pathlib import Path

from interscale.config import load_config
from interscale.config.cli import add_config_args, print_registry, resolve_cfg_from_args
from interscale.data import load_graph_datasets
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.model import MODEL_CLASSES
from interscale.tl import set_full_reproducibility

# geome calls the deprecated `sq.gr.spatial_neighbors` entrypoint; silence its
# FutureWarnings so they don't flood the training logs.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")


def build_run(cfg, model_type: str, num_workers: int = 1):
    """The model and datamodule of a training run: what ``main`` and ``main_sweep`` train."""
    set_full_reproducibility(cfg.optim.seed)  # graph construction may draw (segmentation noise)
    datasets = load_graph_datasets(cfg)
    set_full_reproducibility(cfg.optim.seed)  # so the initial weights depend only on optim.seed
    model = MODEL_CLASSES[model_type](datasets[0], cfg=cfg)
    return model, GraphAnnDataModule.from_cfg(datasets, cfg, num_workers=num_workers)


def main(cfg, model_type):
    """Train a single model.

    Parameters
    ----------
    cfg : CN or str or pathlib.Path
        An already-resolved config, or a path to a single config file to load.
    model_type : str
        One of ``LocalModel``, ``GlobalModel``, ``CombinedModel``.
    """
    # Accepts a path as well as a cfg so the historical main(cfg_path, model_type) call still
    # works; the CLI now resolves the config itself, since --dataset/--task layers several files.
    if isinstance(cfg, str | Path):
        cfg = load_config(cfg)

    print(cfg)
    model, dm = build_run(cfg, model_type)
    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one InterScale model on a registered (dataset, task) pair.",
        epilog=(
            "examples:\n"
            "  %(prog)s --dataset melton25 --task graph_clas --model_type CombinedModel\n"
            "  %(prog)s --list\n"
            "  %(prog)s --cfg config_files/legnini_example.yaml --model_type CombinedModel\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_config_args(parser)
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        default=None,
        choices=list(MODEL_CLASSES),
        help="The model type: LocalModel, GlobalModel or CombinedModel.",
    )
    args = parser.parse_args()

    if args.list_pairs:
        print_registry(args.registry)
        raise SystemExit(0)

    # Not required=True because --list must work without it.
    if args.model_type is None:
        parser.error("--model_type is required (LocalModel, GlobalModel or CombinedModel).")

    main(resolve_cfg_from_args(args, parser=parser), args.model_type)
