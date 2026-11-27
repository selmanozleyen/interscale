import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data import GraphDataset
from interscale.tl.geome_utils import prepare_a2d_dataset
from interscale.tl.utils import get_model_filename_prefix

logger = logging.getLogger(__name__)

from typing import NamedTuple


class _SAVE_KEYS_NT(NamedTuple):
    MODEL_FNAME: str = "model.pt"
    MODEL_STATE_DICT_KEY: str = "state_dict"
    ATTR_DICT_KEY: str = "attr_dict"


SAVE_KEYS = _SAVE_KEYS_NT()

# State entries that may legitimately be absent from an older checkpoint. The PCA front-end's
# buffers were added after some checkpoints were written; `BaseModel.load` warns and carries on
# for these, and raises for anything else. See tests/test_global_pca_persistence.py.
_OPTIONAL_STATE_PREFIXES = ("pca_",)
# Saved with a checkpoint and checked by `load` against the dataset it is given.
_CKPT_ATTRS = ("var_names", "class_labels")


class BaseModel(ABC):
    """Abstract class for InterScale models.

    A model is built from a :class:`~interscale.data.GraphDataset` -- normally the training
    split -- and reads only metadata from it: the feature width, the class names and, for
    ``WeightedCE``, the per-class training counts. It holds no reference to the data.

    Parameters
    ----------
    dataset
        The training split, as an InterScale contract dataset.
    cfg
        The run config.
    """

    local_component: bool = False
    global_component: bool = False

    def __init__(self, dataset: GraphDataset, cfg: CN):
        self._cfg = cfg

        self.prediction_task = cfg.dataset.prediction_task
        self.prediction_level = cfg.dataset.prediction_level
        assert self.prediction_task in ["classification", "regression"], (
            "Prediction task must be either 'classification' or 'regression'."
        )
        assert self.prediction_level in ["node", "graph"], "Prediction level must be either 'node' or 'graph'."

        self.var_names = list(dataset.var_names)
        self.class_labels = None
        if self.prediction_task == "classification":
            if dataset.class_names is None:
                raise ValueError("A classification model needs a dataset with class_names.")
            self.class_labels = list(dataset.class_names)

        self.n_input = dataset.num_node_features
        self.n_embed = self._cfg.model.n_embed
        self.n_output = dataset.num_classes if self.prediction_task == "classification" else self.n_input

        self.is_trained_ = False
        self._model_summary_string = f"{self.prediction_task} model for {self.prediction_level} prediction. \n"
        self.history_ = None

        self.class_weights = None
        if self._cfg.optim.loss == "WeightedCE":
            self.class_weights = self._compute_train_class_weights(dataset)

    def _compute_train_class_weights(self, dataset: GraphDataset) -> torch.Tensor:
        """Compute ``"balanced"`` class weights over the training units of ``dataset``.

        A unit is a cell for ``prediction_level == "node"`` and a graph -- counted by its majority
        label -- for ``"graph"``. Weights are ordered like :attr:`class_labels`, the column order
        of the one-hot ``data.y``, so index *i* of the returned tensor lines up with logit *i*.

        Returns
        -------
        torch.Tensor
            Float32 tensor of per-class weights.
        """
        classes = self.class_labels
        counts = np.asarray(dataset.label_counts(self.prediction_level), dtype=np.float64)
        if (counts == 0).any():
            missing = [c for c, n in zip(classes, counts, strict=True) if n == 0]
            raise ValueError(f"Classes {missing} have no training units; WeightedCE weights undefined.")
        # Same formula as sklearn's compute_class_weight("balanced").
        weights = counts.sum() / (len(classes) * counts)
        print(
            f"WeightedCE ({self.prediction_level}-level): "
            + ", ".join(f"{c}: n={int(n)} w={w:.4f}" for c, n, w in zip(classes, counts, weights, strict=True))
        )
        return torch.as_tensor(weights, dtype=torch.float32)

    def _prepare_output(self, adata: AnnData) -> tuple[AnnData, list]:
        """Common start of ``get_model_output``: check ``adata`` and rebuild its graphs.

        Raises unless the model is trained and ``adata`` has the model's genes in the same order.
        A view is made actual in place, since outputs are written into it, and ``obs_names`` are
        renumbered ``"1".."N"`` -- the ids the graphs carry in ``obs_names``.
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")
        if list(map(str, adata.var_names)) != self.var_names:
            raise ValueError("AnnData var_names differ from the ones the model was built with (names or order).")
        if adata.is_view:
            adata._init_as_actual(adata.copy())
        adata.obs_names = [str(i) for i in range(1, adata.n_obs + 1)]
        graphs, _ = prepare_a2d_dataset(self._cfg)(adata)
        return adata, graphs

    def save_evaluation_results(
        self,
        adata: AnnData,
        prefix: str,
        y_pred_local_df: pd.DataFrame | None = None,
        y_pred_global_df: pd.DataFrame | None = None,
        local_embeddings_df: pd.DataFrame | None = None,
        global_embeddings_df: pd.DataFrame | None = None,
        attention_matrix_df: pd.DataFrame | None = None,
        cls_token_horizontal: np.ndarray | None = None,
        cls_token_vertical: np.ndarray | None = None,
    ):
        """Save the evaluation results in the adata object.

        Parameters
        ----------
        adata: AnnData
        prefix: str
        local_embeddings_df: pd.DataFrame
        global_embeddings_df: pd.DataFrame
        attention_matrix_df: pd.DataFrame
        y_pred_local_df: pd.DataFrame
        y_pred_global_df: pd.DataFrame
        cls_token_horizontal: np.ndarray
        cls_token_vertical: np.ndarray

        Returns
        -------
        adata: AnnData
            AnnData object with the evaluation results saved in the obsm and layers.
        """
        if local_embeddings_df is not None:
            adata.obsm[f"{prefix}_local_emb"] = local_embeddings_df.values
        if global_embeddings_df is not None:
            adata.obsm[f"{prefix}_global_emb"] = global_embeddings_df.values
        if attention_matrix_df is not None:
            adata.obsm[f"{prefix}_attn_matrix"] = attention_matrix_df.values
        if cls_token_horizontal is not None:
            adata.obs[f"{prefix}_cls_horizontal"] = cls_token_horizontal
        if cls_token_vertical is not None:
            adata.obs[f"{prefix}_cls_vertical"] = cls_token_vertical

        if self.prediction_task == "classification" and y_pred_local_df is not None:
            adata.obsm[f"{prefix}_y_pred_local"] = y_pred_local_df.values  # [cells, classes]
        elif y_pred_local_df is not None:
            adata.layers[f"{prefix}_y_pred_local"] = y_pred_local_df.values  # [cells, genes]

        if self.prediction_task == "classification" and y_pred_global_df is not None:
            adata.obsm[f"{prefix}_y_pred_global"] = y_pred_global_df.values  # [cells, classes]
        elif y_pred_global_df is not None:
            adata.layers[f"{prefix}_y_pred_global"] = y_pred_global_df.values  # [cells, genes]

        return adata

    @abstractmethod
    def train(self):
        """Trains the model."""

    def save(
        self,
        dir_path: str | None = None,
        overwrite: bool = False,
        postfix: str | None = None,
        save_kwargs: dict | None = None,
    ):
        """Save the state of the model.
        File is saved as <dataset_name>_<prediction_task[:4]>_<prediction_level>_<local_component_name>_<global_component_name>_<model_state_dict>.pt

        Parameters
        ----------
        dir_path
            Path to a directory or cfg.model.save_path
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_kwargs
            Keyword arguments passed into :func:`~torch.save`.
        """
        import warnings

        if dir_path is None:
            dir_path = self._cfg.model.save

        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        file_name_prefix = get_model_filename_prefix(self._cfg, self.local_component, self.global_component)

        if postfix is not None:
            file_name_prefix = file_name_prefix + f"{postfix}"

        save_kwargs = save_kwargs or {}

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}{SAVE_KEYS.MODEL_FNAME}")

        # Check if file exists and warn if it does
        if os.path.exists(model_save_path) and not overwrite:
            warnings.warn(
                f"File {model_save_path} already exists. Set overwrite=True to overwrite.",
                UserWarning,
                stacklevel=2,
            )
            return

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()

        torch.save(
            {
                SAVE_KEYS.MODEL_STATE_DICT_KEY: model_state_dict,
                SAVE_KEYS.ATTR_DICT_KEY: {key: getattr(self, key) for key in _CKPT_ATTRS},
            },
            model_save_path,
            **save_kwargs,
        )

    def _check_saved_attrs(self, attrs: dict | None, path: str) -> None:
        """Raise if a checkpoint was written for different genes or classes than ``self``."""
        for key in _CKPT_ATTRS:
            saved = (attrs or {}).get(key)  # checkpoints before the attrs were saved have none
            if saved is not None and list(saved) != list(getattr(self, key) or []):
                raise ValueError(
                    f"{key} of the dataset do not match the ones checkpoint {path} was trained with "
                    f"(names or order). The checkpoint's input/output columns would be misaligned."
                )

    @classmethod
    def load(
        cls,
        dir_path: str,
        dataset: GraphDataset,
        cfg: CN,
        model_name: str | None = None,
        local_component: bool | None = None,
        global_component: bool | None = None,
        postfix: str | None = None,
        wandb_save: bool = False,
        enable_remapping: bool = True,
        allow_partial_load: bool = False,
    ):
        """Load a saved model.

        Parameters
        ----------
        dir_path
            Path to saved model directory.
        dataset
            Contract dataset to build the model from, normally the training split. Its
            ``var_names`` and ``class_names`` must match the ones saved with the checkpoint
            (checkpoints written before these were saved are not checked).
        cfg
            Configuration object.
        model_name: str | None
            Name of the model to load. If None, the model name is inferred from the config file.
        local_component, global_component
            Which components the checkpoint's filename names; default to the class's own.
        wandb_save
            Whether this was saved via wandb.
        allow_partial_load
            If False (default), a checkpoint whose keys do not match the model this cfg builds
            raises, instead of silently leaving the unmatched tensors randomly initialised. Set
            True only when a partly initialised model is deliberate.
        enable_remapping
            Whether to enable automatic state dict key remapping.

        Returns
        -------
        model
            Loaded model.
        """
        if model_name is not None:
            file_name_prefix = model_name
        else:
            file_name_prefix = get_model_filename_prefix(
                cfg,
                cls.local_component if local_component is None else local_component,
                cls.global_component if global_component is None else global_component,
            )

        if postfix is not None:
            file_name_prefix = file_name_prefix + f"{postfix}"

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}{SAVE_KEYS.MODEL_FNAME}")

        # Initialize model
        model = cls(dataset, cfg)

        print(f"Loading model from {model_save_path}")

        # Determine map_location based on CUDA availability
        # map_location = 'cpu' if cfg.optim.accelerator == 'cpu' else None

        # Always force CPU when CUDA is unavailable
        if torch.cuda.is_available():
            map_location = None  # load to GPU as usual
        else:
            map_location = torch.device("cpu")

        if not os.path.exists(model_save_path):
            print("Try with .ckpt extension.")
            model_save_path = model_save_path.replace(".pt", ".ckpt")
            if not os.path.exists(model_save_path):
                print(f"Model file {model_save_path} not found.")
                raise FileNotFoundError(f"Model file {model_save_path} not found.")
        checkpoint = torch.load(model_save_path, map_location=map_location)
        state_dict = checkpoint[SAVE_KEYS.MODEL_STATE_DICT_KEY]
        model._check_saved_attrs(checkpoint.get(SAVE_KEYS.ATTR_DICT_KEY), model_save_path)

        # Apply remapping if enabled
        if enable_remapping:
            try:
                from interscale.tl.utils import detect_and_remap_state_dict_keys

                state_dict, source = detect_and_remap_state_dict_keys(state_dict)
                print(f"State dict remapping applied. Source detected: {source}")
            except ImportError:
                print("Warning: Could not import remapping functions. Loading without remapping.")

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Replace 'global_module.module.' with 'global_module.'
            new_key = k.replace("global_module.module.", "global_module.")

            # Also handle generic 'module.' prefix if present at the start (legacy DataParallel)
            if new_key.startswith("module."):
                new_key = new_key.replace("module.", "", 1)

            new_state_dict[new_key] = v
        state_dict = new_state_dict

        # Legacy wandb remapping (kept for backward compatibility)
        if wandb_save:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "", 1)  # only remove the first 'module.'
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        # Load the state dict
        missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)

        # strict=False is load_state_dict's "fill in what you can and say nothing" mode. It is
        # needed for the legacy carve-out below, but on its own it turns an architecture mismatch
        # -- a config that does not describe the checkpoint -- into a model whose unmatched
        # tensors keep their random initialisation. That model runs, produces plausible-looking
        # embeddings and gene loadings, and is wrong. Printed warnings do not survive a notebook
        # with hundreds of lines of output, so anything but the documented benign case raises.
        benign_missing = [k for k in missing_keys if k.rsplit(".", 1)[-1].startswith(_OPTIONAL_STATE_PREFIXES)]
        hard_missing = [k for k in missing_keys if k not in benign_missing]

        if benign_missing:
            print(
                f"Warning: checkpoint predates these buffers, loading without them: {benign_missing}. "
                f"A PCA front-end will refit on the first batch it sees rather than reusing the "
                f"basis it was trained with (see tests/test_global_pca_persistence.py)."
            )

        if (hard_missing or unexpected_keys) and not allow_partial_load:
            raise RuntimeError(
                f"Checkpoint does not match the model this cfg builds, so the mismatched tensors "
                f"would keep their random initialisation and the model would be silently wrong.\n"
                f"  missing from the checkpoint (left random): {hard_missing}\n"
                f"  present in the checkpoint but not in the model: {unexpected_keys}\n"
                f"  checkpoint: {model_save_path}\n"
                f"Check that dataset/task, n_embed, decoder type and dual_decoder in cfg match the "
                f"run that wrote it. Pass allow_partial_load=True only if a partly initialised "
                f"model is genuinely what you want."
            )

        model.is_trained_ = True

        return model
