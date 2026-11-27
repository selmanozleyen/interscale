import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data import GraphDataset
from interscale.model.base._base_model import BaseModel
from interscale.module.base import LocalModule
from interscale.train._training import NodeMaskingTrainingPlan


class LocalModel(NodeMaskingTrainingPlan, BaseModel):
    """Local model with only local component."""

    local_component = True

    def __init__(
        self,
        dataset: GraphDataset,
        cfg: CN,
    ):
        super().__init__(dataset, cfg)
        self.module = LocalModule.from_config(
            cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=self._cfg.model.decoder.type,
            dropout_decoder=self._cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
            mask_percentage=self._cfg.dataset.mask_percentage,
            mask_strategy=self._cfg.dataset.mask_strategy,
        )

    @torch.inference_mode()
    def get_model_output(self, adata: AnnData, prefix: str = ""):
        """Save the embeddings, predictions and attentionsin the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. Must have the genes the model was built with;
            the outputs are written into it.
        prefix
            Prefix for the output columns.
        """
        adata, pyg = self._prepare_output(adata)
        obs_names_str = adata.obs_names

        # Create empty DataFrame with correct shape
        local_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)
        y_pred_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)

        for batch in pyg:
            local_embedding = self.module.forward(batch.x, batch.edge_index)
            # Get indices for this sample
            sample_mask = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            # Fill embeddings directly into the DataFrame
            local_embeddings_df.loc[sample_mask] = local_embedding.detach().cpu().numpy()

            y_pred = self.module.predict(local_embedding, self.prediction_level)
            y_pred_df.loc[sample_mask] = y_pred.detach().cpu().numpy()

        # Save embeddings in adata.obsm
        adata = self.save_evaluation_results(
            adata,
            prefix,
            local_embeddings_df=local_embeddings_df,
            # decoder_weight_df = decoder_weight_df,
            y_pred_local_df=y_pred_df,
        )

        return adata
