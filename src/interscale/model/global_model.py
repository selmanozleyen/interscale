import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data import GraphDataset
from interscale.model.base._base_model import BaseModel
from interscale.module.base import GlobalModule
from interscale.train._training import NodeMaskingTrainingPlan


class GlobalModel(NodeMaskingTrainingPlan, BaseModel):
    """Global model with only global component."""

    global_component = True

    def __init__(
        self,
        dataset: GraphDataset,
        cfg: CN,
    ):
        super().__init__(dataset, cfg)
        self.module = GlobalModule.from_config(
            cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=self._cfg.model.decoder.type,
            dropout_decoder=self._cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=self._cfg.model.decoder.hidden_dims,
            mask_percentage=self._cfg.dataset.mask_percentage,
            mask_strategy=self._cfg.dataset.mask_strategy,
            type_gex_embedding=self._cfg.model.global_component.parameters.type_gex_embedding,
        )

    # @torch.inference_mode() Not possible because of pytorch hook for self attention relevance
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
        global_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)
        attention_matrix_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self._cfg.model.global_component.parameters.max_seq_len),
            dtype=np.float32,
        )

        y_pred_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)

        cls_token_horizontal = np.full(len(adata.obs_names), np.nan)
        cls_token_vertical = np.full(len(adata.obs_names), np.nan)

        for batch in pyg:
            if hasattr(batch, "embeddings"):
                embedding = batch.embeddings
            else:
                embedding = self.module.create_gex_embedding(
                    batch.x, type=self._cfg.model.global_component.parameters.type_gex_embedding
                )
            embedding = torch.tensor(embedding, dtype=torch.float32, device=batch.x.device)
            transformer_in, global_embedding, src_padding_mask, pad_index_nodes, I = self.module.evaluate(
                batch, embedding
            )
            # no masking during evaluation
            y_pred = self.module.predict(global_embedding, src_padding_mask, self.prediction_level)

            batch_obs_names_str = batch.obs_names.numpy().astype(int).astype(str)[pad_index_nodes[0]]
            sample_mask = global_embeddings_df.index.isin(batch_obs_names_str)
            global_embeddings_df.loc[sample_mask] = global_embedding[:-1].squeeze(1).detach().cpu().numpy()
            cls_token_horizontal[sample_mask] = I[-1, :-1].squeeze().cpu().detach().numpy()
            cls_token_vertical[sample_mask] = I[:-1, -1].squeeze().cpu().detach().numpy()
            attn_matrix = I[:-1, :-1].cpu().detach().numpy()

            # Pad attention matrix to match max_seq_len with NaN
            padded_attn = np.full(
                (attn_matrix.shape[0], self._cfg.model.global_component.parameters.max_seq_len), np.nan
            )
            padded_attn[:, : attn_matrix.shape[1]] = attn_matrix
            attention_matrix_df.loc[sample_mask] = padded_attn
            y_pred_df.loc[sample_mask] = y_pred.detach().cpu().numpy()

        # Save embeddings in adata.obsm
        adata = self.save_evaluation_results(
            adata,
            prefix,
            y_pred_global_df=y_pred_df,
            global_embeddings_df=global_embeddings_df,
            attention_matrix_df=attention_matrix_df,
            cls_token_horizontal=cls_token_horizontal,
            cls_token_vertical=cls_token_vertical,
        )

        return adata
