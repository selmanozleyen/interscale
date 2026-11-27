import numpy as np
import pandas as pd
from anndata import AnnData
from yacs.config import CfgNode as CN

from interscale.data import GraphDataset
from interscale.model.base._base_model import BaseModel
from interscale.module import CombinedModule, DualDecoderCombinedModule
from interscale.train import NodeMaskingTrainingPlan


class CombinedModel(NodeMaskingTrainingPlan, BaseModel):
    """Combined model with both local and global components."""

    local_component = True
    global_component = True

    def __init__(
        self,
        dataset: GraphDataset,
        cfg: CN,
    ):
        super().__init__(dataset, cfg)
        # With dual_decoder, the local and global components each get their own decoder; the
        # container itself never decodes.
        module_cls = DualDecoderCombinedModule if cfg.model.decoder.dual_decoder else CombinedModule
        self.module = module_cls(
            cfg=cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=None,
            dropout_decoder=cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=cfg.model.decoder.hidden_dims,
            mask_percentage=cfg.dataset.mask_percentage,
            mask_strategy=cfg.dataset.mask_strategy,
        )
        self._model_summary_string = self._model_summary_string + self.module.get_model_summary()

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

        global_embeddings_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_embed), dtype=np.float32)
        attention_matrix_df = pd.DataFrame(
            index=obs_names_str,
            columns=range(self._cfg.model.global_component.parameters.max_seq_len),
            dtype=np.float32,
        )

        if self._cfg.model.decoder.dual_decoder:
            y_pred_local_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)
        else:
            y_pred_local_df = None

        y_pred_global_df = pd.DataFrame(index=obs_names_str, columns=range(self.n_output), dtype=np.float32)

        cls_token_horizontal = np.full(len(adata.obs_names), np.nan)
        cls_token_vertical = np.full(len(adata.obs_names), np.nan)

        for batch in pyg:
            ## Local model output
            local_embedding = self.module.local_module.forward(batch.x, batch.edge_index)
            if self._cfg.model.decoder.dual_decoder:
                # node_idx=None: inference decodes every cell, in the batch's own order. This
                # used to pass an explicit arange, which was the same identity gather.
                y_pred_local = self.module.predict_local(local_embedding)

            sample_mask_local = local_embeddings_df.index.isin(batch.obs_names.numpy().astype(int).astype(str))
            local_embeddings_df.loc[sample_mask_local] = local_embedding.detach().cpu().numpy()
            if self._cfg.model.decoder.dual_decoder:
                y_pred_local_df.loc[sample_mask_local] = y_pred_local.detach().cpu().numpy()

            ## Global model output
            transformer_in, global_embedding, src_padding_mask, pad_index_nodes, I = self.module.global_module.evaluate(
                batch, local_embedding
            )
            # no masking during evaluation
            y_pred_global = self.module.predict_global(global_embedding, src_padding_mask, self.prediction_level)
            ## Save model output

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

            y_pred_global_df.loc[sample_mask] = y_pred_global.detach().cpu().numpy()

        adata = self.save_evaluation_results(
            adata,
            prefix,
            y_pred_local_df=y_pred_local_df,
            y_pred_global_df=y_pred_global_df,
            local_embeddings_df=local_embeddings_df,
            global_embeddings_df=global_embeddings_df,
            attention_matrix_df=attention_matrix_df,
            cls_token_horizontal=cls_token_horizontal,
            cls_token_vertical=cls_token_vertical,
        )

        return adata
