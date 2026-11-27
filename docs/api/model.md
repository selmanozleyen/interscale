# Model

InterScale is a model descigned for spatial transcpriptomics analysis. It provides, 1) **local and global embeddings** for gene level analysis and 2) **attention matrix** for cell-to-cell analysis.

![InterScale concept](../_static/img/InterScale_model.png)

## Overview

InterScale is a two component model. The local model learns cell representation of a local, spatial neighborhood and the global compponent learns tissue wide interactions between these neighborhoods.

### InterScale model

This is the main model class that can be used to define, train, and evaluate the model on an anndata. InterScale's `model` uses `module` to initialize the local and global components in the model (see below).

```{eval-rst}
.. module:: interscale.model
    :no-index:
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    model.CombinedModel
    model.GlobalModel
    model.LocalModel
```

### Data contract

Models are built from, and trained on, :class:`torch_geometric.data.Dataset` subclasses that follow a small contract: one graph per sample/window with `x [N, G]`, `edge_index` and, for classification, a one-hot `y [N, C]`, plus `var_names`, `class_names` and `label_counts()` as metadata. `build_graph_datasets` builds the in-memory implementation from an AnnData; any other storage that follows the contract trains the same models.

```{eval-rst}
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    data.GraphDataset
    data.InMemoryGraphDataset
    data.MaskedGraphDataset
    data.build_graph_datasets
    data.ZarrGraphDataset
    data.write_graph_store
    data.open_graph_store
    data.load_graph_datasets
```

For data that does not fit in memory, write the graphs once to a zarr store and train from it. Given a file path, the writer reads only each graph's rows; training then reads only the graphs in each batch:

```
from interscale.data import open_graph_store, write_graph_store

write_graph_store("data.h5ad", cfg, "graphs.zarr", drop_zero_expression=True)
datasets = open_graph_store("graphs.zarr", cfg)  # same contract as build_graph_datasets
```

From the command line, set `dataset.graph_store` to a directory and `main.py` does both steps.

### InterScale module

InterScale is built from composable local and global modules. The combined modules wire them together into a full model. The base classes (`LocalModule`, `GlobalModule`) define the interface — subclass either to implement a custom architecture and pass it to any model class.

#### Combined modules

Two combined modules are available. `DualDecoderCombinedModule` is the default and trains a separate decoder for each scale. `CombinedModule` uses a single shared (global) decoder.

```{eval-rst}
.. module:: interscale.module
    :no-index:
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    module.DualDecoderCombinedModule
    module.CombinedModule
```

#### Local modules

`LocalModule` is the base class. Four ready-made implementations are provided; subclass `LocalModule` to define your own.

```{eval-rst}
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    module.GIN
    module.GCN
    module.SCVILocalModule
    module.PrecomputedEmbeddingModule
```

#### Global modules

`GlobalModule` is the base class. The default implementation is a transformer encoder with self-attention relevance hooks; subclass `GlobalModule` to define your own.

```{eval-rst}
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    module.TransformerNodeEncoderHook
```



## Usage example

```
import anndata as ad
import interscale
from interscale.config import load_config
from interscale.data import build_graph_datasets
from interscale.geome_dataloader import GraphAnnDataModule

# Load your model and training configurations
cfg = load_config(cfg_path)

# Load your data and build one graph per sample, split into train / val (/ test)
adata = ad.read_h5ad("your_data.h5ad")
datasets = build_graph_datasets(adata, cfg)

# The model reads its input/output sizes and class names from the training split
model = interscale.model.CombinedModel(datasets[0], cfg=cfg)

dm = GraphAnnDataModule.from_cfg(datasets, cfg)

# Train the model
model.train(max_epochs=20, datamodule=dm, early_stopping=True, wandb_use=False)

# Write embeddings, predictions and attention into the AnnData
result = model.get_model_output(adata)

# Please check tutorials for more details and downstream steps
```
