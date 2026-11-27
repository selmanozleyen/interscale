from .base._base_model import BaseModel
from .combined_model import CombinedModel
from .global_model import GlobalModel
from .local_model import LocalModel

MODEL_CLASSES = {cls.__name__: cls for cls in (LocalModel, GlobalModel, CombinedModel)}

__all__ = ["BaseModel", "LocalModel", "GlobalModel", "CombinedModel", "MODEL_CLASSES"]
