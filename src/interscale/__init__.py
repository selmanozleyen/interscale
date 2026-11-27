from importlib.metadata import PackageNotFoundError, version

from . import config, data, datasets, evaluation, model, module, pl, tl

__all__ = ["config", "data", "datasets", "evaluation", "module", "tl", "model", "pl"]

try:
    __version__ = version("interscale")
except PackageNotFoundError:
    pass
