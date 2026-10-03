from __future__ import annotations

from .model_base import _ForestModelBase
from .params import _IsolationForestParams


class IsolationForestModel(_ForestModelBase, _IsolationForestParams):
    """Trained standard isolation forest. `transform` appends `scoreCol` and `predictionCol`."""
    _TREE_KIND = "standard"
