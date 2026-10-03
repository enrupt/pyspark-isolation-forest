from __future__ import annotations

from .model_base import _ForestModelBase
from .params import _ExtendedIsolationForestParams


class ExtendedIsolationForestModel(_ForestModelBase, _ExtendedIsolationForestParams):
    """Trained extended isolation forest. `extensionLevel` holds the level resolved at fit time."""
    _TREE_KIND = "extended"
