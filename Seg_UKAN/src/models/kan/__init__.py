"""KAN linear implementations package.

Exports the factory function ``get_kan_linear`` which returns the appropriate
KAN linear class based on a string key.
"""

from src.models.kan.fasterkan import KANLinear
from src.models.kan.variants import (
    ReLUKANLinear,
    HardSwishKANLinear,
    PWLOKANLinear,
    TeLUKANLinear,
)

__all__ = [
    "get_kan_linear",
    "KANLinear",
    "ReLUKANLinear",
    "HardSwishKANLinear",
    "PWLOKANLinear",
    "TeLUKANLinear",
]

_REGISTRY: dict = {
    "FasterKAN": KANLinear,
    "ReLU": ReLUKANLinear,
    "HardSwish": HardSwishKANLinear,
    "PWLO": PWLOKANLinear,
    "TeLU": TeLUKANLinear,
}


def get_kan_linear(kan_type: str):
    """Return the KAN linear class for *kan_type*.

    Open/Closed principle: add a new variant to ``_REGISTRY`` without touching
    any other file.

    Args:
        kan_type: One of ``"FasterKAN"``, ``"ReLU"``, ``"HardSwish"``,
            ``"PWLO"``, ``"TeLU"``.

    Returns:
        An ``nn.Module`` subclass with the ``(in_features, out_features, …)``
        constructor interface.

    Raises:
        ValueError: Unknown *kan_type*.
    """
    try:
        return _REGISTRY[kan_type]
    except KeyError:
        raise ValueError(
            f"Unknown kan_type '{kan_type}'. "
            f"Choose from: {sorted(_REGISTRY.keys())}"
        )
