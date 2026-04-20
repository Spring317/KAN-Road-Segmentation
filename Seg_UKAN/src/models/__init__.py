"""Models package — public surface.

Exposes ``UKAN`` and the ``__all__`` list used by
``archs.__dict__[config['arch']]`` lookups in the training script.
"""

from src.models.ukan import UKAN

__all__ = ["UKAN"]
