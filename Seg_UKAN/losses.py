# losses.py — compatibility shim
from src.training.losses import *  # noqa: F401, F403
from src.training.losses import BCEDiceLoss, LovaszHingeLoss, __all__  # noqa: F401
