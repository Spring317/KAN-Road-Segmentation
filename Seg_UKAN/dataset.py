# dataset.py — compatibility shim
# All production code should use ``from src.data import …`` directly.
from src.data import *  # noqa: F401, F403
from src.data import (  # noqa: F401
    BDD100K_CLASSES,
    BDD100K_COLOR_DICT,
    BDD100K_NUM_CLASSES,
    BDD100KDataset,
    colorize_mask,
    mask_to_onehot,
    onehot_to_mask,
)
