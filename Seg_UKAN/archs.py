# archs.py — compatibility shim
# All production code should use ``from src.models import UKAN`` directly.
from src.models import *  # noqa: F401, F403
from src.models import UKAN, __all__  # noqa: F401
