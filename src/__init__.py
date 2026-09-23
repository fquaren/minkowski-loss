# Mink-DDPM: Minkowski Image Loss for Precipitation Super-Resolution

# Shared-node limits (GPU 1 only, cores 4-11): applied before anything else in `src` runs,
# so every script importing from `src` inherits them. See src/node_limits.py.
from src.node_limits import enforce as _enforce_node_limits

_enforce_node_limits()
