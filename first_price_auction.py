"""Prime environment entrypoint for Hub installs.

Prime's integration tests import the environment by its normalized package name,
``first_price_auction``, so this module re-exports the actual implementation
from ``auction_env.py``.
"""

from auction_env import AuctionEnvConfig, load_environment

__all__ = ["AuctionEnvConfig", "load_environment"]
