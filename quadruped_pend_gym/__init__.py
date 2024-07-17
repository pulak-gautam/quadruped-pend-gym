""" Registers the gym environments and exports the `gym.make` function.
"""

# Exporting envs:
from quadruped_pend_gym.envs.quadruped_pend import QuadrupedPendEnv

# Exporting gym.make:
from gymnasium import make

# Registering environments:
from gymnasium.envs.registration import register

register(
    id="Quadruped-Pend-v0",
    entry_point="quadruped_pend_gym:QuadrupedPendEnv",
    max_episode_steps=1000,
)

# Main names:
__all__ = [
    make.__name__,
    QuadrupedPendEnv.__name__,
]