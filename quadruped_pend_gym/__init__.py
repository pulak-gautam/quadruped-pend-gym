""" Registers the gym environments and exports the `gym.make` function.
"""

# Exporting envs:
from quadruped_pend_gym.envs.quadruped_pend_v0 import QuadrupedPendEnv_v0
from quadruped_pend_gym.envs.quadruped_pend_v1 import QuadrupedPendEnv_v1

# Exporting gym.make:
from gymnasium import make

# Registering environments:
from gymnasium.envs.registration import register

register(
    id="Quadruped-Pend-v0",
    entry_point="quadruped_pend_gym:QuadrupedPendEnv_v0",
    max_episode_steps=1000,
)

register(
    id="Quadruped-Pend-v1",
    entry_point="quadruped_pend_gym:QuadrupedPendEnv_v1",
    max_episode_steps=1000,
)

# Main names:
__all__ = [
    make.__name__,
    QuadrupedPendEnv_v0.__name__,
    QuadrupedPendEnv_v1.__name__,
]