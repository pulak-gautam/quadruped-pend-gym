
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import quadruped_pend_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Quadruped-Pend-v2"
    """the id of the environment"""
    total_timesteps: int = 10
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, env, exploration_noise=1, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mu = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer("exploration_noise", torch.as_tensor(exploration_noise, device=device))

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = self.fc_mu(obs).tanh()
        return obs * self.action_scale + self.action_bias

    def explore(self, obs):
        act = self(obs)
        return act + torch.randn_like(act).mul(self.action_scale * self.exploration_noise)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    action_low, action_high = float(envs.single_action_space.low[0]), float(envs.single_action_space.high[0])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)
    actor_detach = Actor(env=envs, n_obs=n_obs, n_act=n_act, device=device, exploration_noise=args.exploration_noise)

    exp_name="walking-0.2-0.0"
    actor_params = torch.load(f"runs/{exp_name}/actor.pth", map_location=device, weights_only=True)
    actor.load_state_dict(actor_params)
    actor.eval()
    actor_detach.eval()
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    envs.single_observation_space.dtype = np.float32

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    action_scale = actor.action_scale

    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    start_time = None
    max_ep_ret = -float("inf")
    autoreset = False
    r = l = 0

    n_episodes=0
    while n_episodes < 10:
        actions = policy(obs=obs)
        actions = actions.clamp(action_low, action_high).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "episode" in infos:
            r = float(infos["episode"]["r"].reshape(()))
            l = float(infos["episode"]["l"].reshape(()))
            max_ep_ret = max(max_ep_ret, r)
            n_episodes+=1
            print(f"episode_n={n_episodes}, episodic_ret={r} episode_len={l} (max={max_ep_ret: 4.2f})")


        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        autoreset = np.logical_or(terminations, truncations)
        obs = next_obs

    envs.close()
