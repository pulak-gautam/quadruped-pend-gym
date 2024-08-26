import time
from typing import Callable
import argparse

import gymnasium as gym
import torch
import torch.nn as nn

from quadruped_pend_gym.quad_pend_rl.modules import Actor, QNetwork, make_env

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    gamma: float = 0.99
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[1](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf2.load_state_dict(qf2_params)
    qf1.eval()
    qf2.eval()
    # note: qf1 and qf2 are not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, _, _, _, infos = envs.step(actions)
        if "episode"  in infos:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}")
            episodic_returns += [infos["episode"]["r"]]
        obs = next_obs

    return episodic_returns

if __name__ == "__main__":
    run_name = f"InvertedPendulumQuadruped__{int(time.time())}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "./runs/Quadruped-Pend-v1__td3_per__1__1724575336/td3_per.cleanrl_model"

    evaluate(
        model_path,
        make_env,
        "Quadruped-Pend-v1",
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_video=True,
    )