import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import stable_baselines3 as sb3
if sb3.__version__ < "2.0":
    raise ValueError(
        """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
    )

from quadruped_pend_gym.quad_pend_rl.modules import Actor, QNetwork, make_env

class TD3():
    def __init__(self, config_path: str = "./quadruped_pend_gym/config/algo_config.yaml"):
        self.args = yaml.safe_load(open(config_path))

        self.exp_name = os.path.basename(__file__)[: -len(".py")]
        self.run_name = f"{self.args['env_id']}__{self.exp_name}__{self.args['seed']}__{int(time.time())}"

        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args['cuda'] else "cpu")

        self.envs = gym.vector.SyncVectorEnv([make_env(self.args['env_id'], 0, self.args['capture_video'], self.run_name, self.args['gamma']) for i in range(self.args['num_envs'])])
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
# torch.Size([256, 12])
# torch.Size([2, 12])

    def setup(self):
        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = QNetwork(self.envs).to(self.device)
        self.qf2 = QNetwork(self.envs).to(self.device)
        self.qf1_target = QNetwork(self.envs).to(self.device)
        self.qf2_target = QNetwork(self.envs).to(self.device)

        self.target_actor = Actor(self.envs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args['learning_rate'])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args['learning_rate'])

        self.envs.single_observation_space.dtype = np.float32
        self.rb = ReplayBuffer(
            int(self.args['buffer_size']), #pyyaml issue: https://github.com/yaml/pyyaml/pull/555
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
            n_envs=self.args['num_envs'],
        )

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in (self.args).items()])),
        )

    def run(self):
        # seeding
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.backends.cudnn.deterministic = self.args['torch_deterministic']

        start_time = time.time()

        # start the env
        obs, _ = self.envs.reset(seed=self.args['seed'])

        for global_step in range(int(self.args['total_timesteps'])):
            if global_step < int(self.args['learning_starts']):
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(0, self.actor.action_scale * self.args['exploration_noise'])
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # logging returns
            if "episode" in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", np.sum(r for r in infos["episode"]["r"] if r != 0) / np.sum(1 for r in infos["episode"]["r"] if r != 0), global_step) 
                self.writer.add_scalar("charts/episodic_length", np.sum(l for l in infos["episode"]["l"] if l != 0) / np.sum(1 for l in infos["episode"]["l"] if l != 0), global_step) 

            real_next_obs = next_obs.copy()
            # print(obs.shape)
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            obs = next_obs

            # learning
            if global_step > int(self.args['learning_starts']):
                data = self.rb.sample(int(self.args['batch_size']))
                with torch.no_grad():
                    clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.args['policy_noise']).clamp(
                        -self.args['noise_clip'], self.args['noise_clip']
                    ) * self.target_actor.action_scale

                    next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                        self.envs.single_action_space.low[0], self.envs.single_action_space.high[0]
                    )
                    self.qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    self.qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(self.qf1_next_target, self.qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args['gamma'] * (min_qf_next_target).view(-1)

                self.qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                self.qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                self.qf1_loss = F.mse_loss(self.qf1_a_values, next_q_value)
                self.qf2_loss = F.mse_loss(self.qf2_a_values, next_q_value)
                qf_loss = self.qf1_loss + self.qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.args['policy_frequency'] == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.args['tau'] * param.data + (1 - self.args['tau']) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args['tau'] * param.data + (1 - self.args['tau']) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args['tau'] * param.data + (1 - self.args['tau']) * target_param.data)

                # logging returns
                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", self.qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", self.qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", self.qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", self.qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    print("SPS:", int((global_step * self.args['num_envs']) / (time.time() - start_time)))
                    self.writer.add_scalar("charts/SPS", int((global_step  * self.args['num_envs'])/ (time.time() - start_time)), global_step)
            
        
        if self.args['save_model']:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save((self.actor.state_dict(), self.qf1.state_dict(), self.qf2.state_dict()), model_path)
            print(f"model saved to {model_path}")
            
        self.envs.close()
        self.writer.close()

if __name__ == "__main__":
    runner = TD3()
    runner.setup()
    runner.run()

    

    
