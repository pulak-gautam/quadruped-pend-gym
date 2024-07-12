import numpy as np
import gymnasium as gym
import quadruped_pend_gym

def main():
    env = gym.make('Quadruped-Pend-v0', reset_noise_scale=0.01, render_mode='human')

    timesteps=1000
    observation, info = env.reset(seed=42)
    for _ in range(timesteps):
        action = np.zeros(np.shape(env.action_space.sample()))
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()