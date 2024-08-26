import numpy as np
import gymnasium as gym
import quadruped_pend_gym

if __name__ == '__main__':
    env = gym.make('Quadruped-Pend-v1', reset_noise_scale=0.01, render_mode='human')

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = np.zeros(np.shape(env.action_space.sample()))
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()