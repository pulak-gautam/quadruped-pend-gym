## Inverted Pendulum balancing on Quadruped using off-policy RL
This repository consists of a gymnasium environment which implements a reinforcement learning environment to facilitate training controllers for balancing inverted pendulum on a quadruped.

Currently, only Unitree's **Go2** with **TD3 (Twin Delayed DDPG)** algorithm is supported out-of-the-box. The codebase follows modular structure and it is easy to add other models and algorithms.

This repository uses ``stable_baselines3`` with changes made to make it work along with ``gymnasium v1.0.0``. (Refer [here](https://github.com/DLR-RM/stable-baselines3/pull/1837) for changes. You may use my [fork](https://github.com/pulak-gautam/stable-baselines3) in the meanwhile.

<p style="text-align: center;">
    <img src="assets/sample.gif" alt="", width=400, height=400>
</p>
