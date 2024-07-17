from quadruped_pend_gym.quad_pend_rl.algorithms import TD3

if __name__ == "__main__":
    runner = TD3()
    runner.setup()
    runner.run()