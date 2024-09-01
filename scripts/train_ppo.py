import yaml
from rl_games.torch_runner import Runner

config_path = "./quadruped_pend_gym/config/ppo_config.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

print(type(config))
config['params']['config']['full_experiment_name'] = 'Quadruped_Pendulum_PPO'
config['params']['config']['max_epochs'] = 500
config['params']['config']['horizon_length'] = 512
config['params']['config']['num_actors'] = 8
config['params']['config']['minibatch_size'] = 1024

runner = Runner()
runner.load(config)
runner.run({
    'train': True,
})