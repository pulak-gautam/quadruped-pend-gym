import gymnasium as gym

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger= lambda t: t % 1 == 0, disable_logger=True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk