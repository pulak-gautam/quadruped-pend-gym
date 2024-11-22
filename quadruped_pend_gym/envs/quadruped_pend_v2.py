__credits__ = ["Pulak-Gautam"]

from typing import Dict, Union

import os
import math
import numpy as np
import quaternion
import mujoco as mj
import yaml

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box

from scipy.spatial.transform import Rotation
from scipy.stats import norm

from quadruped_pend_gym.envs.utils import XmlGenerator, display, phi_F

class QuadrupedPendEnv_v2(MujocoEnv, utils.EzPickle):
    """
    ## Action Space
    The agent take a 12-element vector for actions.

    The action space is a continuous where `action` represents:

    | Num | Action                                   | Control Min | Control Max | Name (in corresponding XML file)  |      Joint     |Type (Unit)|
    |-----|------------------------------------------|-------------|-------------|-----------------------------------|----------------|-----------|
    | 0   | joint angle of FR_hip                    | -23.7       | 23.7        | FR_hip                            | FR_hip_joint   |  radians  |
    | 1   | joint angle of FR_thigh                  | -23.7       | 23.7        | FR_thigh                          | FR_thigh_joint |  radians  |
    | 2   | joint angle of FR_calf                   | -45.43      | 45.43       | FR_calf                           | FR_calf_joint  |  radians  |
    | 3   | joint angle of FL_hip                    | -23.7       | 23.7        | FL_hip                            | FL_hip_joint   |  radians  |
    | 4   | joint angle of FL_thigh                  | -23.7       | 23.7        | FL_thigh                          | FL_thigh_joint |  radians  |
    | 5   | joint angle of FL_calf                   | -45.43      | 45.43       | FL_calf                           | FL_calf_joint  |  radians  |
    | 6   | joint angle of RR_hip                    | -23.7       | 23.7        | RR_hip                            | RR_hip_joint   |  radians  |
    | 7   | joint angle of RR_thigh                  | -23.7       | 23.7        | RR_thigh                          | RR_thigh_joint |  radians  |
    | 8   | joint angle of RR_calf                   | -45.43      | 45.43       | RR_calf                           | RR_calf_joint  |  radians  |
    | 9   | joint angle of RL_hip                    | -23.7       | 23.7        | RL_hip                            | RL_hip_joint   |  radians  |
    | 10  | joint angle of RL_thigh                  | -23.7       | 23.7        | RL_thigh                          | RL_thigh_joint |  radians  |
    | 11  | joint angle of RL_calf                   | -45.43      | 45.43       | RL_calf                           | RL_calf_joint  |  radians  |


    ## Observation Space
    The observation space is a `Box(-Inf, Inf, (68,), float64)` where the elements are as follows:
        qpos of all joints, and previous two joint angles (last and second-last action)

    ## Rewards
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible 
    a reward of +10 is given for each timestep that the pole is $ |axis-angle| < 0.2 $
    a reward of +2 is given for each timestep that the pole is $ 0.2 < |axis-angle| < 0.4 $
    a reward of +0.5 is given for each timestep that the pole is $ 0.4 < |axis-angle| < 0.5 $
    a reward of +0.1 is given for each timestep that the pole is $ 0.5 < |axis-angle| < 0.6 $
    a reward of -1 in all other cases and terminate 
    
    and `info` also contains the reward.

    ## Starting State is perturbation around:
    Joint 'FR_hip_joint' qpos is [0.] and qvel is [0.]
    Joint 'FR_thigh_joint' qpos is [0.] and qvel is [0.]
    Joint 'FR_calf_joint' qpos is [0.] and qvel is [0.]
    Joint 'FL_hip_joint' qpos is [0.] and qvel is [0.]
    Joint 'FL_thigh_joint' qpos is [0.] and qvel is [0.]
    Joint 'FL_calf_joint' qpos is [0.] and qvel is [0.]
    Joint 'RR_hip_joint' qpos is [0.] and qvel is [0.]
    Joint 'RR_thigh_joint' qpos is [0.] and qvel is [0.]
    Joint 'RR_calf_joint' qpos is [0.] and qvel is [0.]
    Joint 'RL_hip_joint' qpos is [0.] and qvel is [0.]
    Joint 'RL_thigh_joint' qpos is [0.] and qvel is [0.]
    Joint 'RL_calf_joint' qpos is [0.] and qvel is [0.]
    Joint 'pole_joint' qpos is [1. 0. 0. 0.] and qvel is [0. 0. 0.]

    The initial position state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
    The initial velocity state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.

    where $\\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The environment terminates when:
    I.  Inverted Pendulum is unhealthy.
        The Inverted Pendulum is unhealthy if any of the following happens:
            1. Any of the state space values is no longer finite.
            2. The absolute value of the axis angle between the pole and the quadruped is greater than TIPPING_ANGLE radians wrt global frame
    II. Base of quadruped is tilted more than TIPPING_BASE_ANGLE

    ### Truncation
    The default duration of an episode is 1000 timesteps.

    ## Arguments
    Quadruped-Pend-v0 provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Quadruped-Pend-v0', reset_noise_scale=0.1)
    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `xml_file`              | **str**    |`"go2/scene.xml"`        | Path to a MuJoCo model                                                                        |
    | `reset_noise_scale`     | **float**  | `0.01`                  | Scale of random perturbations of initial position and velocity (see `Starting State` section) |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "../models/go2/scene.xml") , #TODO: add robot model choice in config (eg. config['robot'] = go2)
        frame_skip: int = 2, 
        default_camera_config: Dict[str, Union[float, int]] = None,
        reset_noise_scale: float = None,
        config_file: str = None,
        **kwargs,
    ):
        try:
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
        except:
            display("WARNING", "Yaml file not found, using default config")
            with open(os.path.join(os.path.dirname(__file__), "../config/env_config.yaml"), 'r') as file:
                self.config = yaml.safe_load(file)

        if self.config['set_pend_params']:
            display("INFO", f"Setting pendulum params: density={self.config['rho']}, height={self.config['h']}, radius={self.config['r']}")
            XmlGenerator(rho=self.config['rho'], h=self.config['h'], r=self.config['r']).run()
        else:
            display("INFO", f"Using default pendulum params: density={2710.0}, height={1.0}, radius={0.01}")
            XmlGenerator().run()
        
        self._reset_noise_scale = self.config['reset_noise_scale']
        frame_skip = self.config['frame_skip']
        camera_config = self.config['camera_config']

        observation_space = Box(low=-np.inf, high=np.inf, shape=(46 + 
                                                                 self.config['action_buffer_l']*12 + 
                                                                 self.config['obs_buffer_l']*2,), dtype=np.float64)
        # print((46 + self.config['action_buffer_l']*12 + self.config['obs_buffer_l']*2,))
        
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "theta" : np.size([0.0]), 
            "base_theta" : np.size([0.0]),
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "prev_actions" : (np.size(self.config['stand_up_joint_pos'])) * self.config['action_buffer_l'],
            "prev_states"  : (np.size([0.0, 0.0])) * self.config['obs_buffer_l'] # history of (theta, base_theta)
        }

        self.action_size = np.shape(self.config['stand_up_joint_pos'])[0]
        self.joint_pos = np.zeros(self.action_size)
        self.prev_actions = np.zeros(self.config['action_buffer_l'] * self.action_size)
        self.prev_states = np.zeros(self.config['obs_buffer_l'] * 2)
        self.init_base_angle = None
        self.init_yaw = None
        self.theta = None
        self.base_theta = None
        self.pos = None
        self.vel = None
        self.quat = None
        self.ang_vel = None
        self.roll = self.pitch = self.yaw = None
        self.last_contacts = None
        self.contacts = np.array([True, True, True, True])
        self.feet_air_time = np.zeros(np.size([0.0, 0.0, 0.0, 0.0]))
        self.foot_positions = None
        self.foot_velocities = None

        self.reward_scales = self.config['reward_scales']
        self.reward_dict = {}
        self.command_vel = 0.2
        self.command_yaw = 0.0

        self.reward_container = None
        self.reward_functions = None
        self.rew_buf = None
        self.rew_buf_neg = None
        self.rew_buf_neg = None

        self.gaits = {"pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]}
        self.gait = np.array(self.gaits["trotting"])
        self.control_dt = 0.02 * self.frame_skip
        self.step_frequency_cmd = 3.0
        self.gait_duration = 0.5
        self.gait_indices = 0
        self.foot_indices = None
        self.clock_inputs = np.zeros(4)
        self.doubletime_clock_inputs = np.zeros(4)
        self.halftime_clock_inputs = np.zeros(4)
        
    def step(self, action):
        self.joint_pos = action

        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # populate prev_actions and prev_states buffers
        assert action is not None, "action chosen is None"
        assert self.theta is not None, "theta is None"
        assert self.base_theta is not None, "base_theta is None" 

        self.prev_actions[self.action_size:] = self.prev_actions[:-self.action_size]
        self.prev_actions[:self.action_size] = action
        self.prev_states[2:] = self.prev_states[:-2]
        self.prev_states[:2] = np.concatenate([np.array([self.theta, self.base_theta])]).ravel().astype(np.float32)       

        q_init = np.quaternion(1.0, 0.0, 0.0, 0.0)
        q_final = np.quaternion(self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2], self.data.sensordata[3])
        qd = np.conjugate(q_init) * q_final
        self.theta = 2 * np.arctan2(np.sqrt(qd.x*qd.x + qd.y*qd.y + qd.z*qd.z), qd.w)

        q_init_base = self.init_base_angle
        q_final_base = np.quaternion(self.data.sensordata[4], self.data.sensordata[5], self.data.sensordata[6], self.data.sensordata[7])
        qd_base = np.conjugate(q_init_base) * q_final_base
        self.base_theta = 2 * np.arctan2(np.sqrt(qd_base.x*qd_base.x + qd_base.y*qd_base.y + qd_base.z*qd_base.z), qd_base.w)

        self.pole_quat = self.data.sensor('pole_quat').data.copy()
        self.base_quat = self.data.sensor('base_quat').data.copy()
        self.pos = self.data.sensor('frame_pos').data.copy()
        self.vel = self.data.sensor('frame_vel').data.copy()
        self.quat = self.data.sensor('imu_quat').data.copy()
        self.ang_vel = self.data.sensor('frame_ang_vel').data.copy()

        q_imu = np.quaternion(self.quat[0], self.quat[1], self.quat[2], self.quat[3])
        self.yaw = np.arctan2(2.0*(q_imu.w*q_imu.z + q_imu.x*q_imu.y), 1.0 - 2.0*(q_imu.y*q_imu.y + q_imu.z*q_imu.z))
        self.roll = np.arctan2(2.0*(q_imu.w*q_imu.x + q_imu.y*q_imu.z), 1.0 - 2.0*(q_imu.x*q_imu.x + q_imu.y*q_imu.y))
        self.pitch = np.arcsin(2.0*(q_imu.w*q_imu.y - q_imu.z*q_imu.x))

        self.contact_forces = [  self.data.sensor('FR_contact').data.copy(), 
                                 self.data.sensor('FL_contact').data.copy(), 
                                 self.data.sensor('RR_contact').data.copy(), 
                                 self.data.sensor('RL_contact').data.copy()
                              ]
        self.last_contacts = self.contacts
        self.contacts = all(contact[2] > 10. for contact in self.contact_forces)
        self.foot_positions = [  self.data.sensor('FR_pos').data.copy(), 
                                 self.data.sensor('FL_pos').data.copy(), 
                                 self.data.sensor('RR_pos').data.copy(), 
                                 self.data.sensor('RL_pos').data.copy()
                              ]
        self.foot_velocities = [ self.data.sensor('FR_vel').data.copy(), 
                                 self.data.sensor('FL_vel').data.copy(), 
                                 self.data.sensor('RR_vel').data.copy(), 
                                 self.data.sensor('RL_vel').data.copy()
                               ]
        
        self.dof_vel = []
        self.dof_acc = self.data.qacc 
        self.torques = []
        for i, JOINT_NAME in enumerate(self.config['joint_names']):
            if JOINT_NAME == "pole_joint":
                pass
            else:
                self.torques.append(self.data.sensor(JOINT_NAME[:-5] + "torque").data)
                self.dof_vel.append(self.data.sensor(JOINT_NAME[:-5] + "vel").data)


        observation = self._get_obs()
        terminated = self.get_terminated(observation)

        if terminated:
            reward = self.config['rewards']['termination_reward']
        else:
            reward = self.get_reward()

        info = {"reward_survive": reward,
                "reward_dict" : self.reward_dict}

        self.step_contact_targets()

        if self.render_mode == "human":
            self.render()

        if self.config['verbose']:
            display("INFO", f"Number of geoms in model is:{self.model.ngeom}")
            for i in range(self.model.ngeom):
                display("INFO", f"geometry at index {i} is of type:{self.model.geom(i).type[0]} and size:{self.model.geom(i).size}")
                display("INFO", f"position and orientation of body frame attached to geom at index {i} ({self.model.geom(i).name}) is ({(self.data.geom(i).xmat.reshape((3,3)))},{self.data.geom(i).xpos})")
                display("INFO", f"sensor data:{self.data.sensordata}")

            display("INFO", f"joint_pos: {self.pos}")
            display("INFO", f"joint_vel: {self.vel}")
            display("INFO", f"rpy: {self.roll, self.pitch, self.yaw - self.init_yaw}")
            display("INFO", f"theta: {180 * (self.theta / math.pi)}")
            display("INFO", f"base_theta: {180 * (self.base_theta / math.pi)}")
            display("INFO", f"""Contact forces at each leg:{self.data.sensor('FR_contact').data.copy(), 
                                                            self.data.sensor('FL_contact').data.copy(), 
                                                            self.data.sensor('RR_contact').data.copy(), 
                                                            self.data.sensor('RL_contact').data.copy()}""" )

            q_ = [self.data.joint('pole_joint').qpos[0], self.data.joint('pole_joint').qpos[1], self.data.joint('pole_joint').qpos[2], self.data.joint('pole_joint').qpos[3]]
            for i, pos_ in enumerate(self.data.qpos):
                if pos_ == q_[0]:
                    if self.data.qpos[i+1] == q_[1]:
                        if self.data.qpos[i+2] == q_[2]:
                            if self.data.qpos[i+3] == q_[3]:
                                display("INFO", f"pole_joint qpos at index {i}")

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):

        mj.set_mjcb_control(self.controller)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        #IMPORTANT: setting larger noise (10 times reset_scale_noise) in pole joint, so as to start it slightly off the upwards pose
        qpos[7:11] = self.init_qpos[7:11] + self.np_random.uniform(
            size=4, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        self.init_base_angle = np.quaternion(self.data.sensordata[4], self.data.sensordata[5], self.data.sensordata[6], self.data.sensordata[7])

        q_init = np.quaternion(1.0, 0.0, 0.0, 0.0)
        q_final = np.quaternion(self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2], self.data.sensordata[3])
        qd = np.conjugate(q_init) * q_final
        self.theta = 2 * np.arctan2(np.sqrt(qd.x*qd.x + qd.y*qd.y + qd.z*qd.z), qd.w)
        # print((theta / math.pi) * 180)

        q_init_base = self.init_base_angle
        q_final_base = np.quaternion(self.data.sensordata[4], self.data.sensordata[5], self.data.sensordata[6], self.data.sensordata[7])
        qd_base = np.conjugate(q_init_base) * q_final_base
        self.base_theta = 2 * np.arctan2(np.sqrt(qd_base.x*qd_base.x + qd_base.y*qd_base.y + qd_base.z*qd_base.z), qd_base.w)

        curr_quat = self.data.sensor('imu_quat').data.copy()
        q_imu = np.quaternion(curr_quat[0], curr_quat[1], curr_quat[2], curr_quat[3])
        self.init_yaw = np.arctan2(2.0*(q_imu.w*q_imu.z + q_imu.x*q_imu.y), 1.0 - 2.0*(q_imu.y*q_imu.y + q_imu.z*q_imu.z))   

        self.init_rewards()
        self.step_contact_targets()

        return self._get_obs()

    def _get_obs(self):
        pos = self.data.sensor('frame_pos').data[:2].copy()   
        vel = self.data.sensor('frame_vel').data[:2].copy()
        quat = self.data.sensor('imu_quat').data.copy()

        q = np.quaternion(quat[0], quat[1], quat[2], quat[3])
        yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        assert self.theta is not None, "self.theta is None, expected a float64"   
        assert self.base_theta is not None, "self.base_theta is None, expected a float64"   
        ret  = np.concatenate([np.array([self.theta, self.base_theta]), self.data.qpos, self.data.qvel, self.prev_actions, self.prev_states]).ravel().astype(np.float32)
        return ret

    def controller(self, model, data):
        #pd controller : takes error and desired velocity as input, outputs the instantaneous torque
        kp = 50 #TODO: kp, kd params from config
        kd = 2

        for i, JOINT_NAME in enumerate(self.config['joint_names']):
            if JOINT_NAME == "pole_joint":
                pass
            else:
                dq = (self.config['sensitivity'] * self.joint_pos[i] + self.config['stand_up_joint_pos'][i]) - self.data.joint(JOINT_NAME).qpos[0]
                dv = -self.data.joint(JOINT_NAME).qvel[0]

                tau = kp * dq + kd * dv
                self.data.ctrl[i] = tau
    
    def _set_action_space(self):
        self.action_space = spaces.Box(low=-3, high=3, shape=(12,), dtype=np.float32)
        return self.action_space
    
    def get_terminated(self, observation):
        if not np.isfinite(observation).all():
            self.reward_dict["infinite_obs"] = -1
            return True
        elif np.abs(self.theta) > self.config['tipping_angle']:
            self.reward_dict["pend_tipping_penalty"] = -1
            return True
        elif np.abs(self.base_theta) > self.config['tipping_base_angle']:
            self.reward_dict["base_tipping_penalty"] = -1
            return True

        return False

    def init_rewards(self):
        from .rewards import Rewards
        self.reward_container = Rewards(self, self.config)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= 1

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                # display("WARNING", f"reward {'_reward_' + name} has nonzero coefficient but was not found!")
                pass
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

    def get_reward(self):
        self.rew_buf = 0.
        self.rew_buf_pos = 0.
        self.rew_buf_neg = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.reward_dict[name] = rew
            self.rew_buf += rew
            if np.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif np.sum(rew) <= 0:
                self.rew_buf_neg += rew
        
        # print(self.rew_buf_neg)
        # print(self.rew_buf_pos)
        self.rew_buf = self.rew_buf_pos * np.exp(self.rew_buf_neg / self.config['rewards']['sigma_rew_neg'])

        return self.rew_buf


    def step_contact_targets(self):
        frequencies = self.step_frequency_cmd
        phases = self.gait[0]
        offsets = self.gait[1]
        bounds = self.gait[2]
        durations = self.gait_duration
        self.gait_indices = np.remainder(self.gait_indices + self.control_dt * frequencies, 1.0)
        #TODO: initialize gait indices to some value right now it is initialized as None

        self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = np.remainder(self.foot_indices, 1.0)

        stance_idxs = np.remainder(self.foot_indices, 1) < durations
        swing_idxs = np.remainder(self.foot_indices, 1) > durations
        self.foot_indices[stance_idxs] = np.remainder(self.foot_indices[stance_idxs], 1) * (0.5 / durations)
        self.foot_indices[swing_idxs] = 0.5 + (np.remainder(self.foot_indices[swing_idxs], 1) - durations) * (
                    0.5 / (1 - durations))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[0] = np.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[1] = np.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[2] = np.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[3] = np.sin(2 * np.pi * self.foot_indices[3])

        self.doubletime_clock_inputs[0] = np.sin(4 * np.pi * self.foot_indices[0])
        self.doubletime_clock_inputs[1] = np.sin(4 * np.pi * self.foot_indices[1])
        self.doubletime_clock_inputs[2] = np.sin(4 * np.pi * self.foot_indices[2])
        self.doubletime_clock_inputs[3] = np.sin(4 * np.pi * self.foot_indices[3])

        self.halftime_clock_inputs[0] = np.sin(np.pi * self.foot_indices[0])
        self.halftime_clock_inputs[1] = np.sin(np.pi * self.foot_indices[1])
        self.halftime_clock_inputs[2] = np.sin(np.pi * self.foot_indices[2])
        self.halftime_clock_inputs[3] = np.sin(np.pi * self.foot_indices[3])

        # von mises distribution
        kappa = self.config['rewards']['kappa_gait_probs']

        smoothing_cdf_start = norm(loc=0, scale=kappa).cdf

        smoothing_multiplier_FL = (
            smoothing_cdf_start(np.remainder(self.foot_indices[0], 1.0)) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[0], 1.0) - 0.5)) +
            smoothing_cdf_start(np.remainder(self.foot_indices[0], 1.0) - 1) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[0], 1.0) - 0.5 - 1))
        )

        smoothing_multiplier_FR = (
            smoothing_cdf_start(np.remainder(self.foot_indices[1], 1.0)) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[1], 1.0) - 0.5)) +
            smoothing_cdf_start(np.remainder(self.foot_indices[1], 1.0) - 1) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[1], 1.0) - 0.5 - 1))
        )

        smoothing_multiplier_RL = (
            smoothing_cdf_start(np.remainder(self.foot_indices[2], 1.0)) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[2], 1.0) - 0.5)) +
            smoothing_cdf_start(np.remainder(self.foot_indices[2], 1.0) - 1) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[2], 1.0) - 0.5 - 1))
        )

        smoothing_multiplier_RR = (
            smoothing_cdf_start(np.remainder(self.foot_indices[3], 1.0)) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[3], 1.0) - 0.5)) +
            smoothing_cdf_start(np.remainder(self.foot_indices[3], 1.0) - 1) *
            (1 - smoothing_cdf_start(np.remainder(self.foot_indices[3], 1.0) - 0.5 - 1))
        )

        self.desired_contact_states = np.zeros_like(self.foot_indices)
        self.desired_contact_states[0] = smoothing_multiplier_FL
        self.desired_contact_states[1] = smoothing_multiplier_FR
        self.desired_contact_states[2] = smoothing_multiplier_RL
        self.desired_contact_states[3] = smoothing_multiplier_RR
