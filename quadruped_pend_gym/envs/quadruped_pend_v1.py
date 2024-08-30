__credits__ = ["Pulak-Gautam"]

from typing import Dict, Union

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

from quadruped_pend_gym.envs.utils import XmlGenerator, display, phi_F

class QuadrupedPendEnv_v1(MujocoEnv, utils.EzPickle):
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
        xml_file: str = "./quadruped_pend_gym/models/go2/scene.xml", #TODO: add robot model choice in config (eg. config['robot'] = go2)
        frame_skip: int = 2, 
        default_camera_config: Dict[str, Union[float, int]] = None,
        reset_noise_scale: float = None,
        config_file: str = None,
        **kwargs,
    ):
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except:
            display("WARNING", "Yaml file not found, using default config")
            with open("./quadruped_pend_gym/config/env_config.yaml", 'r') as file:
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

        observation_space = Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float64)

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
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "prev_action" : np.size(self.config['joint_names']) - 1,
            "prev_prev_action" : np.size(self.config['joint_names']) - 1
        }

        self.joint_pos = np.zeros(np.size(self.config['stand_up_joint_pos']))

        self.prev_actions = [
            np.zeros(np.size(self.config['stand_up_joint_pos'])),
            np.zeros(np.size(self.config['stand_up_joint_pos'])),
        ]

        self.init_base_angle = None
        self.init_yaw = None

        self.reward_dict = {
            "balance_reward" : 0.0,

            "joint_torques_penalty" : 0.0,
            "joint_vel_penalty"  : 0.0,
            "joint_acc_penalty"  : 0.0,

            "pend_tipping_penalty"  : 0.0,
            "base_tipping_penalty" : 0.0,

            "contact_force_penalty" : 0.0,
            "infinite_obs"  : 0.0
        }
    
    def step(self, action):

        if self.config['verbose']:
            display("INFO", f"Number of geoms in model is:{self.model.ngeom}")
            for i in range(self.model.ngeom):
                display("INFO", f"geometry at index {i} is of type:{self.model.geom(i).type[0]} and size:{self.model.geom(i).size}")
                display("INFO", f"position and orientation of body frame attached to geom at index {i} ({self.model.geom(i).name}) is ({(self.data.geom(i).xmat.reshape((3,3)))},{self.data.geom(i).xpos})")
                display("INFO", f"sensor data:{self.data.sensordata}")
            print("\n")
        
        self.joint_pos = action

        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        self.prev_actions[1] = self.prev_actions[0] 
        self.prev_actions[0] = action

        observation = self._get_obs()

        q_init = np.quaternion(1.0, 0.0, 0.0, 0.0)
        q_final = np.quaternion(self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2], self.data.sensordata[3])
        qd = np.conjugate(q_init) * q_final
        theta = 2 * np.arctan2(np.sqrt(qd.x*qd.x + qd.y*qd.y + qd.z*qd.z), qd.w)
        # print((theta / math.pi) * 180)

        q_init_base = self.init_base_angle
        q_final_base = np.quaternion(self.data.sensordata[4], self.data.sensordata[5], self.data.sensordata[6], self.data.sensordata[7])
        qd_base = np.conjugate(q_init_base) * q_final_base
        base_theta = 2 * np.arctan2(np.sqrt(qd_base.x*qd_base.x + qd_base.y*qd_base.y + qd_base.z*qd_base.z), qd_base.w)
        # print((base_theta / math.pi) * 180)

        pos = self.data.sensor('frame_pos').data[:2].copy()   
        vel = self.data.sensor('frame_vel').data[:2].copy()
        quat = self.data.sensor('imu_quat').data.copy()

        q_imu = np.quaternion(quat[0], quat[1], quat[2], quat[3])
        yaw = np.arctan2(2.0*(q_imu.w*q_imu.z + q_imu.x*q_imu.y), 1.0 - 2.0*(q_imu.y*q_imu.y + q_imu.z*q_imu.z))
        roll = np.arctan2(2.0*(q_imu.w*q_imu.x + q_imu.y*q_imu.z), 1.0 - 2.0*(q_imu.x*q_imu.x + q_imu.y*q_imu.y))
        pitch = np.arcsin(2.0*(q_imu.w*q_imu.y - q_imu.z*q_imu.x))

        contact_F = np.linalg.norm(self.data.sensordata[-4:])
        
        joint_vel = []
        joint_acc = self.data.qacc 
        joint_torques = []
        for i, JOINT_NAME in enumerate(self.config['joint_names']):
            if JOINT_NAME == "pole_joint":
                pass
            else:
                joint_torques.append(self.data.sensor(JOINT_NAME[:-5] + "torque").data)
                joint_vel.append(self.data.sensor(JOINT_NAME[:-5] + "vel").data)

        if self.config['verbose']:
            display("INFO", f"joint_pos: {pos}")
            display("INFO", f"joint_vel: {vel}")
            display("INFO", f"rpy: {roll, pitch, yaw - self.init_yaw}")
                
        terminated = self.get_terminated(observation, theta, base_theta, contact_F)

        if terminated:
            reward = -1
        else:
            reward = self.get_reward(theta, contact_F, joint_torques, joint_vel, joint_acc)

        info = {"reward_survive": reward,
                "reward_dict" : self.reward_dict}

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):

        mj.set_mjcb_control(self.controller)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        self.init_base_angle = np.quaternion(self.data.sensordata[4], self.data.sensordata[5], self.data.sensordata[6], self.data.sensordata[7])

        curr_quat = self.data.sensor('imu_quat').data.copy()
        q_imu = np.quaternion(curr_quat[0], curr_quat[1], curr_quat[2], curr_quat[3])
        self.init_yaw = np.arctan2(2.0*(q_imu.w*q_imu.z + q_imu.x*q_imu.y), 1.0 - 2.0*(q_imu.y*q_imu.y + q_imu.z*q_imu.z))   

        return self._get_obs()

    def _get_obs(self):

        pos = self.data.sensor('frame_pos').data[:2].copy()   
        vel = self.data.sensor('frame_vel').data[:2].copy()
        quat = self.data.sensor('imu_quat').data.copy()

        q = np.quaternion(quat[0], quat[1], quat[2], quat[3])
        yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

        return np.concatenate([self.data.qpos, self.data.qvel, self.prev_actions[0], self.prev_actions[1]]).ravel()

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
        self.action_space = spaces.Box(low=-2, high=2, shape=(12,), dtype=np.float32)
        return self.action_space

    def get_reward(self, theta, contact_F, curr_torq, curr_joint_vel, curr_joint_acc):
        self.reward_dict["balance_reward"] = self.config['r_theta_tracking'] - np.abs((theta - 0.0)) * np.exp(-1)
        self.reward_dict["joint_torques_penalty"] = self.config['r_joint_torques_penalty'] * np.linalg.norm(curr_torq)
        self.reward_dict["joint_acc_penalty"] = self.config['r_joint_acc_penalty'] * np.linalg.norm(curr_joint_acc)
        self.reward_dict["joint_vel_penalty"] = self.config['r_joint_vel_penalty'] * np.linalg.norm(curr_joint_vel)
        self.reward_dict["contact_force_penalty"] = self.config['r_contact_force_penalty'] * contact_F
        self.reward_dict["pend_tipping_penalty"] = 0.0
        self.reward_dict["base_tipping_penalty"] = 0.0
        self.reward_dict["infinite_obs"] = 0.0

        if self.config['verbose']:
            display("INFO", f"reward_dict: {self.reward_dict}")

        return sum(self.reward_dict.values())
        
    def get_terminated(self, observation, theta, base_theta, contact_F):
        if not np.isfinite(observation).all():
            self.reward_dict["infinite_obs"] = -1
            return True
        elif np.abs(theta) > self.config['tipping_angle']:
            self.reward_dict["pend_tipping_penalty"] = -1
            return True
        elif np.abs(base_theta) > self.config['tipping_base_angle']:
            self.reward_dict["base_tipping_penalty"] = -1
            return True

        return False
