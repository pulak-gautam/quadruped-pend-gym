import numpy as np
from types import SimpleNamespace
from quadruped_pend_gym.envs.utils import display
import quaternion

class Rewards:
    def __init__(self, env, config):
        self.env = env
        self.cfg = config

    def _reward_pole_balance(self):
        """Balance reward"""
        return np.exp(-np.square(self.env.theta - 0.0) / self.cfg['rewards']['tracking_sigma'])

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (x axes)"""
        return np.exp(-np.square(self.env.vel[0] - self.env.command_vel) / self.cfg['rewards']['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        return np.exp(-np.square(self.env.ang_vel[2] - self.env.command_yaw) / self.cfg['rewards']['tracking_sigma'])

    def _reward_orientation_control(self):
        """Penalize deviation from initial base angle"""
        return np.exp(-np.square(self.env.base_theta - 0.0) / self.cfg['rewards']['tracking_sigma'])

    def _reward_lin_vel_z(self):
        """Penalize z-axis base linear velocity"""
        return np.square(self.env.vel[2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        return np.sum(np.square(self.env.ang_vel[:2]))

    def _reward_torques(self):
        """Penalty for joint torques"""
        return np.sum(np.square(self.env.torques))

    def _reward_dof_acc(self):
        """Penalty for degree-of-freedom accelerations"""
        return np.sum(np.square(self.env.dof_acc))

    def _reward_tracking_contacts_shaped_force(self):
        """Reward for tracking contacts shaped by force"""
        foot_forces = np.linalg.norm(self.env.contact_forces, axis=1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[i]) * (1 - np.exp(-1 * foot_forces[i] ** 2 / self.cfg['rewards']['gait_force_sigma']))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self): 
        """Reward for tracking contacts shaped by velocity"""
        foot_velocities = np.linalg.norm(self.env.foot_velocities, axis=1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (desired_contact[i] * (1 - np.exp(-1 * foot_velocities[i] ** 2 / self.cfg['rewards']['gait_vel_sigma'])))
        return reward / 4

    def _reward_dof_vel(self):
        """Penalty for degree-of-freedom velocities"""
        return np.sum(np.square(self.env.dof_vel))

    def _reward_action_smoothness_1(self):
        """Reward for action smoothness (first order)"""
        at = self.env.prev_actions[:self.env.action_size]
        at_1 = self.env.prev_actions[self.env.action_size:2*self.env.action_size]

        diff = np.square(at - at_1)
        diff = diff * (at_1 != 0)  # ignore first step
        return np.sum(diff)

    def _reward_action_smoothness_2(self):
        """Reward for action smoothness (second order)"""
        at = self.env.prev_actions[:self.env.action_size]
        at_1 = self.env.prev_actions[self.env.action_size:2*self.env.action_size]
        at_2 = self.env.prev_actions[2*self.env.action_size:3*self.env.action_size]

        diff = np.square(at - 2 * at_1 + at_2)
        diff = diff * (at_1 != 0)  # ignore first step
        diff = diff * (at_2 != 0)  # ignore second step
        return np.sum(diff)

    def _reward_feet_air_time(self):
        """Reward for long steps"""
        contact_filt = np.logical_or(self.env.contacts, self.env.last_contacts) 
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt 
        reward = np.sum((self.env.feet_air_time - 0.5) * first_contact) # reward only on first contact with the ground
        self.env.feet_air_time *= ~contact_filt # update airtime
        
        reward = self._coeff.feet_airtime * reward
        return reward

    def _reward_feet_contact_forces(self):
        """Penalty for high contact forces"""
        return np.sum((np.norm(self.env.contact_forces, axis=1) - self.cfg['rewards']['max_contact_force']).clip(0.))

    def _reward_raibert_heuristic(self): 
        #TODO: adding body-frame gravity vector in obs
        
        """Reward based on Raibert's heuristic"""
        cur_footsteps_translated = self.env.foot_positions - self.env.pos
        footsteps_in_body_frame = np.zeros((4, 3))

        quat = np.quaternion(self.env.base_quat[0], self.env.base_quat[1], self.env.base_quat[2], self.env.base_quat[3])
        quat_yaw = quat.copy()
        quat_yaw.x = 0.
        quat_yaw.y = 0.

        for i in range(4):
            footsteps_in_body_frame[i, :] = quaternion.rotate_vectors(quat_yaw.conjugate(), cur_footsteps_translated[i])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.3
        desired_ys_nom = np.array([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2])

        desired_stance_length = 0.45
        desired_xs_nom = np.array([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2])

        # raibert offsets
        phases = np.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        # print(self.env.foot_indices)
        frequencies = self.env.step_frequency_cmd
        x_vel_des = self.env.command_vel
        yaw_vel_des = self.env.command_yaw
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
        desired_ys_offset[2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = -(desired_xs_nom + desired_xs_offset)

        desired_footsteps_body_frame = np.concatenate(
            (np.expand_dims(desired_xs_nom, axis=1), np.expand_dims(desired_ys_nom, axis=1)),
            axis=1
        )
        err_raibert_heuristic = np.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, 0:2])

        reward = np.sum(np.square(err_raibert_heuristic), axis=1)
        reward = np.sum(reward, axis=0)
        return reward
