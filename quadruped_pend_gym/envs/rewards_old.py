
import numpy as np
from types import SimpleNamespace
from quadruped_pend_gym.envs.utils import display
import quaternion

class Rewards:
    def __init__(self, env):
        self.env = env
        
        self._cfg = SimpleNamespace(
            sigma_rew_neg=5,
            tracking_sigma=0.25,
            max_contact_force=100.0,
            terminal_body_ori=0.5,
            kappa_gait_probs=0.07,
            gait_force_sigma=50.0,
            gait_vel_sigma=0.5
        )
        
        self._coeff = SimpleNamespace( 
            # max = 4.5

            balance = 1,
            tracking_lin_vel = 1,
            tracking_ang_vel = 0.5,

            orientation_control = -0.1,
            lin_vel_z = -4.0,
            ang_vel_xy = -0.05,

            feet_airtime = 2,

            feet_contact_forces = -0.0,
            torques = -0.00002,
            dof_vel = -0.001,
            dof_acc = -0.001,

            stand_still = -0.0, #TODO: enable this when we varying commanded velocity

            tracking_contacts_shaped_force = -4.0,
            tracking_contacts_shaped_vel = 1.0,

            raibert_heuristic=1.0,
            action_smoothness_1=1.0,
            action_smoothness_2=1.0,
            feet_clearance_cmd_linear=1.0
        )
        
        self._reward_values = {}



    def _reward_balance(self):
        """Balance reward"""
        reward = np.exp(-np.square(self.env.theta - 0.0) / self._cfg.tracking_sigma)

        reward = self._coeff.balance * reward
        self._reward_values['balance'] = reward 
        return reward



    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        reward = np.exp(-np.square(self.env.vel[0] - self.command_vel[0]) / self._cfg.tracking_sigma)

        reward = self._coeff.tracking_lin_vel * reward
        self._reward_values['tracking_lin_vel'] = reward
        return reward



    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        reward = np.exp(-np.square(self.env.ang_vel[0] - self.command_ang_vel[0]) / self._cfg.tracking_sigma)

        reward = self._coeff.tracking_ang_vel * reward 
        self._reward_values['tracking_ang_vel'] = reward
        return reward



    def _reward_orientation_control(self):
        """Penalize deviation from initial base angle"""
        reward = np.square(self.env.base_theta - 0.0)

        reward = self._coeff.orientation_control * reward
        self._reward_values['orientation_control'] = reward
        return reward



    def _reward_lin_vel_z(self):
        """Penalize z-axis base linear velocity"""
        reward = np.square(self.env.vel[2])

        reward = self._coeff.lin_vel_z * reward
        self._reward_values['lin_vel_z'] = reward
        return reward



    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        reward = np.linalg.norm(self.env.ang_vel[:2])**2

        reward = self._coeff.ang_vel_xy * reward
        self._reward_values['ang_vel_xy'] = reward
        return reward



    def _reward_feet_airtime(self):
        """Reward for long steps"""
        contact_filt = np.logical_or(self.env.contacts, self.env.last_contacts) 
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt 
        reward = np.sum((self.env.feet_air_time - 0.5) * first_contact) # reward only on first contact with the ground
        self.env.feet_air_time *= ~contact_filt # update airtime
        
        reward = self._coeff.feet_airtime * reward
        self.env._reward_values['feet_airtime'] = reward
        return reward



    def _reward_feet_contact_forces(self):
        """Penalty for high contact forces"""
        reward = np.sum((np.norm(self.env.contact_forces, axis=1) - self._cfg.max_contact_force).clip(0.))

        reward = self._coeff.feet_contact_forces * reward
        self._reward_values['feet_contact_forces'] = reward
        return reward



    def _reward_torques(self):
        """Penalty for joint torques"""
        reward = np.linalg.norm(self.env.torques)**2

        reward = self._coeff.torques * reward
        self._reward_values['torques'] = reward
        return reward



    def _reward_dof_vel(self):
        """Penalty for degree-of-freedom velocities"""
        reward = np.linalg.norm(self.env.dof_vel)**2

        reward = self._coeff.dof_vel * reward
        self._reward_values['dof_vel'] = reward
        return reward



    def _reward_dof_acc(self):
        """Penalty for degree-of-freedom accelerations"""
        reward = np.linalg.norm(self.env.dof_acc)**2

        reward = self._coeff.dof_acc * reward
        self._reward_values['dof_acc'] = reward
        return reward



    def _reward_stand_still(self):
        """Penalty for motion when commands are zero"""
        reward = np.sum(np.abs(self.vel - 0.) * (np.norm(self.command_vel) < 0.001))

        reward = self._coeff.stand_still * reward
        self._reward_values['stand_still'] = reward
        return reward



    def _reward_tracking_contacts_shaped_force(self):
        """Reward for tracking contacts shaped by force"""
        foot_forces = np.norm(self.env.contact_forces, axis=1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[i]) * (1 - np.exp(-1 * foot_forces[i] ** 2 / self._cfg.gait_force_sigma))
        reward = reward / 4

        reward = self._coeff.tracking_contacts_shaped_force * reward
        self._reward_values['tracking_contacts_shaped_force'] = reward
        return reward



    def _reward_tracking_contacts_shaped_vel(self): 
        """Reward for tracking contacts shaped by velocity"""
        foot_velocities = np.norm(self.env.foot_velocities, axis=1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (1 - np.exp(-1 * foot_velocities[i] ** 2 / self.env._cfg.gait_vel_sigma)))
        reward = reward / 4

        reward = self._coeff.tracking_contacts_shaped_vel * reward
        self._reward_values['tracking_contacts_shaped_vel'] = reward
        return reward



    def _reward_raibert_heuristic(self): 
        #TODO: implement env.foot_velocites, env.desired_contact_states
        #TODO: implement env.foot_positions, env.command_freq, 
        #TODO: adding body-frame gravity vector in obs, adding timing ref var in obs
        
        """Reward based on Raibert's heuristic"""
        cur_footsteps_translated = self.env.foot_positions - self.env.pos
        footsteps_in_body_frame = np.zeros(4, 3)

        quat = np.quaternion(self.env.base_quat)
        quat_yaw = quat.clone()
        quat_yaw.x = 0.
        quat_yaw.y = 0.

        for i in range(4):
            footsteps_in_body_frame[i] =  np.rotate_vectors(quat_yaw.conjugate(), cur_footsteps_translated[i])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.3
        desired_ys_nom = np.array([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2])

        desired_stance_length = 0.45
        desired_xs_nom = np.array([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2])

        # raibert offsets
        phases = np.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
        self._reward_values['raibert_heuristic'] = reward
        return reward

    def _reward_action_smoothness_1(self):
        """Reward for action smoothness (first order)"""
        at = self.env.prev_actions[:self.env.action_size]
        at_1 = self.env.prev_actions[self.env.action_size:2*self.env.action_size]

        diff = np.square(at - at_1)
        diff = diff * (at_1 != 0)  # ignore first step
        reward = np.sum(diff)

        reward = self._coeff.action_smoothness_1 * reward
        self._reward_values['action_smoothness_1'] = reward
        return reward

    def _reward_action_smoothness_2(self):
        """Reward for action smoothness (second order)"""
        at = self.env.prev_actions[:self.env.action_size]
        at_1 = self.env.prev_actions[self.env.action_size:2*self.env.action_size]
        at_2 = self.env.prev_actions[2*self.env.action_size:3*self.env.action_size]

        diff = np.square(at - 2 * at_1 + at_2)
        diff = diff * (at_1 != 0)  # ignore first step
        diff = diff * (at_2 != 0)  # ignore second step
        reward = np.sum(diff)

        reward = self._coeff.action_smoothness_2 * reward
        self._reward_values['action_smoothness_2'] = reward
        return reward

    def calculate_total_reward(self):
        """Calculate the total reward by summing all individual reward values."""
        self._reward_balance()
        self._reward_tracking_lin_vel()
        self._reward_tracking_ang_vel()
        self._reward_orientation_control()
        self._reward_lin_vel_z()
        self._reward_ang_vel_xy()
        self._reward_feet_airtime()
        self._reward_feet_contact_forces()
        self._reward_torques()
        self._reward_dof_vel()
        self._reward_dof_acc()
        self._reward_motion_at_zero_commands()
        self._reward_tracking_contacts_shaped_force()
        self._reward_tracking_contacts_shaped_vel()
        self._reward_raibert_heuristic()
        self._reward_action_smoothness_1()
        self._reward_action_smoothness_2()
        self._reward_feet_clearance_cmd_linear()
        
        return sum(self._reward_values.values())

    
        # self.reward_dict = {
        #     "linear_vel_tracking" : 0.0, # Tracking of linear velocity commands (xy axes)
        #     "angular_vel_tracking" : 0.0, # Tracking of angular velocity commands (yaw) 
        #     "balance_reward" : 0.0,

        #     "linear_vel_penalty" : 0.0, # Penalize z axis base linear velocity
        #     "angular_vel_penalty" : 0.0, # Penalize xy axes base angular velocity
        #     "feet_air_time_reward": 0.0, # Reward long steps
        #     "contact_force_penalty" : 0.0,
        #     "joint_torques_penalty" : 0.0, # Penalize torques
        #     "joint_vel_penalty"  : 0.0, # Penalize dof velocities
        #     "joint_acc_penalty"  : 0.0, # Penalize dof accelerations

        #     "pend_tipping_penalty"  : 0.0,
        #     "base_tipping_penalty" : 0.0, # Penalize non flat base orientation
            
        #     "infinite_obs"  : 0.0

        #     # Missed out reward terms:
        #     # - Penalize collisions on selected bodies
        #     # - Terminal reward / penalty
        #     # - Penalize dof positions too close to the limit 
        #     # - Penalize dof velocities too close to the limit
        #     # - Penalize torques too close to the limit
        #     # - reward_feet_impact_vel

        #     # - Penalize motion at zero commands -> TODO
        #     # - penalize high contact forces -> TODO

        #     # - reward_tracking_contacts_shaped_force -> TODO
        #     # - reward_tracking_contacts_shaped_vel -> TODO
        #     # - reward_raibert_heuristic -> TODO
        
        #     # - reward_action_smoothness_1 -> TODO
        #     # - reward_action_smoothness_2 -> TODO
        #     # - reward_feet_clearance_cmd_linear -> TODO
        #     # - reward_orientation_control -> TODO
        # }
######### reward functions

    # def get_reward(self):
    #     assert self.contacts is not None, "contacts are None"
    #     assert self.last_contacts is not None, "last_contacts are None"

    #     self.reward_dict["balance_reward"] = self.config['r_theta_tracking'] - np.abs((self.theta - 0.0)) * np.exp(-1)
    #     self.reward_dict["joint_torques_penalty"] = self.config['r_joint_torques_penalty'] * np.linalg.norm(self.torques)
    #     self.reward_dict["joint_acc_penalty"] = self.config['r_joint_acc_penalty'] * np.linalg.norm(self.dof_acc)
    #     self.reward_dict["joint_vel_penalty"] = self.config['r_joint_vel_penalty'] * np.linalg.norm(self.dof_vel)
    #     self.reward_dict["contact_force_penalty"] = self.config['r_contact_force_penalty'] * np.max(np.linalg.norm(self.contact_forces[:, 2]))
    #     self.reward_dict["feet_air_time_reward"] = self.config['r_feet_air_time'] * self.get_air_time()

    #     self.reward_dict["linear_vel_tracking"] = self.config['r_linear_vel_tracking'] - np.abs((self.vel[0] - self.command_vel)) * np.exp(-1)
    #     self.reward_dict["angular_vel_tracking"] = self.config['r_angular_vel_tracking'] - np.abs((self.yaw - self.command_yaw)) * np.exp(-1)
    #     self.reward_dict["linear_vel_penalty"] = self.config['r_linear_vel_penalty'] * np.linalg.norm(self.vel[2])
    #     self.reward_dict["angular_vel_penalty"] = self.config['r_angular_vel_penalty'] * np.linalg.norm([self.roll, self.pitch]) #keep the coeff small (for balancing bending is needed)

    #     self.reward_dict["pend_tipping_penalty"] = 0.0
    #     self.reward_dict["base_tipping_penalty"] = 0.0
    #     self.reward_dict["infinite_obs"] = 0.0

    #     if self.config['verbose']:
    #         display("INFO", f"reward_dict: {self.reward_dict}")

    #     return sum(self.reward_dict.values())

    # def get_air_time(self):
    #     contact_filt = np.logical_or(self.contacts, self.last_contacts) 
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt

    #     display("INFO", f"Feet contacts: {contact_filt}, Feet air times: {self.feet_air_time}")

    #     # reward only on first contact with the ground
    #     rew_airTime = np.sum((self.feet_air_time - 0.5) * first_contact)

    #     # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #TODO:no reward for zero command
        
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime

#   use_terminal_body_height: false
#   terminal_body_height: 0.20
#   use_terminal_foot_height: false
#   terminal_foot_height: -0.005
#   use_terminal_roll_pitch: false
#   terminal_body_ori: 0.5

# env.commands[:, 0] = x_vel_cmd
# env.commands[:, 1] = y_vel_cmd
# env.commands[:, 2] = yaw_vel_cmd
# env.commands[:, 3] = body_height_cmd
# env.commands[:, 4] = step_frequency_cmd
# env.commands[:, 5:8] = gait
# env.commands[:, 8] = 0.5
# env.commands[:, 9] = footswing_height_cmd
# env.commands[:, 10] = pitch_cmd
# env.commands[:, 11] = roll_cmd
# env.commands[:, 12] = stance_width_cmd   base_height_target: 0.30