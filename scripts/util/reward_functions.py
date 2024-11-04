from dm_control.utils import rewards
import numpy as np
import math
'''
this file attempted to restore several reward function of dmcs and openai gym 
based on |textbf{observations}

Recovered: 
dmcs: finger_turn-hard, fish-swim, reacher-hard. 
openai_gym: swimmer-v5, walker2d-v5, hopper-v5, halfcheetah-v5

Impossible to recover: 
The following environments cannot be restored. 
Openai gym: Ant-v5, Humanoid-v5, 
DMCS: The all rest 7 environments are hard to restore. 

Tips:
First thing first, for openai-gym, 'exclude_current_positions_from_observation' 
needs to be set to False !!!!!!
 
For dmcs, it is generally hard to recover reward functions since the components
are from mujoco simulation rather than the observation. 
For openai gym, things are easier. 

ToDos:
The following environments can be restored but ignored:
Openai Gym: Pusher-v5, Pendulum-v5, Reacher-v5

Link to find where is the reward (just search reward and follow the function calls):

https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/ball_in_cup.py
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v4.py

'''

def get_openai_swimmer_reward(prev_obs, action, curr_obs):
    xy_position_before = prev_obs[0:2].copy()
    xy_position_after = curr_obs[0:2].copy()
    xy_velocity = (xy_position_after - xy_position_before) / 0.04
    x_velocity, y_velocity = xy_velocity
    forward_reward = x_velocity
    ctrl_cost = 0.0001 * np.sum(np.square(action))
    reward = forward_reward - ctrl_cost
    return reward


def get_openai_walker_reward(prev_obs, action, curr_obs):
    dt = 0.008
    x_position_before = prev_obs[0]
    x_position_after = curr_obs[0]
    forward_reward = (x_position_after - x_position_before) / dt
    def is_healthy (ddata):
        z, angle = ddata[1:3]
        min_z, max_z = [0.8, 2.0]
        min_angle, max_angle = [-1, 1]
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle
        return is_healthy

    healthy_reward = is_healthy(curr_obs)
    ctrl_cost = 0.001 * np.sum(np.square(action))
    reward = forward_reward + healthy_reward - ctrl_cost
    return reward


def get_openai_hopper_reward(prev_obs, action, curr_obs):
    dt = 0.008
    x_position_before = prev_obs[0]
    x_position_after = curr_obs[0]
    x_velocity = (x_position_after - x_position_before) / dt
    def is_healthy(ddata):
        z, angle = ddata[1:3]
        # state_vector = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        state = ddata[2:]  # Mujoco's qvel + qpos
        min_state, max_state = [-100, 100]
        min_z, max_z = [0.7, math.inf]
        min_angle, max_angle = [-0.2, 0.2]
        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = all((healthy_state, healthy_z, healthy_angle))
        return is_healthy
    healthy_reward = is_healthy(curr_obs)
    ctrl_cost = 0.001 * np.sum(np.square(action))
    reward = x_velocity + healthy_reward - ctrl_cost
    return reward


def get_openai_halfcheetah_reward(prev_obs, act, curr_obs):
    indi = 0
    dt = 0.05
    x_position_before = prev_obs[indi]
    x_position_after = curr_obs[indi]
    x_velo = (x_position_after - x_position_before) / dt
    forward_reward = 1.0 * x_velo
    ctrl_cost = 0.1 * np.sum(np.square(act))
    rwd = forward_reward - ctrl_cost
    return rwd


def get_dmcs_fish_reward(env, observation):
    radii = env.env.physics.named.model.geom_size[['mouth', 'target'], 0].sum()
    in_target = rewards.tolerance(np.linalg.norm(observation[8:11]),
                                  bounds=(0, radii), margin=2 * radii)
    is_upright = 0.5 * (observation[7] + 1)
    return (7 * in_target + is_upright) / 8


def get_dmcs_finger_reward(env, observation):
    return float(observation[11] <= 0)


def get_dmcs_reacher_reward(env, observation):
    # Working!
    radii = env.env.physics.named.model.geom_size[['target', 'finger'], 0].sum()
    dist = np.linalg.norm(observation[2:4])
    return rewards.tolerance(dist, (0, radii))
