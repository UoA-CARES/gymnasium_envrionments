"""
Description: Observation Input: Vector
Created: David
Date: March 2023
"""

import cv2
import torch
import numpy as np
from dm_control import suite

import pandas as pd
import matplotlib.pyplot as plt

from cares_reinforcement_learning.algorithm import TD3
from cares_reinforcement_learning.networks.TD3 import Actor
from cares_reinforcement_learning.networks.TD3 import Critic
from cares_reinforcement_learning.util import MemoryBuffer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# testing Environments
# suite.BENCHMARKING  # (domain, task)  # for interact with several task and envs
#env = suite.load(domain_name="ball_in_cup", task_name="catch")
#env = suite.load(domain_name="cartpole", task_name="balance")  # swingup, balance
#env = suite.load(domain_name="cheetah", task_name="run")
#env = suite.load(domain_name="walker", task_name="walk") # stand, walk, run
#env = suite.load(domain_name="finger", task_name="spin")
#env = suite.load(domain_name="humanoid", task_name="run")

G          = 10
GAMMA      = 0.99
TAU        = 0.005
ACTOR_LR   = 1e-4
CRITIC_LR  = 1e-3
BATCH_SIZE = 32

max_steps_exploration = 10_000
max_steps_training    = 100_000
max_steps_evaluation  = 5_000
SEED                  = 571


DOMAIN_NAME = "cartpole"
TASK_NAME   = "balance"
env = suite.load(DOMAIN_NAME, TASK_NAME, task_kwargs={'random': SEED})


def grab_frame():
    camera0 = env.physics.render(camera_id=0, height=480, width=600)
    camera1 = env.physics.render(camera_id=1, height=480, width=600)
    rgbArr  = np.hstack((camera0, camera1))
    frame   = cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame

def plot_reward_curve(data_reward):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.show()

def evaluation(agent):
    frame = grab_frame()
    fps   = 60
    video_name = f'Result_{DOMAIN_NAME}_{TASK_NAME}.mp4'
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    time_step = env.reset()
    state     = np.hstack(list(time_step.observation.values()))

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        time_step = env.step(action)
        state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
        episode_reward += reward

        video.write(grab_frame())

        if done:
            print(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            time_step = env.reset()
            state     = np.hstack(list(time_step.observation.values()))
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1

    video.release()


def train(agent, memory, max_action_value, min_action_value, action_num):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    time_step = env.reset()
    state     = np.hstack(list(time_step.observation.values()))

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            print(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_num)
        else:
            action = agent.select_action_from_policy(state)

        time_step = env.step(action)
        next_state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
        memory.add(state, action, reward, next_state, done)
        state = next_state

        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(BATCH_SIZE)
                agent.train_policy(experiences)

        if done:
            print(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            time_step = env.reset()
            state     = np.hstack(list(time_step.observation.values()))
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    plot_reward_curve(historical_reward)


def main():
    time_step   = env.reset()
    action_spec = env.action_spec()

    observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
    print("Example of Observation Space of this Environment:", observation)

    observation_size = len(observation)
    action_num       = action_spec.shape[0]

    max_action_value = action_spec.maximum[0]
    min_action_value = action_spec.minimum[0]

    memory = MemoryBuffer()
    actor  = Actor(observation_size, action_num, ACTOR_LR)
    critic = Critic(observation_size, action_num, CRITIC_LR)

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=GAMMA,
        tau=TAU,
        action_num=action_num,
        device=DEVICE,
    )

    train(agent, memory, max_action_value, min_action_value, action_num)
    print("_____________Running Evaluation ____________________________")
    evaluation(agent)


if __name__ == '__main__':
    main()
