"""
Description:
Created: David
"""
import cv2
import torch
import numpy as np

from dm_control import suite
from dm_control import viewer



import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

from cares_reinforcement_learning.algorithm import TD3
from cares_reinforcement_learning.networks.TD3 import Actor
from cares_reinforcement_learning.networks.TD3 import Critic
from cares_reinforcement_learning.util import MemoryBuffer



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# testing Environments

env = suite.load(domain_name="ball_in_cup", task_name="catch")
#env = suite.load(domain_name="cartpole", task_name="swingup")  # swingup, balance
#env = suite.load(domain_name="cheetah", task_name="run")
#env = suite.load(domain_name="walker", task_name="walk") # stand, walk, run
#env = suite.load(domain_name="finger", task_name="spin")
#env = suite.load(domain_name="humanoid", task_name="run")

# suite.BENCHMARKING  # (domain, task)  # for interact with several task and envs


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



def evaluation(agent, observation_type):

    video_duration = 20 # seg

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    frames = []

    time_step = env.reset()
    state     = time_step.observation[observation_type]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        time_step = env.step(action)
        state, reward, done = time_step.observation[observation_type], time_step.reward, time_step.last()
        episode_reward += reward
        camera0 = env.physics.render(camera_id=0, height=200, width=200)
        camera1 = env.physics.render(camera_id=1, height=200, width=200)
        frames.append(np.hstack((camera0, camera1)))

        if done:
            print(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            time_step = env.reset()
            state = time_step.observation[observation_type]
            episode_timesteps = 0
            episode_reward = 0
            episode_num = 0

    fps   = len(frames)/video_duration
    video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (400, 200))

    for frame_count in range(len(frames)):
        video.write(frames[frame_count])
    video.release()



def train(agent, memory, max_action_value, min_action_value, action_num, observation_type):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    time_step = env.reset()
    state     = time_step.observation[observation_type]

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):

        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            print(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(min_action_value, max_action_value, size=action_num)

        else:
            action = agent.select_action_from_policy(state)

        time_step = env.step(action)

        next_state, reward, done = time_step.observation[observation_type], time_step.reward, time_step.last()
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
            state     = time_step.observation[observation_type]
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1


def main():

    time_step   = env.reset()
    action_spec = env.action_spec()

    observation_type = list(time_step.observation.keys())[0]  # position, orientation, joint_angles
    print(f"Working With {observation_type} as observation")

    observation_size = time_step.observation[observation_type].shape[0]
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

    #train(agent, memory, max_action_value, min_action_value, action_num, observation_type)
    evaluation(agent, observation_type)





if __name__ == '__main__':
    main()