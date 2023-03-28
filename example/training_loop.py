"""
Description:
Created: David
"""

import torch
import logging
from dm_control import suite

from cares_reinforcement_learning.algorithm import TD3
from cares_reinforcement_learning.networks.TD3 import Actor
from cares_reinforcement_learning.networks.TD3 import Critic

from cares_reinforcement_learning.util import MemoryBuffer


logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#env = suite.load(domain_name="acrobot", task_name="swingup") # this load one task
env = suite.load(domain_name="cartpole", task_name="swingup") # this load one task
# suite.BENCHMARKING  # (domain, task)  # for interact with several task and envs

action = env.action_spec()
time_step = env.reset()

print(time_step)
print(time_step.last())

def main():
    pass




if __name__ == '__main__':
    main()