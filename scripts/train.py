"""
This script is used to train reinforcement learning agents in DMCS/OpenAI/pyboy.
The main function parses command-line arguments, creates the environment, network, 
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
from datetime import datetime
from pathlib import Path
import yaml
import torch

import train_loops.policy_loop as pbe
import train_loops.ppo_loop as ppe
import train_loops.value_loop as vbe
from envrionments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig

from cares_reinforcement_learning.util import (
    MemoryFactory,
    NetworkFactory,
    Record,
    RLParser,
)
from cares_reinforcement_learning.util import helpers as hlp

logging.basicConfig(level=logging.INFO)


def main():
    """
    The main function that orchestrates the training process.
    """
    parser = RLParser(GymEnvironmentConfig)

    configurations = parser.parse_args()
    env_config = configurations["env_config"]
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

    iterations_folder = f"{alg_config.algorithm}-{env_config.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
    glob_log_dir = f"{Path.home()}/cares_rl_logs/{iterations_folder}"

    logging.info(
        "\n---------------------------------------------------\n"
        "ENVIRONMENT CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(env_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "ALGORITHM CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(alg_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "TRAINING CONFIG\n"
        "---------------------------------------------------"
    )
    
    logging.info(f"\n{yaml.dump(dict(training_config), default_flow_style=False)}")
    logging.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    answer = input("Enter y if you're happy with the experiement configurations: ")
    answer = answer.strip()

    if answer not in ["y", "Y"]:
        logging.info("Terminating Experiement :)")
        exit()

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            f"Training iteration {training_iteration+1}/{len(training_config.seeds)} with Seed: {seed}"
        )
        # This line should be here for seed consistency issues
        env = env_factory.create_environment(env_config)
        hlp.set_seed(seed)
        env.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )

        if agent is None:
            raise ValueError(
                f"Unkown agent for default algorithms {alg_config.algorithm}"
            )

        # TODO manage arguements for future memory types
        memory = memory_factory.create_memory(
            alg_config.memory, training_config.buffer_size, args=[]
        )
        logging.info(f"Memory: {alg_config.memory}")

        # create the record class - standardised results tracking
        log_dir = f"{seed}"
        record = Record(
            glob_log_dir=glob_log_dir,
            log_dir=log_dir,
            algorithm=alg_config.algorithm,
            task=env_config.task,
            network=agent,
            plot_frequency=training_config.plot_frequency,
            checkpoint_frequency=training_config.checkpoint_frequency,
        )

        record.save_config(env_config, "env_config")
        record.save_config(training_config, "train_config")
        record.save_config(alg_config, "alg_config")

        # Train the policy or value based approach
        if alg_config.algorithm == "PPO":
            ppe.ppo_train(env, agent, record, training_config, alg_config)
        elif agent.type == "policy":
            pbe.policy_based_train(
                env, agent, memory, record, training_config, alg_config
            )
        elif agent.type == "value":
            vbe.value_based_train(
                env, agent, memory, record, training_config, alg_config
            )
        else:
            raise ValueError(f"Agent type is unkown: {agent.type}")

        record.save()


if __name__ == "__main__":
    main()
