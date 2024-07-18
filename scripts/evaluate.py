"""
This script is used to load and evaluate reinforcement learning agents in DMCS/OpenAI/pyboy.
The main function parses command-line arguments, creates the environment, network, 
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from envrionments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, Record, RLParser
from cares_reinforcement_learning.util import helpers as hlp

from train_loops.policy_loop import evaluate_policy_network

logging.basicConfig(level=logging.INFO)


def main():
    """
    The main function that orchestrates the evaluation process.
    """
    parser = RLParser(GymEnvironmentConfig)

    configurations = parser.parse_args()
    env_config = configurations["env_config"]
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()

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

    input("Double check your experiement configurations :) Press ENTER to continue.")

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            f"Iteration {training_iteration+1}/{len(training_config.seeds)} with Seed: {seed}"
        )
        # This line should be here for seed consistency issues
        env = env_factory.create_environment(env_config, alg_config.image_observation)
        hlp.set_seed(seed)
        env.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )

        file_path = (
            "/home/pokemon/cares_rl_logs/SACAE/SACAE-pokemon-fight-24_07_03_12:35:07/10"
        )
        model_name = "SACAE-checkpoint-80"
        agent.load_models(file_path, model_name)

        iterations_folder = f"{alg_config.algorithm}/{alg_config.algorithm}-{env_config.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
        glob_log_dir = f"{Path.home()}/cares_rl_logs/{iterations_folder}"

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

        evaluate_policy_network(env, agent=agent, config=training_config, record=record)


if __name__ == "__main__":
    main()
