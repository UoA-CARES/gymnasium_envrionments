"""
This script is used to train reinforcement learning agents in DMCS/OpenAI/pyboy.
The main function parses command-line arguments, creates the environment, network, 
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
import os
import sys
from datetime import datetime

import train_loops.policy_loop as pbe
import train_loops.ppo_loop as ppe
import train_loops.value_loop as vbe
import yaml
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, Record, RLParser
from cares_reinforcement_learning.util import helpers as hlp
from environments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig

logging.basicConfig(level=logging.INFO)


def main():
    """
    The main function that orchestrates the training process.
    """
    parser = RLParser(GymEnvironmentConfig)

    configurations = parser.parse_args()
    env_config = configurations["env_config"]
    training_config = configurations["train_config"]
    alg_config = configurations["algorithm_config"]

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

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

    device = hlp.get_device()
    logging.info(f"Device: {device}")

    run_name = input(
        "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    )

    if device.type == "cpu":
        no_gpu_answer = input(
            "Device being set as CPU - No cuda or mps detected. Do you still want to continue? Note: Training will be slower on cpu only. [y/n]"
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info(
                "Terminating Experiment - check CUDA or mps is installed correctly."
            )
            sys.exit()

    log_path_template = os.environ.get(
        "CARES_LOG_PATH_TEMPLATE",
        "{algorithm}/{algorithm}-{domain_task}-{date}/{seed}",
    )

    date = datetime.now().strftime("%y_%m_%d_%H-%M-%S")

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            f"Training iteration {training_iteration+1}/{len(training_config.seeds)} with Seed: {seed}"
        )
        # This line should be here for seed consistency issues
        env = env_factory.create_environment(env_config, alg_config.image_observation)
        env_eval = env_factory.create_environment(
            env_config, alg_config.image_observation
        )
        hlp.set_seed(seed)
        env.set_seed(seed)
        env_eval.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )

        if agent is None:
            raise ValueError(
                f"Unknown agent for default algorithms {alg_config.algorithm}"
            )

        memory = memory_factory.create_memory(alg_config)

        log_dir = hlp.create_path_from_format_string(
            log_path_template,
            algorithm=alg_config.algorithm,
            domain=env_config.domain,
            task=env_config.task,
            gym=env_config.gym,
            seed=seed,
            run_name=run_name,
            date=date,
        )
        # create the record class - standardised results tracking
        record = Record(
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
            ppe.ppo_train(
                env,
                env_eval,
                agent,
                record,
                training_config,
                alg_config,
                display=env_config.display,
            )
        elif agent.type == "policy":
            pbe.policy_based_train(
                env,
                env_eval,
                agent,
                memory,
                record,
                training_config,
                alg_config,
                display=env_config.display,
                normalisation=True,
            )
        elif agent.type == "discrete_policy":
            pbe.policy_based_train(
                env,
                env_eval,
                agent,
                memory,
                record,
                training_config,
                alg_config,
                display=env_config.display,
                normalisation=False,
            )
        elif agent.type == "value":
            vbe.value_based_train(
                env,
                env_eval,
                agent,
                memory,
                record,
                training_config,
                alg_config,
                display=env_config.display,
            )
        else:
            raise ValueError(f"Agent type is unknown: {agent.type}")

        record.save()


if __name__ == "__main__":
    main()
