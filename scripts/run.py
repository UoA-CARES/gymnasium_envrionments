"""
This script is used to train reinforcement learning agents in DMCS/OpenAI/pyboy.
The main function parses command-line arguments, creates the environment, network,
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
import os
import sys
from pathlib import Path

import train_loop as tl
import yaml
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from cares_reinforcement_learning.util.record import Record
from environments.environment_factory import EnvironmentFactory
from environments.gym_environment import GymEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
from natsort import natsorted
from util.configurations import GymEnvironmentConfig
from util.rl_parser import RLParser, RunConfig

logging.basicConfig(level=logging.INFO)


def run_evaluation_loop(
    number_eval_episodes: int,
    alg_config: AlgorithmConfig,
    env: GymEnvironment | MultiModalWrapper,
    agent: Algorithm,
    record: Record,
    folders: list[Path],
):
    for folder in folders:
        agent.load_models(folder, f"{alg_config.algorithm}")

        # ewww this is bad - fix this at some point
        try:
            total_steps = int(folder.name.split("_")[-1]) - 1
        except ValueError:
            total_steps = 0

        if agent.policy_type == "policy":
            tl.evaluate_agent(
                env,
                agent,
                number_eval_episodes,
                record=record,
                total_steps=total_steps,
                normalisation=True,
                # display=env_config.display,
            )
        elif agent.policy_type == "usd":
            tl.evaluate_usd(
                env,
                agent,
                record=record,
                total_steps=total_steps,
                normalisation=True,
                # display=env_config.display,
            )
        elif agent.policy_type == "discrete_policy":
            tl.evaluate_agent(
                env,
                agent,
                number_eval_episodes,
                record=record,
                total_steps=total_steps,
                normalisation=False,
                # display=env_config.display,
            )
        elif agent.policy_type == "value":
            tl.evaluate_agent(
                env,
                agent,
                number_eval_episodes,
                record=record,
                total_steps=total_steps,
                normalisation=False,
                # display=env_config.display,
            )
        else:
            raise ValueError(f"Agent type is unknown: {agent.policy_type}")


def test(
    data_path: str,
    number_eval_episodes: int,
    alg_config: AlgorithmConfig,
    env: GymEnvironment | MultiModalWrapper,
    agent: Algorithm,
    record: Record,
):
    # Model Path is the seeds directory - remove files
    algorithm_directory = Path(f"{data_path}/")
    algorithm_data = list(algorithm_directory.glob("*"))

    seed_folders = [entry for entry in algorithm_data if os.path.isdir(entry)]

    seed_folders = natsorted(seed_folders)

    for folder in seed_folders:
        model_path = Path(f"{folder}/models/final")
        run_evaluation_loop(
            number_eval_episodes, alg_config, env, agent, record, [model_path]
        )


def evaluate(
    data_path: str,
    number_eval_episodes: int,
    seed: int,
    alg_config,
    env: GymEnvironment | MultiModalWrapper,
    agent: Algorithm,
    record: Record,
):

    model_path = Path(f"{data_path}/{seed}/models/")
    folders = list(model_path.glob("*"))

    # sort folders and remove the final and best model folders
    folders = natsorted(folders)[:-2]

    run_evaluation_loop(
        number_eval_episodes,
        alg_config,
        env,
        agent,
        record,
        folders,
    )


def train(
    env_config,
    training_config,
    alg_config,
    env,
    env_eval,
    agent: Algorithm,
    memory,
    record,
    start_training_step: int = 0,
):
    # TODO you can collapse this...apply normalisation in the agent class is the only difference
    if agent.policy_type == "policy" or agent.policy_type == "usd":
        tl.train_agent(
            env,
            env_eval,
            agent,
            memory,
            record,
            training_config,
            alg_config,
            display=env_config.display,
            apply_action_normalisation=True,
            start_training_step=start_training_step,
        )
    elif agent.policy_type == "discrete_policy" or agent.policy_type == "value":
        tl.train_agent(
            env,
            env_eval,
            agent,
            memory,
            record,
            training_config,
            alg_config,
            display=env_config.display,
            apply_action_normalisation=False,
            start_training_step=start_training_step,
        )
    else:
        raise ValueError(f"Agent type is unknown: {agent.policy_type}")


def main():
    """
    The main function that orchestrates the training process.
    """
    parser = RLParser()

    configurations = parser.parse_args()
    run_config: RunConfig = configurations["run_config"]  # type: ignore
    env_config: GymEnvironmentConfig = configurations["env_config"]  # type: ignore
    training_config: TrainingConfig = configurations["train_config"]  # type: ignore
    alg_config: AlgorithmConfig = configurations["alg_config"]  # type: ignore

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

    logging.info(
        "\n---------------------------------------------------\n"
        "RUN CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(run_config.dict(), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "ENVIRONMENT CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(env_config.dict(), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "ALGORITHM CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(alg_config.dict(), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "TRAINING CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(training_config.dict(), default_flow_style=False)}")

    device = hlp.get_device()
    logging.info(f"Device: {device}")

    run_name = input(
        "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    )

    if device.type == "cpu":
        no_gpu_answer = input(
            "Device being set as CPU - No cuda or mps detected. Do you want to continue? Note: Training will be slower on cpu only. [y/n]"
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info(
                "Terminating Experiment - check CUDA or mps is installed correctly."
            )
            sys.exit()

    if env_config.save_train_checkpoints:
        logging.warning(
            "Training checkpoints will be saved - be aware this will increase disk usage (memory buffer)."
        )
        if alg_config.image_observation:
            no_gpu_answer = input(
                "Image observations are being used with checkpoints - this will take up a lot of disk space: Do you want to disable this? [y/n]"
            )

            if no_gpu_answer in ["y", "Y"]:
                logging.info("Disabling training checkpoint saving.")
                env_config.save_train_checkpoints = False

    logging.info(f"Command: {run_config.command}")

    logging.info(f"Data Path: {run_config.data_path}")

    base_log_dir = Record.create_base_directory(
        domain=env_config.domain,
        task=env_config.task,
        gym=env_config.gym,
        algorithm=alg_config.algorithm,
        run_name=run_name,
    )

    logging.info(f"Base Log Directory: {base_log_dir}")

    record = Record(
        base_directory=f"{base_log_dir}",
        algorithm=alg_config.algorithm,
        task=env_config.task,
        agent=None,
        record_video=training_config.record_eval_video,
        record_checkpoints=env_config.save_train_checkpoints,
        checkpoint_interval=training_config.checkpoint_interval,
    )

    record.save_configurations(configurations)

    # # update to have only the remaining seeds in the list for restarting training
    if run_config.command == "resume":
        if run_config.seed in training_config.seeds:
            idx = training_config.seeds.index(run_config.seed)
            training_config.seeds = training_config.seeds[idx:]

    seeds = run_config.seeds if run_config.command == "test" else training_config.seeds

    # Split the evaluation and training loop setup
    for iteration, seed in enumerate(seeds):
        logging.info(f"Iteration {iteration+1}/{len(seeds)} with Seed: {seed}")

        # Create the Environment
        # This line should be here for seed consistency issues
        logging.info(f"Loading Environment: {env_config.gym}")
        env, env_eval = env_factory.create_environment(
            env_config, alg_config.image_observation
        )

        # Set the seed for everything
        hlp.set_seed(seed)
        env.set_seed(seed)
        env_eval.set_seed(seed)

        # Create the algorithm
        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent: Algorithm = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )

        # legacy handler for other gyms - expcetion should in factory
        if agent is None:
            raise ValueError(
                f"Unknown agent for default algorithms {alg_config.algorithm}"
            )

        # create the record class - standardised results tracking
        record.set_agent(agent)
        record.set_sub_directory(f"{seed}")

        if run_config.command == "train":
            # Train the policy or value based approach

            # Create the memory - only required for training
            memory = memory_factory.create_memory(alg_config)
            record.set_memory_buffer(memory)

            train(
                env_config,
                training_config,
                alg_config,
                env,
                env_eval,
                agent,
                memory,
                record,
            )
        elif run_config.command == "resume":
            # Steps to restart training
            # Set the seeds for training - tick
            # Load the agent - tick
            # Load the memory buffer - tick
            # Load the training and evaluation data into Record - tick
            # Set the steps to the correct number to restart training from - tick
            # Set the episode to the correct number to restart training from

            restart_path = Path(run_config.data_path) / str(seed)
            logging.info(f"Restarting from path: {restart_path}")

            logging.info("Loading training and evaluation data")
            record.load(restart_path)

            logging.info("Loading memory buffer")
            memory = MemoryBuffer.load(restart_path / "memory", "memory")
            record.set_memory_buffer(memory)

            logging.info("Loading agent models")
            agent.load_models(
                restart_path / "models" / "checkpoint", f"{alg_config.algorithm}"
            )

            start_training_step = record.get_last_logged_step()

            train(
                env_config,
                training_config,
                alg_config,
                env,
                env_eval,
                agent,
                memory,
                record,
                start_training_step,
            )

        elif run_config.command == "evaluate":
            # Evaluate the policy or value based approach
            evaluate(
                run_config.data_path,
                training_config.number_eval_episodes,
                seed,
                alg_config,
                env_eval,
                agent,
                record,
            )
        elif run_config.command == "test":
            test(
                run_config.data_path,
                run_config.episodes,
                alg_config,
                env_eval,
                agent,
                record,
            )
        else:
            raise ValueError(f"Unknown command {run_config.command}")

        record.save()


if __name__ == "__main__":
    main()
