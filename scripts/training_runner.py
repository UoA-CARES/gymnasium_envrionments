"""
TrainingRunner class for orchestrating reinforcement learning training across multiple seeds.
This class handles the parallel execution, configuration management, and coordination
of training runs for statistical validation.
"""

import concurrent.futures
import logging
import os
from functools import partial
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import train_loop as tl
import training_logger as logs
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
from util.rl_parser import RunConfig

# Module-level loggers - created once when module loads
logger = logs.get_main_logger()
parallel_logger = logs.get_parallel_logger()


class TrainingRunner:
    """
    Orchestrates reinforcement learning training across multiple seeds with support
    for parallel execution, configuration management, and different run modes.
    """

    def __init__(self, configurations: dict[str, Any]):
        """
        Initialize the TrainingRunner with parsed configurations.

        Args:
            configurations: Dictionary containing all parsed configurations
        """
        self.configurations = configurations
        self.run_config: RunConfig = configurations["run_config"]
        self.env_config: GymEnvironmentConfig = configurations["env_config"]
        self.training_config: TrainingConfig = configurations["train_config"]
        self.alg_config: AlgorithmConfig = configurations["alg_config"]

        self.seeds: list[int] = self._validate_and_prepare_seeds()

        self.max_workers: int = self.training_config.max_workers
        self.max_workers = min(len(self.seeds), self.max_workers)
        self.max_workers = max(1, self.max_workers)

        self.base_log_dir: str | None = None

        # Log all configurations for debugging
        self.log_configurations()

    def setup_logging_and_directories(self, run_name: str = "") -> str:
        """
        Set up logging directories and validate configurations.

        Args:
            run_name: Optional name for the training run

        Returns:
            Base log directory path
        """
        self.base_log_dir = Record.create_base_directory(
            domain=self.env_config.domain,
            task=self.env_config.task,
            gym=self.env_config.gym,
            algorithm=self.alg_config.algorithm,
            run_name=run_name,
        )

        logger.info(f"Base Log Directory: {self.base_log_dir}")
        return self.base_log_dir or ""

    def _validate_and_prepare_seeds(self) -> list[int]:
        """
        Validate configurations and prepare seed list for execution.
        """
        # Handle resume command seed filtering
        if self.run_config.command == "resume":
            if self.run_config.seed in self.training_config.seeds:
                idx = self.training_config.seeds.index(self.run_config.seed)
                self.training_config.seeds = self.training_config.seeds[idx:]

        # Set seeds based on command type
        seeds = (
            self.run_config.seeds
            if self.run_config.command == "test"
            else self.training_config.seeds
        )

        logger.info(f"Running with seeds: {seeds}")
        return seeds

    def log_configurations(self) -> None:
        """Log all configurations for debugging and reproducibility."""
        logger.info(
            "\n---------------------------------------------------\n"
            "RUN CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.run_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "ENVIRONMENT CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.env_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "ALGORITHM CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(f"\n{yaml.dump(self.alg_config.dict(), default_flow_style=False)}")

        logger.info(
            "\n---------------------------------------------------\n"
            "TRAINING CONFIG\n"
            "---------------------------------------------------"
        )
        logger.info(
            f"\n{yaml.dump(self.training_config.dict(), default_flow_style=False)}"
        )

    def run_single_seed(
        self,
        iteration: int,
        seed: int,
    ) -> None:
        """
        Execute training/evaluation for a single seed.
        This is the core logic extracted from the original run_seed_instance function.

        Args:
            iteration: Current iteration number
            seed: Random seed for this run
        """
        parallel_logger.info(f"Iteration {iteration+1} with Seed: {seed}")

        # Create factory instances (each process needs its own)
        env_factory = EnvironmentFactory()
        network_factory = NetworkFactory()
        memory_factory = MemoryFactory()

        # Create record for this seed
        record = Record(
            base_directory=f"{self.base_log_dir}",
            algorithm=self.alg_config.algorithm,
            task=self.env_config.task,
            agent=None,
            record_video=self.training_config.record_eval_video,
            record_checkpoints=self.env_config.save_train_checkpoints,
            checkpoint_interval=self.training_config.checkpoint_interval,
        )

        # Save configurations only on first iteration to avoid conflicts
        if iteration == 0:
            record.save_configurations(self.configurations)

        # Create the Environment
        parallel_logger.info(f"Loading Environment: {self.env_config.gym}")
        env, env_eval = env_factory.create_environment(
            self.env_config, self.alg_config.image_observation
        )

        # Set the seed for everything
        hlp.set_seed(seed)
        env.set_seed(seed)
        env_eval.set_seed(seed)

        # Create the algorithm
        parallel_logger.info(f"Algorithm: {self.alg_config.algorithm}")
        agent: Algorithm = network_factory.create_network(
            env.observation_space, env.action_num, self.alg_config
        )

        # Validate agent creation
        if agent is None:
            raise ValueError(
                f"Unknown agent for default algorithms {self.alg_config.algorithm}"
            )

        # Set up record with agent and subdirectory
        record.set_agent(agent)
        record.set_sub_directory(f"{seed}")

        # Execute based on command type
        if self.run_config.command == "train":
            self._handle_train_command(
                env, env_eval, agent, memory_factory, record, seed
            )
        elif self.run_config.command == "resume":
            self._handle_resume_command(env, env_eval, agent, record, seed)
        elif self.run_config.command == "evaluate":
            self._handle_evaluate_command(env_eval, agent, record, seed)
        elif self.run_config.command == "test":
            self._handle_test_command(env_eval, agent, record)
        else:
            raise ValueError(f"Unknown command {self.run_config.command}")

        record.save()

    def _handle_train_command(
        self,
        env: GymEnvironment | MultiModalWrapper,
        env_eval: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        memory_factory: MemoryFactory,
        record: Record,
        seed: int,
    ) -> None:
        """Handle the train command logic."""
        memory = memory_factory.create_memory(self.alg_config)
        record.set_memory_buffer(memory)

        manager = Manager()
        progress_dict = manager.dict()

        tl.train_agent(
            env,
            env_eval,
            agent,
            memory,
            record,
            self.training_config,
            self.alg_config,
            progress_dict,
            seed,
            display=bool(self.env_config.display),
        )

    def _handle_resume_command(
        self,
        env: GymEnvironment | MultiModalWrapper,
        env_eval: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
        seed: int,
    ) -> None:
        """Handle the resume command logic."""
        if not self.run_config.data_path:
            raise ValueError("Data path is required for resume command")
        restart_path = Path(self.run_config.data_path) / str(seed)
        parallel_logger.info(f"Restarting from path: {restart_path}")

        parallel_logger.info("Loading training and evaluation data")
        record.load(restart_path)

        parallel_logger.info("Loading memory buffer")
        memory = MemoryBuffer.load(restart_path / "memory", "memory")
        record.set_memory_buffer(memory)

        parallel_logger.info("Loading agent models")
        agent.load_models(
            restart_path / "models" / "checkpoint", f"{self.alg_config.algorithm}"
        )

        start_training_step = record.get_last_logged_step()

        manager = Manager()
        progress_dict = manager.dict()

        tl.train_agent(
            env,
            env_eval,
            agent,
            memory,
            record,
            self.training_config,
            self.alg_config,
            progress_dict,
            seed,
            display=bool(self.env_config.display),
            start_training_step=start_training_step,
        )

    def _handle_evaluate_command(
        self,
        env_eval: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
        seed: int,
    ) -> None:
        """Handle the evaluate command logic."""
        if not self.run_config.data_path:
            raise ValueError("Data path is required for evaluate command")
        self._evaluate_seed(
            self.run_config.data_path,
            self.training_config.number_eval_episodes,
            seed,
            self.alg_config,
            env_eval,
            agent,
            record,
        )

    def _handle_test_command(
        self,
        env_eval: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
    ) -> None:
        """Handle the test command logic."""
        if not self.run_config.data_path:
            raise ValueError("Data path is required for test command")
        if not self.run_config.episodes:
            raise ValueError("Episodes count is required for test command")
        self._test_models(
            self.run_config.data_path,
            self.run_config.episodes,
            self.alg_config,
            env_eval,
            agent,
            record,
        )

    def _evaluate_seed(
        self,
        data_path: str,
        number_eval_episodes: int,
        seed: int,
        alg_config: AlgorithmConfig,
        env: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
    ) -> None:
        """Evaluate a specific seed (copied from original evaluate function)."""
        model_path = Path(f"{data_path}/{seed}/models/")
        folders = list(model_path.glob("*"))

        # Sort folders and remove the final and best model folders
        folders = natsorted(folders)[:-2]

        self._run_evaluation_loop(
            number_eval_episodes,
            alg_config,
            env,
            agent,
            record,
            folders,
        )

    def _test_models(
        self,
        data_path: str,
        number_eval_episodes: int,
        alg_config: AlgorithmConfig,
        env: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
    ) -> None:
        """Test models across all seeds (copied from original test function)."""
        algorithm_directory = Path(f"{data_path}/")
        algorithm_data = list(algorithm_directory.glob("*"))

        seed_folders = [entry for entry in algorithm_data if os.path.isdir(entry)]
        seed_folders = natsorted(seed_folders)

        for folder in seed_folders:
            model_path = Path(f"{folder}/models/final")
            self._run_evaluation_loop(
                number_eval_episodes, alg_config, env, agent, record, [model_path]
            )

    def _run_evaluation_loop(
        self,
        number_eval_episodes: int,
        alg_config: AlgorithmConfig,
        env: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
        folders: list[Path],
    ) -> None:
        """Run evaluation loop for given model folders (copied from original)."""
        for folder in folders:
            agent.load_models(folder, f"{alg_config.algorithm}")

            # Extract total steps from folder name
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
                )
            elif agent.policy_type == "usd":
                tl.evaluate_usd(
                    env,
                    agent,
                    record=record,
                    total_steps=total_steps,
                    normalisation=True,
                )
            elif agent.policy_type == "discrete_policy":
                tl.evaluate_agent(
                    env,
                    agent,
                    number_eval_episodes,
                    record=record,
                    total_steps=total_steps,
                    normalisation=False,
                )
            elif agent.policy_type == "value":
                tl.evaluate_agent(
                    env,
                    agent,
                    number_eval_episodes,
                    record=record,
                    total_steps=total_steps,
                    normalisation=False,
                )
            else:
                raise ValueError(f"Agent type is unknown: {agent.policy_type}")

    def run_parallel_seeds(self) -> None:
        """
        Execute training/evaluation across multiple seeds in parallel.

        Args:
            max_workers: Maximum number of parallel workers (default 1 for testing)
        """
        logger.info(f"Running with {self.max_workers} parallel workers")

        # Split the evaluation and training loop setup
        run_task_partial = partial(
            self.run_single_seed,
        )

        # Use ProcessPoolExecutor with limited workers
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(
                    run_task_partial,
                    iteration=i,
                    seed=seed,
                )
                for i, seed in enumerate(self.seeds)
            ]

            # Wait for all futures to complete and handle any exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                    logger.info("Seed completed successfully")
                except Exception as e:
                    logger.error(f"Error in seed execution: {e}")
                    raise

        logger.info(f"Completed all {len(self.seeds)} seeds")

    def run_sequential_seeds(self) -> None:
        """
        Execute training/evaluation across multiple seeds sequentially.
        Useful for debugging or when parallel execution is not desired.
        """
        for iteration, seed in enumerate(self.seeds):
            logger.info(
                f"Running iteration {iteration+1}/{len(self.seeds)} with Seed: {seed}"
            )
            self.run_single_seed(iteration, seed)

        logger.info(f"Completed all {len(self.seeds)} seeds sequentially")

    def run(self) -> None:
        """
        Main entry point to run the training process.

        Args:
            max_workers: Maximum number of parallel workers
            use_parallel: Whether to use parallel execution
        """
        if self.max_workers > 1 and len(self.seeds) > 1:
            self.run_parallel_seeds()
        else:
            logs.set_logger_level("parallel", logging.INFO)
            self.run_sequential_seeds()
