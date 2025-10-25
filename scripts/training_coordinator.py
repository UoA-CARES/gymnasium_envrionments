"""
TrainingRunner class for orchestrating reinforcement learning training across multiple seeds.
This class handles the parallel execution, configuration management, and coordination
of training runs for statistical validation.
"""

import concurrent.futures
import logging
import multiprocessing
import os
import time
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any

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
from environments.environment_factory import EnvironmentFactory
from environments.gym_environment import GymEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
from natsort import natsorted
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from training_runner import TrainingRunner
from util.configurations import GymEnvironmentConfig
from util.record import Record
from util.rl_parser import RunConfig

# Module-level loggers - created once when module loads
logger = logs.get_main_logger()
parallel_logger = logs.get_parallel_logger()


class TrainingCoordinator:
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

    def setup_logging_and_directories(self, run_name: str = "") -> None:
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
        progress_queue: Queue | None = None,
    ) -> None:
        """
        Execute training/evaluation for a single seed.
        This is the core logic extracted from the original run_seed_instance function.

        Args:
            iteration: Current iteration number
            seed: Random seed for this run
        """
        parallel_logger.info(f"[SEED {seed}] Starting iteration {iteration+1}")

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
            record_checkpoints=bool(self.env_config.save_train_checkpoints),
            checkpoint_interval=self.training_config.checkpoint_interval,
            logger=logs.get_seed_logger(),
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

        memory = memory_factory.create_memory(self.alg_config)

        # Set up record with agent and subdirectory
        record.set_agent(agent)
        record.set_sub_directory(f"{seed}")
        record.set_memory_buffer(memory)

        start_training_step = 0
        if self.run_config.command == "resume":
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

        # Execute based on command type
        if self.run_config.command == "train" or self.run_config.command == "resume":
            # Create and run the training loop using the new TrainingLoop class
            training_loop = TrainingRunner(
                env=env,
                env_eval=env_eval,
                agent=agent,
                memory=memory,
                record=record,
                train_config=self.training_config,
                alg_config=self.alg_config,
                seed=seed,
                progress_queue=progress_queue,
                logger=logs.get_seed_logger(),  # Use seed-specific logger
                display=bool(self.env_config.display),
                start_training_step=start_training_step,
            )

            training_loop.run_training()
        # elif self.run_config.command == "evaluate":
        #     self._handle_evaluate_command(env_eval, agent, record, seed)
        # elif self.run_config.command == "test":
        #     self._handle_test_command(env_eval, agent, record)
        else:
            raise ValueError(f"Unknown command {self.run_config.command}")

        record.save()

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
            self.alg_config,
            env_eval,
            agent,
            record,
        )

    def _evaluate_seed(
        self,
        data_path: str,
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
            alg_config,
            env,
            agent,
            record,
            folders,
        )

    def _test_models(
        self,
        data_path: str,
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
            self._run_evaluation_loop(alg_config, env, agent, record, [model_path])

    def _run_evaluation_loop(
        self,
        alg_config: AlgorithmConfig,
        env: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        record: Record,
        folders: list[Path],
    ) -> None:
        """Run evaluation loop for given model folders (using TrainingLoop for evaluation)."""
        for folder in folders:
            agent.load_models(folder, f"{alg_config.algorithm}")

            # Extract total steps from folder name
            try:
                total_steps = int(folder.name.split("_")[-1]) - 1
            except ValueError:
                total_steps = 0

            # Create a minimal TrainingLoop instance for evaluation only
            # We create dummy objects for required parameters that won't be used in evaluation
            memory = MemoryFactory().create_memory(alg_config)

            evaluator = TrainingRunner(
                env=env,  # Use evaluation environment for both env and env_eval
                env_eval=env,
                agent=agent,
                memory=memory,
                record=record,
                train_config=self.training_config,
                alg_config=alg_config,
                seed=0,  # Seed is not relevant for evaluation here
                logger=logs.get_main_logger(),
                display=False,
                start_training_step=0,
            )

            if agent.policy_type == "usd":
                evaluator._evaluate_usd(total_steps)
            else:
                # Update the evaluator to not normalize actions
                evaluator._evaluate_agent(total_steps)

    def listen_for_progress(self, queue, futures):
        progress = Progress(
            TextColumn("[bold blue]{task.fields[seed]}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[status]}"),
            TextColumn("[cyan]{task.completed}/{task.total}"),  # <-- added this
            TimeElapsedColumn(),
        )

        tasks: dict[int, int] = {}
        done_seeds: set[int] = set()

        with progress:
            while True:
                try:
                    msg = queue.get_nowait()
                except Empty:
                    msg = None

                if msg:
                    seed = msg["seed"]

                    if seed not in tasks:
                        total = msg.get("total", 1)
                        tasks[seed] = progress.add_task(
                            f"Seed {seed}",
                            total=total,
                            seed=f"Seed {seed}",
                            status=msg.get("status", ""),
                        )

                    progress.update(
                        tasks[seed],
                        completed=msg.get("step", 0),
                        status=msg.get("status", ""),
                    )

                    if msg.get("status") == "done":
                        done_seeds.add(seed)
                        progress.console.log(f"[green]Seed {seed} completed!")

                if len(done_seeds) == len(futures):
                    break

                # time.sleep(0.1)

    def run_parallel_seeds(self) -> None:
        """
        Execute training/evaluation across multiple seeds in parallel.

        Args:
            max_workers: Maximum number of parallel workers (default 1 for testing)
        """
        logger.info(f"Running with {self.max_workers} parallel workers")

        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()

            # Use ProcessPoolExecutor with limited workers
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self.run_single_seed,
                        iteration=i,
                        seed=seed,
                        progress_queue=progress_queue,  # type: ignore[arg-type]
                    )
                    for i, seed in enumerate(self.seeds)
                ]

                self.listen_for_progress(progress_queue, futures)

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
        start_time = time.time()
        if self.max_workers > 1 and len(self.seeds) > 1:
            self.run_parallel_seeds()
        else:
            logs.set_logger_level("parallel", logging.INFO)
            logs.set_logger_level("seed", logging.INFO)
            self.run_sequential_seeds()

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Training completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )
