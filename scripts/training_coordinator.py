"""
TrainingCoordinator class for orchestrating reinforcement learning training across multiple seeds.
This class handles the parallel execution, configuration management, and coordination
of training runs for statistical validation.
"""

import concurrent.futures
import logging
import multiprocessing
import time
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

import training_logger as logs
import yaml
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from environments.gym_environment import GymEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
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
        Initialize the TrainingCoordinator with parsed configurations.

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
        self._print_configurations()

    def _validate_and_prepare_seeds(self) -> list[int]:
        """
        Validate configurations and prepare seed list for execution.
        """
        # Set seeds based on command type
        seeds = (
            self.run_config.seeds
            if self.run_config.command == "test"
            else self.training_config.seeds
        )

        logger.info(f"Running with seeds: {seeds}")
        return seeds

    def _print_configurations(self) -> None:
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

    def _evaluate(self) -> None:
        """Handle the evaluate command logic."""
        if not self.run_config.data_path:
            raise ValueError("Data path is required for evaluate command")

        # data_path = self.run_config.data_path

        # """Evaluate a specific seed (copied from original evaluate function)."""
        # model_path = Path(f"{data_path}/{seed}/models/")
        # folders = list(model_path.glob("*"))

        # # Sort folders and remove the final and best model folders
        # folders = natsorted(folders)[:-2]

    def _test(
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

        # data_path = self.run_config.data_path

        # algorithm_directory = Path(f"{data_path}/")
        # algorithm_data = list(algorithm_directory.glob("*"))

        # seed_folders = [entry for entry in algorithm_data if os.path.isdir(entry)]
        # seed_folders = natsorted(seed_folders)

        # for folder in seed_folders:
        #     model_path = Path(f"{folder}/models/final")

    def _listen_for_progress(self, queue, futures):
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

    def _train_single_seed(
        self,
        seed: int,
        progress_queue: Queue | None = None,
        save_configurations: bool = False,
    ) -> None:
        """
        Execute training/evaluation for a single seed.
        This now delegates all setup to TrainingRunner.

        Args:
            seed: Random seed for this run
            progress_queue: Queue for progress updates (if any)
            save_configurations: Whether to save configurations to disk
        """
        if self.base_log_dir is None:
            raise ValueError("Base log directory must be set before running seeds")

        # Create and run TrainingRunner - it handles all setup internally
        resume_path = None
        if self.run_config.command == "resume":
            resume_path = self.run_config.data_path

        runner = TrainingRunner(
            seed=seed,
            configurations=self.configurations,
            base_log_dir=self.base_log_dir,
            progress_queue=progress_queue,
            resume_path=resume_path,
            save_configurations=save_configurations,
        )

        runner.run_training()

    def _train_parallel_seeds(self) -> None:
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
                        self._train_single_seed,
                        seed=seed,
                        progress_queue=progress_queue,  # type: ignore[arg-type]
                        save_configurations=(
                            i == 0
                        ),  # Save configs only for first seed
                    )
                    for i, seed in enumerate(self.seeds)
                ]

                self._listen_for_progress(progress_queue, futures)

        logger.info(f"Completed all {len(self.seeds)} seeds")

    def _train_sequential_seeds(self) -> None:
        """
        Execute training/evaluation across multiple seeds sequentially.
        Useful for debugging or when parallel execution is not desired.
        """
        logs.set_logger_level("parallel", logging.INFO)
        logs.set_logger_level("seed", logging.INFO)

        for iteration, seed in enumerate(self.seeds):
            logger.info(
                f"Running seed {iteration+1}/{len(self.seeds)} with Seed: {seed}"
            )
            self._train_single_seed(
                seed=seed,
                save_configurations=(
                    iteration == 0
                ),  # Save configs only for first seed
            )

        logger.info(f"Completed all {len(self.seeds)} seeds sequentially")

    def _train(self) -> None:
        """
        Execute training/evaluation across multiple seeds.
        Chooses between parallel and sequential execution based on configuration.
        """
        start_time = time.time()
        if self.max_workers > 1 and len(self.seeds) > 1:
            self._train_parallel_seeds()
        else:
            self._train_sequential_seeds()

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Training completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

    def run(self) -> None:
        """
        Main entry point to run the training process.
        """
        if self.run_config.command in ["train", "resume"]:
            self._train()
        elif self.run_config.command == "evaluate":
            raise NotImplementedError(
                "Evaluate command is not yet implemented in TrainingCoordinator."
            )
        elif self.run_config.command == "test":
            raise NotImplementedError(
                "Test command is not yet implemented in TrainingCoordinator."
            )
