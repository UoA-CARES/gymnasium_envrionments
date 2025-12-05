"""
Example usage of the TrainingRunner class.
This demonstrates how to replace the main() function logic with the new runner.
"""

import multiprocessing
import sys

from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser
from batch_coordinator import get_batch_coordinators
from cares_reinforcement_learning.util import helpers as hlp

# Set up logging first - choose your preferred preset
logs.LoggingPresets.production()  # or .production(), .quiet(), .debug()

# Get the main logger for this function
logger = logs.get_main_logger()


def main_with_runner():
    """
    Simplified main function using TrainingRunner.
    This replaces the complex main() function in run.py
    """
    # Parse configurations (same as before)
    parser = RLParser()
    configurations = parser.parse_args()
    is_batch = configurations.get("env_config").batch == 1  # type: ignore

    # Create the execution coordinator
    coordinator = ExecutionCoordinator(configurations)

    # Device validation
    device = hlp.get_device()
    logger.info(f"Device: {device}")

    # Interactive prompts
    run_name = input(
        f"Double check your experiment configurations :) Press ENTER to continue. {'' if is_batch else '(Optional - Enter a name for this run)'}\n"
    )

    if device.type == "cpu":
        no_gpu_answer = input(
            "Device being set as CPU - No cuda or mps detected. Do you want to continue? Note: Training will be slower on cpu only. [y/n]"
        )
        if no_gpu_answer not in ["y", "Y"]:
            logger.info(
                "Terminating Experiment - check CUDA or mps is installed correctly."
            )
            sys.exit()

    # Checkpoint warnings
    if coordinator.env_config.save_train_checkpoints:
        logger.warning(
            "Training checkpoints will be saved - be aware this will increase disk usage (memory buffer)."
        )
        if coordinator.alg_config.image_observation:
            no_gpu_answer = input(
                "Image observations are being used with checkpoints - this will take up a lot of disk space: Do you want to disable this? [y/n]"
            )
            if no_gpu_answer in ["y", "Y"]:
                logger.info("Disabling training checkpoint saving.")
                coordinator.env_config.save_train_checkpoints = False

    # Log command and data path
    logger.info(f"Command: {coordinator.run_config.command}")
    logger.info(f"Data Path: {coordinator.run_config.data_path}")

    if is_batch:
        batch_coordinators = get_batch_coordinators()

        # Support negative indexing (default: [b_start, b_end] = [0, -1])
        b_start = configurations.get("env_config").b_start  # type: ignore
        b_end = configurations.get("env_config").b_end  # type: ignore
        if b_start < 0:
            b_start = len(batch_coordinators) + b_start + 1
        if b_end < 0:
            b_end = len(batch_coordinators) + b_end + 1

        # User confirmation
        print("---------------------------------------------------")
        print("BATCH RUNS")
        print("---------------------------------------------------")
        for i, (batch_coordinator, batch_run_name) in enumerate(batch_coordinators):
            print(
                f"[{i+1}/{len(batch_coordinators)}] {batch_run_name}{' <- SKIPPED' if i+1 < b_start or i+1 > b_end else ''}"
            )
        batch_confirmation = input(
            f"Running batch of {len(batch_coordinators)} experiments. Do you want to continue? [y/n]\n"
        )
        if batch_confirmation not in ["y", "Y"]:
            logger.info("Terminating Batch Experiment as per user request.")
            sys.exit()

        # Execute batch runs
        for i, (batch_coordinator, batch_run_name) in enumerate(batch_coordinators):
            # Enable running only a range
            if i < b_start or i >= b_end:
                logger.info(
                    f"[{i+1}/{len(batch_coordinators)}] Skipping {batch_run_name}"
                )
                continue

            logger.info(f"[{i+1}/{len(batch_coordinators)}] Running {batch_run_name}")
            batch_coordinator.setup_logging_and_directories(batch_run_name)
            batch_coordinator.run()
        logger.info(f"Completed all {len(batch_coordinators)} batch experiments.")
    else:
        # Single Run
        logger.info("Running single experiment.")
        coordinator.setup_logging_and_directories(run_name)
        coordinator.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main_with_runner()
