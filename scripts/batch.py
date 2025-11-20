"""
Example usage of the TrainingRunner class.
This demonstrates how to replace the main() function logic with the new runner.
"""

import multiprocessing
import itertools

from cares_reinforcement_learning.util import helpers as hlp
from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser

# Set up logging first - choose your preferred preset
logs.LoggingPresets.production()  # or .production(), .quiet(), .debug()

# Get the main logger for this function
logger = logs.get_main_logger()


def batch_main_with_runner():
    """
    Simplified main function using TrainingRunner.
    This replaces the complex main() function in run.py
    """
    # Parse configurations (same as before)
    parser = RLParser()
    configurations = parser.parse_args()

    # Create the execution coordinator
    coordinator = ExecutionCoordinator(configurations)

    # Device validation
    device = hlp.get_device()
    logger.info(f"Device: {device}")

    # Warnings
    if device.type == "cpu":
        logger.warning(
            "Device being set as CPU - No cuda or mps detected. Training will be slower on cpu only."
        )
    if coordinator.env_config.save_train_checkpoints:
        logger.warning(
            "Training checkpoints will be saved - be aware this will increase disk usage (memory buffer)."
        )
        if coordinator.alg_config.image_observation:
            logger.warning(
                "Image observations are being used with checkpoints - this will take up a lot of disk space."
            )
    logger.info(f"Command: {coordinator.run_config.command}")
    logger.info(f"Data Path: {coordinator.run_config.data_path}")

    # MARK: BATCHING LOGIC
    config_templates: dict[str, list] = {
        "train_seeds": [[42], [43], [44]],
        "alg_config.actor_lr": [0.001, 0.0005],
        "alg_config.batch_size": [64, 128],
    }
    keys = list(config_templates.keys())
    configs = [dict(zip(keys, config_values)) for config_values in itertools.product(*config_templates.values())]

    for i, config in enumerate(configs):
        # Setup specific name and config
        run_name = get_name_from_config(config, i+1)
        replace_configurations(coordinator, config)
        logger.info(f"Starting run: {run_name} ({i+1}/{len(configs)})")

        # Deep copy and run coordinator
        coordinator.setup_logging_and_directories(run_name)
        coordinator.run()

def replace_configurations(coordinator: ExecutionCoordinator, config: dict):
    for key, value in config.items():
        keys = key.split('.')
        obj = coordinator
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)

def get_name_from_config(config: dict, index: int) -> str:
    name_parts = []
    for key, value in config.items():
        name_parts.append(f"{key.split('.')[-1]}-{value}")
    return f"[{index}]_" + "_".join(name_parts)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    batch_main_with_runner()
