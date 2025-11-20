"""
This module enables batch execution of multiple reinforcement learning experiments
with varying configurations. It constructs ExecutionCoordinator instances for each
configuration combination and prepares them for execution.
"""

import itertools

from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser

# MARK: BATCH CONFIG
# Configure batch parameters here. The cross-product of these lists will be used
# to create multiple experiment configurations.
batch_config: dict[str, list] = {
    "train_seeds": [[42], [43], [44]],
    "alg_config.actor_lr": [0.001, 0.0005],
    "alg_config.batch_size": [64, 128],
}

# Get the main logger for this function
logger = logs.get_main_logger()

def get_batch_coordinators() -> list[tuple[ExecutionCoordinator, str]]:
    """Create coordinators for every combination in batch_config.

    The batch_config maps attribute paths (possibly dotted, e.g. "alg_config.actor_lr")
    to lists of values. This function expands the Cartesian product of those lists,
    creates an ExecutionCoordinator for each combination, applies the configuration
    values to the coordinator, and returns a list of (coordinator, run_name) tuples.

    Returns:
        list[tuple[ExecutionCoordinator, str]]: A list where each tuple contains a
            configured ExecutionCoordinator and a human-readable run name generated
            by get_name_from_config.

    Notes:
        - This function only constructs and names coordinators; it does not start or run them.
        - Keys in batch_config are resolved via attribute access on the coordinator.
    """
    # Expand batch configs into all combinations
    keys = list(batch_config.keys())
    configs = [dict(zip(keys, config_values)) for config_values in itertools.product(*batch_config.values())]

    coordinators: list[tuple[ExecutionCoordinator, str]] = []
    for i, config in enumerate(configs):
        # Setup specific name and coordinator
        run_name = get_name_from_config(config, i+1)
        coordinator = config_to_coordinator(config)
        replace_configurations(coordinator, config)
        coordinators.append((coordinator, run_name))

    return coordinators

def get_name_from_config(config: dict, index: int) -> str:
    name_parts = []
    for key, value in config.items():
        name_parts.append(f"{key.split('.')[-1]}-{value}")
    return f"[{index}]_" + "_".join(name_parts)

def config_to_coordinator(config: dict) -> ExecutionCoordinator:
    parser = RLParser()
    base_configs = parser.parse_args()
    coordinator = ExecutionCoordinator(base_configs)
    replace_configurations(coordinator, config)
    return coordinator

def replace_configurations(coordinator: ExecutionCoordinator, config: dict):
    for key, value in config.items():
        keys = key.split('.')
        obj = coordinator
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)
