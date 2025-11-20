"""
This module enables batch execution of multiple reinforcement learning experiments
with varying configurations. It constructs ExecutionCoordinator instances for each
configuration combination and prepares them for execution.
"""

import itertools
from typing import Any

from cares_reinforcement_learning.util.configurations import FunctionLayer, MLPConfig, TrainableLayer

from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser

relu: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=64),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=64, out_features=64),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=64),
        ]
    )

gelu: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=64),
            FunctionLayer(layer_type="GELU"),
            TrainableLayer(layer_type="Linear", in_features=64, out_features=64),
            FunctionLayer(layer_type="GELU"),
            TrainableLayer(layer_type="Linear", in_features=64),
        ]
    )

golu: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=64),
            FunctionLayer(layer_type="GoLU"),
            TrainableLayer(layer_type="Linear", in_features=64, out_features=64),
            FunctionLayer(layer_type="GoLU"),
            TrainableLayer(layer_type="Linear", in_features=64),
        ]
    )

# MARK: BATCH CONFIG
# Configure batch parameters here. The cross-product of these lists will be used
# to create multiple experiment configurations.
batch_config: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.network_config": [(relu, "relu"), (gelu, "gelu"), (golu, "golu")],
    # "env_config.domain": ["ball_in_cup", "walker"],
    # "env_config.task": ["walk", "catch"],
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
    configs = [create_config(keys, config_values) for config_values in itertools.product(*batch_config.values())]

    coordinators: list[tuple[ExecutionCoordinator, str]] = []
    for i, config in enumerate(configs):
        # Setup specific name and coordinator
        run_name = get_name_from_config(config, i+1)
        coordinator = config_to_coordinator(config)
        replace_configurations(coordinator, config)
        coordinators.append((coordinator, run_name))

    return coordinators

def create_config(keys: list[str], config_values: tuple[Any | tuple[Any, str], ...]) -> dict[str, tuple[Any, str]]:
    config: dict[str, tuple[Any, str]] = {}
    for i, value in enumerate(config_values):
        # Ensure value is a tuple (actual_value, name)
        if isinstance(value, tuple):
            config[keys[i]] = value
        else:
            config[keys[i]] = (value, f"{keys[i]}-{value}")
    return config

def get_name_from_config(config: dict[str, tuple[Any, str]], index: int) -> str:
    name_parts = []
    for value in config.values():
        name_parts.append(value[1])
    return f"[{index}]_" + "_".join(name_parts)

def config_to_coordinator(config: dict[str, tuple[Any, str]]) -> ExecutionCoordinator:
    parser = RLParser()
    base_configs = parser.parse_args()
    coordinator = ExecutionCoordinator(base_configs)
    replace_configurations(coordinator, config)
    return coordinator

def replace_configurations(coordinator: ExecutionCoordinator, config: dict[str, tuple[Any, str]]):
    for key, value in config.items():
        keys = key.split('.')
        obj = coordinator
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value[0]) # value is a tuple (actual_value, name)
