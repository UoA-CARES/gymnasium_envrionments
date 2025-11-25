"""
This module enables batch execution of multiple reinforcement learning experiments
with varying configurations. It constructs ExecutionCoordinator instances for each
configuration combination and prepares them for execution.
"""

import itertools
from typing import Any

from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser

from cares_reinforcement_learning.util.configurations import FunctionLayer, MLPConfig, TrainableLayer

golu_a: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="GoLU"),
        ]
    )

golu_c: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="GoLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
    )

gelu_a: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="GELU"),
        ]
    )

gelu_c: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="GELU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
    )

relu_a: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

relu_c: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
    )

# MARK: BATCH CONFIG
# Configure batch parameters here. The cross-product of these lists will be used
# to create multiple experiment configurations.

# python run.py train cli --gym dmcs --domain cartpole --task swingup --batch 1 SAC --seeds 10 20 30 40 50 --max_workers 5
batch_config: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.actor_config": [(relu_a, "relu"), (gelu_a, "gelu"), (golu_a, "golu")],
    "alg_config.critic_config": [(relu_c, "relu"), (gelu_c, "gelu"), (golu_c, "golu")],
    "env_config.domain": ["cheetah", "cartpole", "finger", "walker"],
    "env_config.task": ["run", "swingup", "spin", "walk"],
}

# python run.py train cli --gym openai --task HalfCheetah-v4 --batch 1 SAC --seeds 10 20 30 40 50 --max_workers 5
# batch_config: dict[str, list[Any | tuple[Any, str]]] = {
#     "alg_config.actor_config": [(relu_a, "relu"), (gelu_a, "gelu"), (golu_a, "golu")],
#     "alg_config.critic_config": [(relu_c, "relu"), (gelu_c, "gelu"), (golu_c, "golu")],
#     "env_config.task": ["HalfCheetah-v4", "Humanoid-v4", "Ant-v4", "Hopper-v4"],
# }

def _skip(config: dict[str, tuple[Any, str]]) -> bool:
    """Determine if a given configuration combination should be skipped.
    E.g., task walker.catch doesn't exist.

    Args:
        config (dict[str, tuple[Any, str]]): A configuration mapping where each
            key is a configuration attribute path and each value is a tuple of
            (actual_value, name).
    Returns:
        bool: True if the configuration should be skipped, False otherwise.
    """
    # Homogeneous activations for actor and critic
    if config.get("alg_config.actor_config", (None, "A"))[1] != config.get("alg_config.critic_config", (None, "B"))[1]:
        return True

    # OpenAI Gym tasks have no domain, only task
    if config.get("env_config.domain") is None:
        return False # Do not skip

    # Match domain to task
    return not ((config.get("env_config.domain", (None,))[0] == "cartpole" and config.get("env_config.task", (None,))[0] == "swingup") \
        or (config.get("env_config.domain", (None,))[0] == "finger" and config.get("env_config.task", (None,))[0] == "spin") \
        or (config.get("env_config.domain", (None,))[0] == "cheetah" and config.get("env_config.task", (None,))[0] == "run") \
        or (config.get("env_config.domain", (None,))[0] == "walker" and config.get("env_config.task", (None,))[0] == "walk"))

# -------------------------------------------------------------------

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
    configs = [_create_config(keys, config_values) for config_values in itertools.product(*batch_config.values())]

    coordinators: list[tuple[ExecutionCoordinator, str]] = []
    i = 0
    for config in configs:
        # Certain combinations may be invalid - skip these
        if _skip(config):
            continue
        i += 1

        # Setup specific name and coordinator
        run_name = _get_name_from_config(config, i)
        coordinator = _config_to_coordinator(config)
        _replace_configurations(coordinator, config)
        coordinators.append((coordinator, run_name))

    return coordinators

def _create_config(keys: list[str], config_values: tuple[Any | tuple[Any, str], ...]) -> dict[str, tuple[Any, str]]:
    config: dict[str, tuple[Any, str]] = {}
    for i, value in enumerate(config_values):
        # Ensure value is a tuple (actual_value, name)
        if isinstance(value, tuple):
            config[keys[i]] = value
        else:
            config[keys[i]] = (value, f"{keys[i]}-{value}")
    return config

def _get_name_from_config(config: dict[str, tuple[Any, str]], index: int) -> str:
    name_parts = []
    for value in config.values():
        name_parts.append(value[1])
    return f"[{index}]_" + "_".join(name_parts)

def _config_to_coordinator(config: dict[str, tuple[Any, str]]) -> ExecutionCoordinator:
    parser = RLParser()
    base_configs = parser.parse_args()
    coordinator = ExecutionCoordinator(base_configs)
    _replace_configurations(coordinator, config)
    return coordinator

def _replace_configurations(coordinator: ExecutionCoordinator, config: dict[str, tuple[Any, str]]):
    for key, value in config.items():
        keys = key.split('.')
        obj = coordinator
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value[0]) # value is a tuple (actual_value, name)
