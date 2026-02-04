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

from cares_reinforcement_learning.util.configurations import (
    FunctionLayer,
    MLPConfig,
    TrainableLayer,
)

# MARK: ACTIVATION LAYERS

# ---------------------------------------------------------------------
# MARK: 1 Layer
# ---------------------------------------------------------------------

# LeakyReLU -> GELU
# PReLU -> GoLU

# Leaky ReLU
leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
    ]
)
leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# PReLU
prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
    ]
)
prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# ---------------------------------------------------------------------
# MARK: 2 Layers
# ---------------------------------------------------------------------

# All - both
all_both_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
    ]
)
all_both_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_both_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

all_both_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
    ]
)
all_both_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_both_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# All - actor
all_actor_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
    ]
)
all_actor_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_actor_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

all_actor_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
    ]
)
all_actor_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_actor_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# All - critic
all_critic_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
all_critic_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_critic_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

all_critic_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
all_critic_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
all_critic_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# First - both
first_both_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_both_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_both_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

first_both_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_both_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_both_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# First - actor
first_actor_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_actor_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_actor_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

first_actor_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_actor_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_actor_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# First - critic
first_critic_leaky_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_critic_leaky_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_critic_leaky_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="LeakyReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

first_critic_prelu_a_sac: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
first_critic_prelu_a_td3: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256),
        FunctionLayer(layer_type="Tanh"),
    ]
)
first_critic_prelu_c: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="PReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
    ]
)

# MARK: BATCH CONFIG
# Configure batch parameters here. The cross-product of these lists will be used
# to create multiple experiment configurations.

# python3 run.py train cli --gym dmcs --domain cartpole --task swingup --batch 1 TD3 --seeds 10 20 30 40 50 --max_workers 5
batch_config_dmcs: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.actor_config": [(XXX_leaky_a_td3, "XXX_leaky"), (XXX_prelu_a_td3, "XXX_prelu")],
    "alg_config.critic_config": [(XXX_leaky_c, "XXX_leaky"), (XXX_prelu_c, "XXX_prelu")],
    "env_config.domain": ["cheetah", "cartpole", "finger", "walker"],
    "env_config.task": ["run", "swingup", "spin", "walk"],
}
# python3 run.py train cli --gym openai --task HalfCheetah-v4 --batch 1 TD3 --seeds 10 20 30 40 50 --max_workers 5
batch_config_openai: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.actor_config": [(XXX_leaky_a_td3, "XXX_leaky"), (XXX_prelu_a_td3, "XXX_prelu")],
    "alg_config.critic_config": [(XXX_leaky_c, "XXX_leaky"), (XXX_prelu_c, "XXX_prelu")],
    "env_config.task": ["HalfCheetah-v4", "Humanoid-v4", "Ant-v4", "Hopper-v4"],
}
batch_config = batch_config_openai

# This function can be customized to skip certain invalid or undesired configurations.
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
    if (
        config.get("alg_config.actor_config", (None, "A"))[1]
        != config.get("alg_config.critic_config", (None, "B"))[1]
    ):
        return True

    # OpenAI Gym tasks have no domain, only task
    if config.get("env_config.domain") is None:
        return False  # Do not skip

    # Match domain to task
    return not (
        (
            config.get("env_config.domain", (None,))[0] == "cartpole"
            and config.get("env_config.task", (None,))[0] == "swingup"
        )
        or (
            config.get("env_config.domain", (None,))[0] == "finger"
            and config.get("env_config.task", (None,))[0] == "spin"
        )
        or (
            config.get("env_config.domain", (None,))[0] == "cheetah"
            and config.get("env_config.task", (None,))[0] == "run"
        )
        or (
            config.get("env_config.domain", (None,))[0] == "walker"
            and config.get("env_config.task", (None,))[0] == "walk"
        )
    )


# -------------------------------------------------------------------
# MARK: INTERNALS
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
    configs = [
        _create_config(keys, config_values)
        for config_values in itertools.product(*batch_config.values())
    ]

    coordinators: list[tuple[ExecutionCoordinator, str]] = []
    i = 0
    for config in configs:
        # Certain combinations may be invalid - skip these
        # These are DIFFERENT from the [b_start, b_end] range filtering and don't show up at all
        if _skip(config):
            continue
        i += 1

        # Setup specific name and coordinator
        coordinator = _config_to_coordinator(config)
        _replace_configurations(coordinator, config)
        run_name = _get_name_from_config(config, i)
        coordinator.env_config.index = i  # Set index for range filtering
        coordinators.append((coordinator, run_name))

    # Support negative indexing (i.e. replace -1 with last index)
    num_coordinators = len(coordinators)
    b_start = coordinators[0][0].env_config.b_start
    b_end = coordinators[0][0].env_config.b_end
    if b_start < 0:
        b_start = num_coordinators + b_start + 1
    if b_end < 0:
        b_end = num_coordinators + b_end + 1
    for coordinator, _ in coordinators:
        coordinator.env_config.b_start = b_start
        coordinator.env_config.b_end = b_end

    return coordinators


def _create_config(
    keys: list[str], config_values: tuple[Any | tuple[Any, str], ...]
) -> dict[str, tuple[Any, str]]:
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
    coordinator = ExecutionCoordinator(base_configs, options={"noprint": True})
    _replace_configurations(coordinator, config)
    return coordinator


def _replace_configurations(
    coordinator: ExecutionCoordinator, config: dict[str, tuple[Any, str]]
):
    for key, value in config.items():
        keys = key.split(".")
        obj = coordinator
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value[0])  # value is a tuple (actual_value, name)
