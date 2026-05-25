"""
Batch coordinator for testing one fractional activation at a time.

Environment variables:
    ACTIVATION:
        FractionalSwish
        FractionalSwishBeta
        FALU
        FractionalGELU
        SafeFractionalSwish
        SafeFALU
        SafeFractionalGELU
        SafeGLFractionalGELU
        ResidualFractionalGELU
        AdaptiveResidualFractionalGELU

    ALGORITHM:
        TD3 or SAC

    LAYERS:
        1 or 2

    PLACEMENT:
        all_both
        all_actor
        all_critic
        first_both
        first_actor
        first_critic
"""

import itertools
import os
from typing import Any

from execution_coordinator import ExecutionCoordinator
import execution_logger as logs
from util.rl_parser import RLParser

from cares_reinforcement_learning.util.configurations import (
    FunctionLayer,
    MLPConfig,
    TrainableLayer,
)

FRACTIONAL_ACTIVATIONS = [
    "FractionalSwish",
    "FractionalSwishBeta",
    "FALU",
    "FractionalGELU",
    "SafeFractionalSwish",
    "SafeFALU",
    "SafeFractionalGELU",
    "SafeGLFractionalGELU",
    "AdaptiveResidualFractionalGELU",
    "ResidualFractionalGELU"
]

PLACEMENTS = [
    "all_both",
    "all_actor",
    "all_critic",
    "first_both",
    "first_actor",
    "first_critic",
]

SELECTED_ACTIVATION = os.environ.get("ACTIVATION", "FractionalSwish")
ALGORITHM = os.environ.get("ALGORITHM", "TD3")
LAYERS = int(os.environ.get("LAYERS", "1"))
PLACEMENT = os.environ.get("PLACEMENT", "all_both")

if SELECTED_ACTIVATION not in FRACTIONAL_ACTIVATIONS:
    raise ValueError(
        f"Unknown activation {SELECTED_ACTIVATION}. "
        f"Choose from {FRACTIONAL_ACTIVATIONS}"
    )

if ALGORITHM not in ["TD3", "SAC"]:
    raise ValueError("ALGORITHM must be TD3 or SAC")

if LAYERS not in [1, 2]:
    raise ValueError("LAYERS must be 1 or 2")

if PLACEMENT not in PLACEMENTS:
    raise ValueError(f"PLACEMENT must be one of {PLACEMENTS}")

if LAYERS == 1 and PLACEMENT != "all_both":
    raise ValueError("For LAYERS=1, use PLACEMENT=all_both")


def make_actor_sac(hidden_activations: list[str]) -> MLPConfig:
    layers = []

    layers.append(TrainableLayer(layer_type="Linear", out_features=256))
    layers.append(FunctionLayer(layer_type=hidden_activations[0]))

    if len(hidden_activations) == 2:
        layers.append(
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256)
        )
        layers.append(FunctionLayer(layer_type=hidden_activations[1]))

    return MLPConfig(layers=layers)


def make_actor_td3(hidden_activations: list[str]) -> MLPConfig:
    layers = []

    layers.append(TrainableLayer(layer_type="Linear", out_features=256))
    layers.append(FunctionLayer(layer_type=hidden_activations[0]))

    if len(hidden_activations) == 2:
        layers.append(
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256)
        )
        layers.append(FunctionLayer(layer_type=hidden_activations[1]))

    layers.append(TrainableLayer(layer_type="Linear", in_features=256))
    layers.append(FunctionLayer(layer_type="Tanh"))

    return MLPConfig(layers=layers)


def make_critic(hidden_activations: list[str]) -> MLPConfig:
    layers = []

    layers.append(TrainableLayer(layer_type="Linear", out_features=256))
    layers.append(FunctionLayer(layer_type=hidden_activations[0]))

    if len(hidden_activations) == 2:
        layers.append(
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256)
        )
        layers.append(FunctionLayer(layer_type=hidden_activations[1]))

    layers.append(
        TrainableLayer(layer_type="Linear", in_features=256, out_features=1)
    )

    return MLPConfig(layers=layers)


def get_hidden_activations() -> tuple[list[str], list[str]]:
    act = SELECTED_ACTIVATION

    if LAYERS == 1:
        return [act], [act]

    if PLACEMENT == "all_both":
        return [act, act], [act, act]

    if PLACEMENT == "all_actor":
        return [act, act], ["ReLU", "ReLU"]

    if PLACEMENT == "all_critic":
        return ["ReLU", "ReLU"], [act, act]

    if PLACEMENT == "first_both":
        return [act, "ReLU"], [act, "ReLU"]

    if PLACEMENT == "first_actor":
        return [act, "ReLU"], ["ReLU", "ReLU"]

    if PLACEMENT == "first_critic":
        return ["ReLU", "ReLU"], [act, "ReLU"]

    raise ValueError(f"Unsupported placement {PLACEMENT}")


actor_hidden_activations, critic_hidden_activations = get_hidden_activations()

if ALGORITHM == "SAC":
    actor_config = make_actor_sac(actor_hidden_activations)
else:
    actor_config = make_actor_td3(actor_hidden_activations)

critic_config = make_critic(critic_hidden_activations)

experiment_name = f"{LAYERS}layer_{PLACEMENT}_{SELECTED_ACTIVATION}"


batch_config_dmcs: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.actor_config": [(actor_config, experiment_name)],
    "alg_config.critic_config": [(critic_config, experiment_name)],
    "env_config.domain": ["cheetah", "cartpole", "finger", "walker"],
    "env_config.task": ["run", "swingup", "spin", "walk"],
}

batch_config_openai: dict[str, list[Any | tuple[Any, str]]] = {
    "alg_config.actor_config": [(actor_config, experiment_name)],
    "alg_config.critic_config": [(critic_config, experiment_name)],
    "env_config.task": ["HalfCheetah-v4", "Humanoid-v4", "Ant-v4", "Hopper-v4"],
}

batch_config = batch_config_openai


def _skip(config: dict[str, tuple[Any, str]]) -> bool:
    if config.get("env_config.domain") is None:
        return False

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


logger = logs.get_main_logger()


def get_batch_coordinators() -> list[tuple[ExecutionCoordinator, str]]:
    keys = list(batch_config.keys())
    configs = [
        _create_config(keys, config_values)
        for config_values in itertools.product(*batch_config.values())
    ]

    coordinators: list[tuple[ExecutionCoordinator, str]] = []
    i = 0

    for config in configs:
        if _skip(config):
            continue

        i += 1
        coordinator = _config_to_coordinator(config)
        _replace_configurations(coordinator, config)
        run_name = _get_name_from_config(config, i)
        coordinator.env_config.index = i
        coordinators.append((coordinator, run_name))

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

        setattr(obj, keys[-1], value[0])
