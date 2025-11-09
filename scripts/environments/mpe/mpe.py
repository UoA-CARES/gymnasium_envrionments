from functools import cached_property
from typing import Any

import cv2
import numpy as np
from environments.marl_environment import MARLEnvironment
from mpe2 import all_modules as mpe_all
from util.configurations import MPEConfig
from pettingzoo.utils.env import ParallelEnv, AgentID

ALL_ENV_MODULES = {
    **mpe_all.mpe_environments,
    # Add others here, e.g., "smac/3m": smac_3m
}


def make_env(env_name: str, render_mode=None) -> ParallelEnv:
    if f"mpe/{env_name}" not in ALL_ENV_MODULES:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {list(ALL_ENV_MODULES.keys())}"
        )

    module = ALL_ENV_MODULES[f"mpe/{env_name}"]
    return (module.parallel_env)(render_mode=render_mode)


class MPE2Environment(MARLEnvironment):
    def __init__(self, config: MPEConfig, evaluation: bool = False) -> None:
        super().__init__(config)

        self.env = make_env(env_name=self.task, render_mode="rgb_array")

        self.agents: list[AgentID] = []

        self.seed = 10

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def observation_space(self) -> dict[str, int]:
        """Return core observation/state dimensions for homogeneous MPE2 tasks."""
        agent_name = self.env.agents[0]

        obs_shape = self.env.observation_space(agent_name).shape[0]
        num_agents = self.env.num_agents
        state_shape = obs_shape * num_agents  # simple concatenated global state

        return {
            "obs": obs_shape,
            "state": state_shape,
            "num_agents": num_agents,
        }

    @cached_property
    def action_num(self) -> int:
        return self.env.action_space(self.env.agents[0]).n

    def get_available_actions(self) -> np.ndarray:
        return np.ones((len(self.agents), self.action_num), dtype=np.int32)

    def sample_action(self) -> list[int]:
        actions = []
        avail_actions = self.get_available_actions()
        for agent_id in range(len(self.agents)):
            avail_actions_ind = np.nonzero(avail_actions[agent_id])[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)
        return actions

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.env.reset(seed=seed)

    def reset(self, training: bool = True) -> dict[str, Any]:
        """Reset PettingZoo parallel env and return MARL-compatible state dict."""
        obs_dict, info = self.env.reset(seed=self.seed)

        self.agents = self.env.agents

        # Stack per-agent observations into an array
        obs = np.stack([obs_dict[a] for a in self.env.agents])
        state = obs.flatten()  # simple concatenated global state

        marl_state = {
            "obs": obs,
            "state": state,
            "avail_actions": self.get_available_actions(),
        }
        return marl_state

    def _step(self, actions: list[int]) -> tuple:
        action_dict = {agent: act for agent, act in zip(self.env.agents, actions)}
        obs_dict, rewards, terminations, truncations, infos = self.env.step(action_dict)

        obs = np.stack([obs_dict[a] for a in self.agents])
        state = obs.flatten()
        avail_actions = self.get_available_actions()

        reward = rewards[list(self.agents)[0]]  # shared reward
        done = np.array([terminations[a] or truncations[a] for a in self.agents])
        all_done = np.all(done)

        marl_state = {"obs": obs, "state": state, "avail_actions": avail_actions}
        return marl_state, reward, all_done, all_done, infos

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
