from functools import cached_property
from typing import Any

import cv2
import numpy as np
from environments.marl_environment import MARLEnvironment
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from util.configurations import SMAC2Config


class SMAC2Environment(MARLEnvironment):
    def __init__(self, config: SMAC2Config, evaluation: bool = False) -> None:
        super().__init__(config)

        self.distribution_config = {
            "n_units": 5,
            "n_enemies": 1,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine"],
                "weights": [1.0],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 3,
                "map_x": 32,
                "map_y": 32,
            },
        }

        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=self.distribution_config,
            map_name="10gen_terran",
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        self.env_info = self.env.get_env_info()

        self.reset(training=not evaluation)

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def observation_space(self) -> dict[str, int]:
        observation_space: dict[str, int] = {}

        observation_space["obs"] = self.env_info["obs_shape"]

        observation_space["state"] = self.env_info["state_shape"]
        observation_space["num_agents"] = self.env_info["n_agents"]

        return observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env_info["n_actions"]

    def get_available_actions(self) -> np.ndarray:
        actions = []
        for agent_id in range(self.env_info["n_agents"]):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            actions.append(avail_actions)
        return np.array(actions)

    def sample_action(self) -> list[int]:
        actions = []
        for agent_id in range(self.env_info["n_agents"]):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)
        return actions

    def set_seed(self, seed: int) -> None:
        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=self.distribution_config,
            map_name="10gen_terran",
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            seed=seed,
        )

        self.env_info = self.env.get_env_info()

        self.reset()

    def reset(self, training: bool = True) -> dict[str, Any]:
        marl_state = {}
        obs, state = self.env.reset()

        marl_state["state"] = state
        marl_state["obs"] = obs
        marl_state["avail_actions"] = self.env.get_avail_actions()

        return marl_state

    def _step(self, actions: list[int]) -> tuple:
        marl_state = {}
        reward, done, info = self.env.step(actions)

        marl_state["state"] = self.env.get_state()
        marl_state["obs"] = self.env.get_obs()
        marl_state["avail_actions"] = self.env.get_avail_actions()

        return marl_state, reward, done, done, info

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render(mode="rgb_array")
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
