from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from util.configurations import SMACConfig


class SMACEnvironment(GymEnvironment):
    def __init__(self, config: SMACConfig, evaluation: bool = False) -> None:
        super().__init__(config)

        team_type: str = config.domain

        opponent_type: str = config.task

        distribution_config = {
            "n_units": 5,
            "n_enemies": 5,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.45, 0.45, 0.1],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }

        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=distribution_config,
            map_name="10gen_terran",
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        self.env_info = self.env.get_env_info()

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def min_action_value(self) -> float:
        return 0

    @cached_property
    def observation_space(self) -> dict[str, int]:
        observation_space: dict[str, int] = {}

        observation_space["obs_shape"] = self.env_info["obs_shape"]

        observation_space["state_shape"] = self.env_info["state_shape"]
        observation_space["n_agents"] = self.env_info["n_agents"]

        return observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env_info["n_actions"]

    def sample_action(self) -> list[int]:
        actions = []
        for agent_id in range(self.env_info["n_agents"]):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)
        return actions

    def set_seed(self, seed: int) -> None:
        pass

    def reset(self, training: bool = True) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def _step(self, action: int) -> tuple:
        state, reward, done, truncated, info = self.env.step(np.int64(action))
        return state, reward, done, truncated, info

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
