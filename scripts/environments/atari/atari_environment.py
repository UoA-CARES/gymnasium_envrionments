from functools import cached_property

import cv2
import numpy as np
from environments.gym_environment import GymEnvironment
from gymnasium import spaces
from util.configurations import AtariConfig
import envpool


class AtariEnvironment(GymEnvironment):
    def __init__(self, config: AtariConfig, seed: int, evaluation: bool) -> None:
        super().__init__(config)
        self.env = envpool.make_gymnasium(
            config.task,
            num_envs=1, 
            seed=seed,
            img_width=config.frame_width,
            img_height=config.frame_height,
            episodic_life=evaluation, 
            reward_clip=evaluation, 
            stack_num=config.frames_to_stack,
        )
        if config.display == 1:
            self.name = f"{config.task}-{seed}"
            cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL)

        self.reset()

    @cached_property
    def max_action_value(self) -> float:
        return self.env.action_space.high[0]

    @cached_property
    def min_action_value(self) -> float:
        return self.env.action_space.low[0]


    @cached_property
    def observation_space(self) -> tuple:
        obs_shape = self.env.observation_space.shape
        return obs_shape


    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = self.env.action_space.n
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num


    def sample_action(self) -> int:
        return np.array([self.env.action_space.sample()], dtype=int)


    def set_seed(self, seed: int) -> None:
        self.env.reset()
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


    def reset(self, training: bool = True) -> np.ndarray:
        state, _ = self.env.reset()
        self.state = state[0]
        return self.state
    

    def _step(self, action: int) -> tuple:
        state, reward, terminated, truncated, info = self.env.step(action)
        self.state = state[0]
        return state[0], reward[0], terminated[0], truncated[0], {}
    

    def grab_frame(self, height: int = 232, width: int = 232) -> np.ndarray:
        if len(self.state.shape) == 4:
            # RGB
            frame = cv2.cvtColor(np.moveaxis(self.state[-3:], 0, -1), cv2.COLOR_RGB2BGR)
        else:
            # Grayscale
            frame = self.state[-1]
            frame = np.stack([frame]*3, axis=-1)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)


    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
