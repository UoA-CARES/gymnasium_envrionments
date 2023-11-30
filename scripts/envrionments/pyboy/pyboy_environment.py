from functools import cached_property

import cv2
import numpy as np
from envrionments.gym_environment import GymEnvironment
from pyboy import PyBoy
from util.configurations import GymEnvironmentConfig


class PyboyEnvironment(GymEnvironment):
    def __init__(
        self, config: GymEnvironmentConfig, rom_name: str, init_name: str
    ) -> None:
        super().__init__(config)

        self.rom_path = f"{config.rom_path}/{self.task}/{rom_name}"
        self.init_path = f"{config.rom_path}/{self.task}/{init_name}"

        self.valid_actions = []

        self.release_button = []

        self.act_freq = config.act_freq

        head, hide_window = ["headless", True] if config.headless else ["SDL2", False]
        self.pyboy = PyBoy(
            self.rom_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window=hide_window,
        )

        self.prior_game_stats = self._generate_game_stats()
        self.screen = self.pyboy.botsupport_manager().screen()

        self.step_count = 0
        self.seed = 0

        self.pyboy.set_emulation_speed(config.emulation_speed)
        self.reset()

    @cached_property
    def min_action_value(self) -> float:
        return -1

    @cached_property
    def max_action_value(self) -> float:
        return 1

    @cached_property
    def observation_space(self) -> int:
        return len(self._stats_to_state(self._generate_game_stats()))

    @cached_property
    def action_num(self) -> int:
        return 1

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        # There isn't a random element to set that I am aware of...

    def reset(self) -> np.ndarray:
        # restart game, skipping credits and selecting first pokemon
        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)
        return self._stats_to_state(self._generate_game_stats())

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        frame = self.screen.screen_ndarray()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to BGR for use with OpenCV
        return frame

    def step(self, action: int) -> tuple:
        # Actions excluding start
        self.step_count += 1
        
        # For test.py: Comment out bins & discrete_action and uncomment following line
        # discrete_action = action
        bins = np.linspace(
            self.min_action_value, self.max_action_value, num=len(self.valid_actions)
        )
        discrete_action = int(np.digitize(action, bins)) - 1

        self._run_action_on_emulator(discrete_action)
        
        current_game_stats = self._generate_game_stats()
        state = self._stats_to_state(current_game_stats)

        reward_stats = self._calculate_reward_stats(current_game_stats)
        reward = self._reward_stats_to_reward(reward_stats)
        
        done = self._check_if_done(current_game_stats)
        
        self.prior_game_stats = current_game_stats

        truncated = self.step_count % 1000 == 0

        return state, reward, done, truncated
    
    def _run_action_on_emulator(self, action: int) -> None:
        # press button then release after some steps - enough to move 
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            self.pyboy.tick()
            if i == 8: # ticks required to carry a "step" in the world
              self.pyboy.send_input(self.release_button[action])

    def _stats_to_state(self, game_stats: dict) -> np.ndarray:
        raise NotImplementedError("Override this method in the child class")

    def _generate_game_stats(self) -> dict:
        raise NotImplementedError("Override this method in the child class")

    def _reward_stats_to_reward(self, reward_stats: dict) -> float:
        raise NotImplementedError("Override this method in the child class")

    def _calculate_reward_stats(self, new_state: dict) -> dict:
        raise NotImplementedError("Override this method in the child class")

    def _check_if_done(self, game_stats: dict) -> bool:
        raise NotImplementedError("Override this method in the child class")

    def _read_m(self, addr: int) -> int:
        return self.pyboy.get_memory_value(addr)

    def _read_bit(self, addr: int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self._read_m(addr))[-bit - 1] == "1"

    # built-in since python 3.10
    def _bit_count(self, bits: int) -> int:
        return bin(bits).count("1")

    def _read_triple(self, start_add: int) -> int:
        return (
            256 * 256 * self._read_m(start_add)
            + 256 * self._read_m(start_add + 1)
            + self._read_m(start_add + 2)
        )
        
    def _read_bcd(self, num: int) -> int:
        return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

    def _get_sprites(self) -> list:
        ss = []
        for i in range(40):  # game boy can only support 40 sprites on screen at a time
            s = self.pyboy.botsupport_manager().sprite(i)
            if s.on_screen:
                ss.append(s)
        return ss

    # function is a work in progress
    def game_area(self) -> np.ndarray:
        # shape = (20, 18)
        shape = (20, 16)
        game_area_section = (0, 2) + shape

        xx = game_area_section[0]
        yy = game_area_section[1]
        width = game_area_section[2]
        height = game_area_section[3]

        tilemap_background = self.pyboy.botsupport_manager().tilemap_background()
        game_area = np.asarray(
            tilemap_background[xx : xx + width, yy : yy + height], dtype=np.uint32
        )

        ss = self._get_sprites()
        for s in ss:
            _x = (s.x // 8) - xx
            _y = (s.y // 8) - yy
            if 0 <= _y < height and 0 <= _x < width:
                game_area[_y][_x] = s.tile_identifier

        return game_area
