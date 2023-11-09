from pathlib import Path

from pyboy import PyBoy
from pyboy.utils import WindowEvent

import logging

import cv2

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from util.configurations import GymEnvironmentConfig
import envrionments.pokemon_constants as pkc

class PokemonEnvironment:
    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training with Task {config.task}")
        headless = False
        self.task = config.task

        self.rom_path = f'{Path.home()}/cares_rl_configs/pokemon/PokemonRed.gb'
        # self.init_path = f'{Path.home()}/cares_rl_configs/pokemon/init.state'has_pokedex
        self.init_path = f'{Path.home()}/cares_rl_configs/pokemon/has_pokedex.state'
        #config.rom_path

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        
        # self.valid_actions.extend([
        #     WindowEvent.PRESS_BUTTON_START,
        #     WindowEvent.PASS
        # ])

        self.release_button = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.act_freq = 24

        head, hide_window = ['headless', True] if headless else ['SDL2', False]
        self.pyboy = PyBoy(
                self.rom_path,
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window=hide_window,
            )

        self.prior_game_stats = self._generate_game_stats(0)

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.screen = self.pyboy.botsupport_manager().screen()

    @cached_property
    def min_action_value(self):
        return 0
    
    @cached_property
    def max_action_value(self):
        return len(self.valid_actions)

    @cached_property
    def observation_space(self):
        raise NotImplementedError() # TODO figure out exactly what our observation space is - note we will have an image based version of this class
    
    @cached_property
    def action_num(self):
        return len(self.valid_actions)

    def set_seed(self, seed):
        self.seed = seed # There isn't a random element to set

    def reset(self):
        # restart game, skipping credits
        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

    def grab_frame(self, height=240, width=300):
        frame = self.screen.screen_ndarray()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
        return frame

    def step(self, action):
        self._run_action_on_emulator(action)

        current_game_stats = self._generate_game_stats(action)

        reward = self._calculate_reward(current_game_stats)
        done = self._check_if_done(current_game_stats)
        
        self.state = self._stats_to_state(current_game_stats)
        
        self.prior_game_stats = current_game_stats
      
        return self.state, reward, done, False # for consistency with open ai gym just add false for truncated
    
    def _run_action_on_emulator(self, action):
        # press button then release after some steps - enough to move 
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            self.pyboy.tick()
            if i == 8: # ticks required to carry a "step" in the world
              self.pyboy.send_input(self.release_button[action])

    def _stats_to_state(self, game_stats):
        state = []
        return state

    def _generate_game_stats(self, action):
        # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        return {
            'location' : self.get_location(),
            'pokemon_count': self.get_pokemon_count(),
            'ids': self.read_party_id(), 
            'levels': self.read_party_level(), 
            'type': self.read_party_type(),
            'hp': self.read_party_hp(),
            'status': self.read_party_status(),
            'badges': self.get_badge_count(),
        }

    def _calculate_reward(self, new_state):
        pass
    
    def _check_if_done(self, game_stats):
        return False
    
    def get_location(self):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)

        return {'x': x_pos, 
                'y': y_pos, 
                'map': map_n}

    def get_pokemon_count(self):
        return self.read_m(0xD163)

    def get_badge_count(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party_id(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/pokemon_constants.asm
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def read_party_type(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/type_constants.asm
        return [self.read_m(addr) for addr in [0xD170, 0xD171, 
                                               0xD19C, 0xD19D, 
                                               0xD1C8, 0xD1C9,
                                               0xD1F4, 0xD1F5,
                                               0xD220, 0xD221,
                                               0xD24C, 0xD24D]]
    
    def read_party_level(self):
        return [self.read_m(addr) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]

    def read_party_status(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/status_constants.asm
        return [self.read_m(addr) for addr in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]]

    def read_party_hp(self):
        hp = [self.read_hp(addr) for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        max_hp = [self.read_hp(addr) for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        return {'current' : hp, 
                'max' : max_hp}

    def read_party_xp(self):
        return [self.read_triple(addr) for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def read_seen_pokemon_count(self):
        return sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])

    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))

    def _get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
        )
        
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
class PokemonImage(PokemonEnvironment):
    def __init__(self, config: GymEnvironmentConfig, k=3):
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 84
        self.frame_height = 84

        logging.info(f"Image Observation is on")
        super().__init__(config)

    # @override
    @property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    # @override
    def reset(self):
        super().reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    # @override
    def step(self, action):
        # _, reward, done, truncated, _ = self.env.step(action)
        _, reward, done, truncated = super().step(action)

        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, truncated
    



