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
import envrionments.pyboy.pokemon.pokemon_constants as pkc

class Pyboy:
    def __init__(self, config: GymEnvironmentConfig, rom_name : str, init_name : str) -> None:
        logging.info(f"Training with Task {config.task}")
        self.task = config.task

        self.rom_path = f'{config.rom_path}/{self.task}/{rom_name}'
        self.init_path = f'{config.rom_path}/{self.task}/{init_name}'

        self.valid_actions = [
        ]
        
        self.release_button = [
        ]

        self.act_freq = config.act_freq

        head, hide_window = ['headless', True] if config.headless else ['SDL2', False]
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

        self.pyboy.set_emulation_speed(config.emulation_speed)
        
        self.reset()

    @cached_property
    def min_action_value(self):
        return -1
    
    @cached_property
    def max_action_value(self):
        return 1

    @cached_property
    def observation_space(self):
        return self._stats_to_state(self._generate_game_stats())
    
    @cached_property
    def action_num(self):
        return 1

    def set_seed(self, seed):
        self.seed = seed
        # There isn't a random element to set that I am aware of...

    def reset(self):
        # restart game, skipping credits and selecting first pokemon
        print("reset")
        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

    def grab_frame(self, height=240, width=300):
        frame = self.screen.screen_ndarray()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to BGR for use with OpenCV
        return frame

    def step(self, action):
        # Actions excluding start
        self.step_count += 1

        bins = np.linspace(self.min_action_value, self.max_action_value, num=len(self.valid_actions))
        discrete_action = int(np.digitize(action, bins)) - 1 # number to index

        self._run_action_on_emulator(discrete_action)
        
        current_game_stats = self._generate_game_stats()
        state = self._stats_to_state(current_game_stats)

        reward_stats = self._calculate_reward_stats(current_game_stats)
        reward = self._reward_stats_to_reward(reward_stats)
        
        done = self._check_if_done(current_game_stats)
        
        self.prior_game_stats = current_game_stats

        truncated = True if self.step_count % 1000 == 0 else False
      
        return state, reward, done, truncated
    
    def _run_action_on_emulator(self, action):
        # press button then release after some steps - enough to move 
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            self.pyboy.tick()
            if i == 8: # ticks required to carry a "step" in the world
              self.pyboy.send_input(self.release_button[action])

    def _stats_to_state(self, game_stats):
        raise NotImplementedError("Override this method in the child class")

    def _generate_game_stats(self):
        raise NotImplementedError("Override this method in the child class")

    def _reward_stats_to_reward(self, reward_stats):
        raise NotImplementedError("Override this method in the child class")

    def _calculate_reward_stats(self, new_state):
        raise NotImplementedError("Override this method in the child class")
    
    def _check_if_done(self, game_stats):
        raise NotImplementedError("Override this method in the child class")
        
    def _read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def _read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self._read_m(addr))[-bit-1] == '1'

    # built-in since python 3.10
    def _bit_count(self, bits):
        return bin(bits).count('1')

    def _read_triple(self, start_add):
        return 256*256*self._read_m(start_add) + 256*self._read_m(start_add+1) + self._read_m(start_add+2)
    
    def _read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
        