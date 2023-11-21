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
import envrionments.pokemon.pokemon_constants as pkc

class PokemonEnvironment:
    def __init__(self, config: GymEnvironmentConfig) -> None:
        logging.info(f"Training with Task {config.task}")
        headless = False
        self.task = config.task

        self.rom_path = f'{Path.home()}/cares_rl_configs/pokemon/PokemonRed.gb'
        # self.init_path = f'{Path.home()}/cares_rl_configs/pokemon/init.state' # Full Initial 
        self.init_path = f'{Path.home()}/cares_rl_configs/pokemon/has_pokedex.state' # Has Squirtle

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
            # Removing start for now to reduce the complexity
        ]
        
        self.release_button = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
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

        self.prior_game_stats = self._generate_game_stats()
        self.screen = self.pyboy.botsupport_manager().screen()

        self.step_count = 0

        self.pyboy.set_emulation_speed(0)
        
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

        if action >= -1 and action < -0.666666666667:
            discrete_action = 0
        elif action >= -0.666666666667 and action < -0.33333333334:
            discrete_action = 1
        elif action >= -0.33333333334 and action < -0.000000001:
            discrete_action = 2
        elif action >= -0.000000001 and action < 0.333333332:
            discrete_action = 3
        elif action >= 0.333333332 and action < 0.666666665:
            discrete_action = 4
        else:
            discrete_action = 5
        
        # logging.info(discrete_action)

        # 7 action version (Incl. start)
        # if action >= -1 and action < -0.71428571429:
        #     discrete_action = 0
        # elif action >= -0.71428571429 and action < -0.428571429:
        #     discrete_action = 1
        # elif action >= -0.428571429 and action < -0.142857143:
        #     discrete_action = 2
        # elif action >= -0.142857143 and action < 0.142857143:
        #     discrete_action = 3
        # elif action >= 0.142857143 and action < 0.428571429:
        #     discrete_action = 4
        # elif action >= 0.428571429 and action < 0.71428571429:
        #     discrete_action = 5
        # else:
        #     discrete_action = 6

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
        # TODO figure out exactly what our observation space is - note we will have an image based version of this class
        state = []
        return state

    def _generate_game_stats(self):
        # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        return {
            'location' : self._get_location(),
            'party_size': self._get_party_size(),
            'ids': self._read_party_id(),
            'pokemon': [pkc.get_pokemon(id) for id in self._read_party_id()], 
            'levels': self._read_party_level(), 
            'type_id': self._read_party_type(),
            'type': [pkc.get_type(id) for id in self._read_party_type()],
            'hp': self._read_party_hp(),
            'xp': self._read_party_xp(),
            'status': self._read_party_status(),
            'badges': self._get_badge_count(),
            'caught_pokemon': self._read_caught_pokemon_count(),
            'seen_pokemon': self._read_seen_pokemon_count(),
            'money': self._read_money(),
            'events': self._read_events(),
        }

    def _reward_stats_to_reward(self, reward_stats):
        reward_total = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        # logging.info('total reward: {0}'.format(reward_total))
        return reward_total

    def _calculate_reward_stats(self, new_state):
        # rewards for new locations?
        # reward for triggering events?
        # reward for beating gynm leaders?
        return {
            'caught_reward': self._caught_reward(new_state),
            'seen_reward': self._seen_reward(new_state),
            'health_reward': self._health_reward(new_state),
            'xp_reward': self._xp_reward(new_state),
            'levels_reward': self._levels_reward(new_state),
            'badges_reward': self._badges_reward(new_state),
            'money_reward': self._money_reward(new_state),
            'event_reward': self._event_reward(new_state),
        }
    
    def _caught_reward(self, new_state):
        return new_state["caught_pokemon"] - self.prior_game_stats["caught_pokemon"]

    def _seen_reward(self, new_state):
        return new_state["seen_pokemon"] - self.prior_game_stats["seen_pokemon"]

    def _health_reward(self, new_state):
        return sum(new_state["hp"]["current"]) - sum(self.prior_game_stats["hp"]["current"])

    def _xp_reward(self, new_state):
        return sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])

    def _levels_reward(self, new_state):
        return sum(new_state["levels"]) - sum(self.prior_game_stats["levels"])

    def _badges_reward(self, new_state):
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _money_reward(self, new_state):
        return new_state["money"] - self.prior_game_stats["money"]
    
    def _event_reward(self, new_state):
        return sum(new_state["events"]) - sum(self.prior_game_stats["events"])

    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first gym (temporary)
        return True if self.prior_game_stats['badges'] > 0 else False
    
    def _get_location(self):
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {'x': x_pos, 
                'y': y_pos, 
                'map_id': map_n,
                'map': pkc.get_map_location(map_n)}

    def _get_party_size(self):
        return self._read_m(0xD163)

    def _get_badge_count(self):
        return self._bit_count(self._read_m(0xD356))

    def _read_party_id(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/pokemon_constants.asm
        return [self._read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def _read_party_type(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/type_constants.asm
        return [self._read_m(addr) for addr in [0xD170, 0xD171, 
                                               0xD19C, 0xD19D, 
                                               0xD1C8, 0xD1C9,
                                               0xD1F4, 0xD1F5,
                                               0xD220, 0xD221,
                                               0xD24C, 0xD24D]]
    
    def _read_party_level(self):
        return [self._read_m(addr) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]

    def _read_party_status(self):
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/status_constants.asm
        return [self._read_m(addr) for addr in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]]

    def _read_party_hp(self):
        hp = [self.read_hp(addr) for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        max_hp = [self.read_hp(addr) for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        return {'current' : hp, 
                'max' : max_hp}

    def _read_party_xp(self):
        return [self._read_triple(addr) for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]

    def read_hp(self, start):
        return 256 * self._read_m(start) + self._read_m(start+1)

    def _read_caught_pokemon_count(self):
        return sum([self._bit_count(self._read_m(i)) for i in range(0xD2F7, 0xD30A)])

    def _read_seen_pokemon_count(self):
        return sum([self._bit_count(self._read_m(i)) for i in range(0xD30A, 0xD31D)])

    def _read_money(self):
        return (100 * 100 * self._read_bcd(self._read_m(0xD347)) + 
                100 * self._read_bcd(self._read_m(0xD348)) +
                self._read_bcd(self._read_m(0xD349)))

    def _read_events(self):
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        # museum_ticket = (0xD754, 0)
        # base_event_flags = 13
        return [self._bit_count(self._read_m(i)) for i in range(event_flags_start, event_flags_end)]
        
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
    