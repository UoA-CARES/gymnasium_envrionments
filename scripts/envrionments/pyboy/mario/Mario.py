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
from envrionments.pyboy.Pyboy import Pyboy

class Mario(Pyboy):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config, rom_name='SuperMarioLand.gb', init_name='init.state')

        self.combo_actions = 1
        
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            # WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
            # Start is pause/unpause so don't need
        ]

        self.release_button = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            # WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

    def _stats_to_state(self, game_stats):
        # TODO figure out exactly what our observation space is - note we will have an image based version of this class
        state = []
        return state
    
    # @override
    def _run_action_on_emulator(self, action):
         # extra action for long jumping to the right
        if action == 5:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for i in range(self.act_freq):
                self.pyboy.tick()
                if i == 24:
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        else:
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
        # https://datacrystal.romhacking.net/wiki/Super_Mario_Land:RAM_map
        return {
            'lives' : self._get_lives(),
            'score': self._get_score(),
            'powerup': self._get_powerup(),
            'coins': self._get_coins(),
            'stage': self._get_stage(),
            'world': self._get_world(),
            'game_over': self._get_game_over(),
            'direction' : self._get_direction(), 
        }
    
    def _reward_stats_to_reward(self, reward_stats):
        reward_total = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        return reward_total
    
    def _calculate_reward_stats(self, new_state):
        # score reward is low priority
        return {
            'lives_reward': self._lives_reward(new_state),
            # 'score_reward': self._score_reward(new_state),
            'powerup_reward': self._powerup_reward(new_state),
            'coins_reward': self._coins_reward(new_state),
            'direction_reward': self._direction_reward(new_state),
            'stage_reward': self._stage_reward(new_state),
            'world_reward': self._world_reward(new_state),
            'game_over_reward': self._game_over_reward(new_state),
        }
    
    def _lives_reward(self, new_state):
        if new_state["lives"] - self.prior_game_stats["lives"] < 0:
            return (new_state["lives"] - self.prior_game_stats["lives"]) * 5
        return 0
    
    def _score_reward(self, new_state):
        if new_state["score"] - self.prior_game_stats["score"] > 0:
            return 1
        elif new_state["score"] - self.prior_game_stats["score"] == 0:
            return 0
        else:
            return -1
        
    def _powerup_reward(self, new_state):
        return new_state["powerup"] - self.prior_game_stats["powerup"]
    
    def _coins_reward(self, new_state):
        if new_state["coins"] - self.prior_game_stats["coins"] > 0:
            return 0.2
        else:
            return 0
        
    def _direction_reward(self, new_state):
        # TODO implement method to reward right direction while not rewarding
        # going right when running into a wall
        return 1 if (new_state['direction'] - self.prior_game_stats['direction'] == 1) else 0
    
    def _stage_reward(self, new_state):
        if new_state["stage"] - self.prior_game_stats["stage"] == -2:
            return 0
        return (new_state["stage"] - self.prior_game_stats["stage"]) * 5
    
    def _world_reward(self, new_state):
        return (new_state["world"] - self.prior_game_stats["world"]) * 5
    
    def _game_over_reward(self, new_state):
        if new_state["game_over"] == 1:
            return -5
        else:
            return 0
        
    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return True if game_stats['stage'] > self.prior_game_stats['stage'] else False
    
    def _get_lives(self):
        return self._read_m(0xDA15)
    
    def _get_score(self):
        return self._bit_count(self._read_m(0xC0A0))
    
    def _get_powerup(self):
        # 0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 0x03 = shrinking, 0x04 = invincibility blinking
        if self._read_m(0xFF99) == 0x02 or self._read_m(0xFF99) == 0x04:
            return 1
        else:
            return 0
        
    def _get_coins(self):
        return self._read_m(0xFFFA)
    
    def _get_stage(self):
        return self._read_m(0x982E)
    
    def _get_world(self):
        return self._read_m(0x982C)
    
    def _get_game_over(self):
        # Resetting game so that the agent doesn't need to use start button to start game
        if self._read_m(0xFFB3) == 0x3A:
            self.reset()
            logging.info('resetting')
            return 1
        return 0
    
    def _get_direction(self):
        return 1 if self._read_m(0xC20D) == 0x10 else 0
    
class MarioImage(Mario):
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