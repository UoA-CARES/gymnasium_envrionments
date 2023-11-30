from typing import Dict, List

from envrionments.pyboy.pyboy_environment import PyboyEnvironment
from pyboy import WindowEvent
from util.configurations import GymEnvironmentConfig
import numpy as np


class MarioEnvironment(PyboyEnvironment):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        self.stuck_count = 0

        self.mario_x_position = 0
        self.mario_y_position = 0

        self.stompable_enemies = {144, 50, 151, 152, 153, 160, 161, 162, 163, 176, 177, 178, 179, 192, 193, 194, 195, 208, 209, 210, 211, 
        164, 165, 166, 167, 180, 181, 182, 183}
        self.unstompable_enemies = {146, 147, 148, 149}

        self.mario_tiles = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80}
        
        self.neutral_blocks = {142, 143, 231, 232, 233, 234, 235, 236, 301, 302, 303, 304, 319, 340, 352, 353, 355, 356, 357, 358, 359,
        360, 361, 362, 381, 382, 383}

        self.projectiles = {172, 188, 196, 197, 212, 213, 226, 227, 221, 222}

        self.combo_actions = 1

        super().__init__(config, rom_name="SuperMarioLand.gb", init_name="init.state")        
        
        self.valid_actions: List[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            # WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_button: List[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            # WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]
    
    # @override
    def _run_action_on_emulator(self, action):
         # extra action for long jumping to the right
        if action == 5:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            for i in range(self.act_freq):  
                self.pyboy.tick()
                if i == 10:
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                if i == 11:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        elif action == 3:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            for i in range(self.act_freq):
                self.pyboy.tick()
                if i == 11:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        else:
            # press button then release after some steps - enough to move
            self.pyboy.send_input(self.valid_actions[action])
            for i in range(self.act_freq):
                self.pyboy.tick()
                if i == 8: # ticks required to carry a "step" in the world
                    self.pyboy.send_input(self.release_button[action])
            
    def _stats_to_state(self, game_stats: Dict[str, int]) -> List:
        # TODO figure out exactly what our observation space is - note we will have an image based version of this class
        state: List = np.array([
            game_stats["lives"], 
            game_stats["score"], 
            game_stats["powerup"], 
            game_stats["stuck"], 
            game_stats["land"][0], 
            game_stats["land"][1], 
            game_stats["projectiles"]
            ])
        return state
    
    def _generate_game_stats(self) -> Dict[str, int]:
        return {
            "lives": self._get_lives(),
            "score": self._get_score(),
            "powerup": self._get_powerup(),
            "coins": self._get_coins(),
            "stage": self._get_stage(),
            "world": self._get_world(),
            "game_over": self._get_game_over(),
            "screen" : self._get_screen(),
            # "direction" : self._get_direction(),
            "x_pos" :self._get_x_position(),
            "stuck": self._get_stuck(),
            "time": self._get_time(),
            "airbourne": self._get_airbourne(),
            "land": self._get_land(),
            "projectiles": self._get_front_projectiles(),
        }
    
    def _reward_stats_to_reward(self, reward_stats: Dict[str, int]) -> int:
        reward_total: int = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        return reward_total
    
    def _calculate_reward_stats(self, new_state: Dict[str, int]) -> Dict[str, int]:
        return {
            "lives_reward": self._lives_reward(new_state),
            "score_reward": self._score_reward(new_state),
            "screen_reward": self._screen_reward(new_state),
            "powerup_reward": self._powerup_reward(new_state),
            # "coins_reward": self._coins_reward(new_state),
            "stage_reward": self._stage_reward(new_state),
            "world_reward": self._world_reward(new_state),
            "game_over_reward": self._game_over_reward(new_state),
            "stuck_reward": self._stuck_reward(new_state),
        }
    
    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        if new_state["lives"] - self.prior_game_stats["lives"] < 0:
            self.reset()
        
        return (new_state["lives"] - self.prior_game_stats["lives"]) * 20
    
    def _score_reward(self, new_state: Dict[str, int]) -> int:
        if new_state["score"] - self.prior_game_stats["score"] > 0:
            return 0.5
        if new_state["score"] - self.prior_game_stats["score"] == 0:
            return 0
        return -0.5

    def _powerup_reward(self, new_state: Dict[str, int]) -> int:
        # Return positive reward for gaining powerup. Negative reward for 
        # losing powerup except for when starman runs out of time
        if new_state["powerup"] - self.prior_game_stats["powerup"] < 0:
            if self.prior_game_stats["powerup"] == 3:
                return 0
            else:
                return -1
        elif new_state["powerup"] - self.prior_game_stats["powerup"] > 0:
            return 1
        return 0

    def _coins_reward(self, new_state: Dict[str, int]) -> int:
        if new_state["coins"] - self.prior_game_stats["coins"] > 0:
            return 0.2
        else:
            return 0

    def _screen_reward(self, new_state):
        return 0.3 if(new_state["screen"] - self.prior_game_stats["screen"] > 0) else 0
    
    def _stage_reward(self, new_state):
        if new_state["stage"] - self.prior_game_stats["stage"] == -2:
            return 0
        return (new_state["stage"] - self.prior_game_stats["stage"]) * 5

    def _world_reward(self, new_state):
        return (new_state["world"] - self.prior_game_stats["world"]) * 5

    def _game_over_reward(self, new_state):
        if new_state["game_over"] == 1:
            return -5
        return 0

    def _stuck_reward(self, new_state):
        if (new_state["screen"] == self.prior_game_stats["screen"] and 
            new_state["x_pos"] == self.prior_game_stats["x_pos"] and
            new_state["time"] != self.prior_game_stats["time"]):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        
        if self.stuck_count >= 10:
            return -2
        else:
            return 0
    
    def _check_if_done(self, game_stats):
        # Setting done to true if agent beats first level
        return game_stats["stage"] > self.prior_game_stats["stage"]

    def _get_lives(self):
        return self._read_m(0xDA15)
    
    def _get_score(self):
        return self._bit_count(self._read_m(0xC0A0))
    
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
            return 1
        return 0
    
    def _get_screen(self):
        return self._read_m(0xC0AB)

    def _get_x_position(self):
        return self._read_m(0xC202)

    def _get_powerup(self):
        # 0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 0x03 = shrinking, 0x04 = invincibility blinking
        # FFB5 (Does Mario have the Superball (0x00 = no, 0x02 = yes)
        # 3 = invincible (starman?), 2 = superball, 1 = big, 0 = small
        if self._read_m(0xFF99) != 0x04:
            if self._read_m(0xFFB5) != 0x02:
                if self._read_m(0xFF99) != 0x02:
                    return 0
                return 1
            return 2
        return 3

    def _get_time(self):
        # DA00       3    Timer (frames, seconds (Binary-coded decimal), 
        # hundreds of seconds (Binary-coded decimal)) (frames count down from 0x28 to 0x01 in a loop)
        return self._read_m(0xDA00)

    def _get_stuck(self):
        if self.stuck_count >= 10:
            return 1
        return 0

    def _get_airbourne(self):
        return self._read_m(0xC20A)

    def _get_boundaries(self, x_distance, y_distance):
        # add function to check if mario big or small

        width = 2
        height = 3 if (self.prior_game_stats["powerup"] != 0) else 2
        
        top_boundary = self.mario_y_position - y_distance
        bot_boundary = self.mario_y_position + y_distance + height
        left_boundary = self.mario_x_position - x_distance
        right_boundary = self.mario_x_position + x_distance + width       
        
        if self.mario_x_position - x_distance <= 0:
            left_boundary = self.mario_x_position
        elif self.mario_x_position + x_distance + width >= 20:
            right_boundary = self.mario_x_position    
            
        if self.mario_y_position - y_distance <= 0:
            top_boundary = self.mario_y_position
        elif self.mario_y_position + y_distance + height >= 16:
            bot_boundary = self.mario_y_position
        
        return (top_boundary, bot_boundary, left_boundary, right_boundary)    

    def _get_enemies(self, top_boundary, bot_boundary, left_boundary, right_boundary):
        for i in range (left_boundary, right_boundary):
            for j in range(top_boundary, bot_boundary):
                # game area variable not added yet so add it in
                if self.game_area[i][j] in self.stompable_enemies:
                    return 1
                elif self.game_area[i][j] in self.unstompable_enemies:
                    return 2
                # add projectiles later        
        return 0    

    def _get_nearby_enemies(self):
        (top_boundary, bot_boundary, left_boundary, right_boundary) = self._get_boundaries(2, 2)
        return self._get_enemies(top_boundary, bot_boundary, left_boundary, right_boundary)    
    
    def _get_midrange_enemies(self):
        (top_boundary, bot_boundary, left_boundary, right_boundary) = self._get_boundaries(4, 4)
        return self._get_enemies(top_boundary, bot_boundary, left_boundary, right_boundary)    
    
    def _get_far_enemies(self):
        (top_boundary, bot_boundary, left_boundary, right_boundary) = self._get_boundaries(6, 6)
        return self._get_enemies(top_boundary, bot_boundary, left_boundary, right_boundary)

    def _get_land(self):
        if self._get_airbourne() == 0:
            return (0,0)
        width = 2
        height = 3 if (self.prior_game_stats["powerup"] != 0) else 2

        (top_boundary, bot_boundary, left_boundary, right_boundary) = self._get_boundaries(2, 1)
        
        mario_bot = self.mario_y_position + height
        mario_left = self.mario_x_position
        mario_right = self.mario_x_position + width

        game_area_array = self.game_area()
        # 0 if floor offscreen, 1 if there is floor, -1 if there is no floor
        if mario_left == left_boundary:
            floor_behind = 0
        else:
            floor_behind = 1
            for a in range(mario_bot, bot_boundary):
                for b in range(left_boundary, mario_left):
                    if game_area_array[a][b] not in self.neutral_blocks:
                        floor_behind = -1
                        break
    
        if mario_right == right_boundary:
            floor_front = 0
        else:
            floor_front = 1
            for x in range(mario_bot, bot_boundary):
                for y in range(mario_right, right_boundary):
                    if game_area_array[x][y] not in self.neutral_blocks:
                        floor_front = -1
                        break

        return(floor_behind, floor_front)

    def _get_front_projectiles(self):
        if self.mario_y_position == 0 or self.mario_x_position == 0:
            return 0

        width = 2

        # 4x4 box that is 1 above mario
        y_distance = 5
        x_distance = 4

        top_boundary = self.mario_y_position - y_distance
        bot_boundary = self.mario_y_position - 1
        left_boundary = self.mario_x_position + width
        right_boundary = self.mario_x_position + width + x_distance      
        
        # 0 if detection box offscreen, 1 if there is projectile, -1 if no projectile

        if self.mario_x_position + width + x_distance >= 20:
            right_boundary = 20
        
        if self.mario_y_position - y_distance <= 0:
            top_boundary = 0

        game_area_array = self.game_area()
        for i in range(top_boundary, bot_boundary):
            for j in range(left_boundary, right_boundary):
                if game_area_array[i][j] in self.projectiles:
                    return 1
        
        return -1

    # @override
    def game_area(self) -> np.ndarray:
        # shape = (20, 18)
        shape = (20, 16)
        game_area_section = (0, 2) + shape

        mario_seen = False

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
                if  not mario_seen and s.tile_identifier in self.mario_tiles:
                    self.mario_x_position = _x
                    self.mario_y_position = _y
                    mario_seen = True
                game_area[_y][_x] = s.tile_identifier

        return game_area
