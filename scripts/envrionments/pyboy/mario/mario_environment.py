from typing import Dict, List
from enum import Enum

from envrionments.pyboy.pyboy_environment import PyboyEnvironment
from pyboy import WindowEvent
from util.configurations import GymEnvironmentConfig
import numpy as np


class Moves(Enum):
    DOWN = 0
    LEFT = 1
    RIGHT = 2
    A = 3
    # B = 4
    F_LEAP = 4
    B_LEAP = 5
    UP = 6


class Tiles(Enum):
    MARIO = 0
    BIG_MARIO = 1
    SUPER_MARIO = 2
    STARMAN = 3
    PROJECTILE = 4
    UNSTOMPABLE = 5
    STOMPABLE = 6
    POWERUP = 7
    MARIO_PROJECTILE = 8
    LEVER = 9
    AIR = 10
    BLOCK = 11


class MarioEnvironment(PyboyEnvironment):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        self.stuck_count = 0
        self.screen_stuck = 0
        
        self.mario_x_position = 0
        self.mario_y_position = 0

        # All tiles & sprites on the screen
        self.mario_tiles = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                            44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80}
        self.plane = {99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
        self.submarine = {112, 113, 114, 115, 116, 117, 118, 119, 120, 121}
        self.mario_projectiles = {96, 110, 122}
        self.powerups = {131, 132, 134, 224, 229}
        self.stompable_enemies = {50, 144, 151, 152, 153, 160, 161, 162, 163, 164, 165, 166, 167, 176, 177,
                                  178, 179, 180, 181, 182, 183, 192, 193, 194, 195, 198, 199, 208, 209, 210, 211, 214, 215}
        # Underwater enemies, everything unstompable: 184, 185, 168, 169 (fish), 192 (jumping fish), 164, 165, 180, 181 (seahorse), 
        # 160, 161, 176, 177 (octopus)
        self.unstompable_enemies = {146, 147, 148, 149}
        # Not all enemies are added
        # self.first_boss = {170, 171, 186, 187, 198, 199, 202, 203, 204, 205, 206, 214, 215, 218, 219, 220}
        # moving_blocks = [230, 238, 239] (added to neutral blocks)
        # pushable_blocks = [128, 130, 354] (added to neutral blocks)
        self.neutral_blocks = {129, 142, 143, 231, 232, 233, 234, 235, 236, 301, 302, 303, 304, 319, 340, 352,
                               353, 355, 356, 357, 358, 359, 360, 361, 362, 381, 382, 383, 230, 238, 239, 128,
                               130, 354, 369, 370, 371}
        # explosion = [157, 158] (added to projectiles)
        # bill = [249] (added to projectiles)
        self.projectiles = {157, 158, 172, 188, 196, 197, 212, 213, 226, 227, 221, 222, 249}
        self.air = {300, 305, 306, 320, 321, 322, 324, 325, 326, 339, 350}

        self.combo_actions = 1
        
        self.mario_width = 2
        self.mario_height = 2

        super().__init__(config, rom_name="SuperMarioLand.gb", init_name="init.state")

        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)    
        
        self.valid_actions: List[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            # WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
        ]

        self.release_button: List[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            # WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
        ]
    
    # @override
    def step(self, action: int) -> tuple:
        # Confirms mario's height every step for game_area calculations
        if self._get_powerup() == 1 or self._get_powerup() == 2:
            self.mario_height = 3
        else:
            self.mario_height = 2

        # Actions excluding start
        self.step_count += 1

        if action == 1:
            action -= 0.01
        bins = np.linspace(
            self.min_action_value, self.max_action_value, num=len(Moves) + 1
        )
        discrete_action = int(np.digitize(action, bins)) - 1        

        self._run_action_on_emulator(discrete_action)
        
        current_game_stats = self._generate_game_stats()
        state = self._stats_to_state(current_game_stats)

        reward_stats = self._calculate_reward_stats(current_game_stats)
        reward = self._reward_stats_to_reward(reward_stats)
        
        done = self._check_if_done(current_game_stats)
        
        self.prior_game_stats = current_game_stats

        truncated = self.step_count % 5000 == 0

        return state, reward, done, truncated

    # @override
    def _run_action_on_emulator(self, action):
        # extra actions for long jumping to the right and left
        if action == Moves.UP.value:
            if self._get_world() == 1:
                action = Moves.A.value
            elif self._get_stage() != 3:
                action = Moves.A.value

        match action:
            case Moves.F_LEAP.value:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                for i in range(self.act_freq):
                    self.pyboy.tick()
                    if i == 10:
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                    if i == 12:
                        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            case Moves.B_LEAP.value:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
                for i in range(self.act_freq):
                    self.pyboy.tick()
                    if i == 10:
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                    if i == 11:
                        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            case Moves.A.value:
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                for i in range(self.act_freq):  
                    self.pyboy.tick()
                    if i == 11:
                        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            case Moves.UP.value:
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
                for i in range(self.act_freq):
                    self.pyboy.tick()
                    if i == 8: # ticks required to carry a "step" in the world
                        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
            case _:
                # press button then release after some steps - enough to move
                self.pyboy.send_input(self.valid_actions[action])
                for i in range(self.act_freq):
                    self.pyboy.tick()
                    if i == 8: # ticks required to carry a "step" in the world
                        self.pyboy.send_input(self.release_button[action])
            
    def _stats_to_state(self, game_stats: Dict[str, int]) -> List:
        # Agent struggles to learn using any variant of the game area state.
        # Even when reduced, it seems to be too big for training.

        # # Simplified state vector
        state: List = np.array([
            game_stats["lives"], 
            # game_stats["score"], 
            game_stats["powerup"], 
            game_stats["stuck"], 
            game_stats["land"][0], 
            game_stats["land"][1], 
            game_stats["far_land"][0],
            game_stats["far_land"][1],
            # game_stats["projectiles"],
            game_stats["nearby_enemies"], 
            game_stats["midrange_enemies"], 
            game_stats["far_enemies"],
            ])

        # # Reduced game area
        # state: List = np.array([self.game_area_red()]).flatten()

        # # Full game area
        # state: List = np.array([self.game_area()]).flatten()
        
        # # Simplified game area + stats
        # state: List = np.append(np.array([self.simplify_game_area()]).flatten(), 
        #                         np.array([game_stats["died"], game_stats["powerup"], game_stats["stuck"]]))

        # # Reduced game area + stats
        # state: List = np.append(np.array([self.game_area_red()]).flatten(),
        #                         np.array([game_stats["died"], game_stats["powerup"], game_stats["stuck"]]))
        
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
            "x_pos" :self._get_x_position(),
            "stuck": self._get_stuck(),
            "time": self._get_time(),
            "airbourne": self._get_airbourne(),
            "land": self._get_close_land(),
            "far_land": self._get_far_land(),
            "projectiles": self._get_front_projectiles(),
            "nearby_enemies": self._get_nearby_enemies(),
            "midrange_enemies": self._get_midrange_enemies(),
            "far_enemies": self._get_far_enemies(),
            "died": self._get_died(),
        }
    
    def _reward_stats_to_reward(self, reward_stats: Dict[str, int]) -> int:
        reward_total: int = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        return reward_total
    
    def _calculate_reward_stats(self, new_state: Dict[str, int]) -> Dict[str, int]:
        return {
            "lives_reward": self._lives_reward(new_state),
            # "score_reward": self._score_reward(new_state),
            "screen_reward": self._screen_reward(new_state),
            "powerup_reward": self._powerup_reward(new_state),
            # "coins_reward": self._coins_reward(new_state),
            "stage_reward": self._stage_reward(new_state),
            "world_reward": self._world_reward(new_state),
            # Game over is delayed and lives reward is basically the same
            # "game_over_reward": self._game_over_reward(new_state),
            "stuck_reward": self._stuck_reward(new_state),
        }
    
    def _lives_reward(self, new_state: Dict[str, int]) -> int:
        # Uses get_died instead of get lives because get lives memory address updates
        # ~8 steps after death whereas this is instant if fell off and ~3 steps if through
        # enemy collision
        if (new_state["died"] == 1 and 
            self.prior_game_stats["died"] == 0):
            return -30
        return 0
    
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
                return -5
        elif new_state["powerup"] - self.prior_game_stats["powerup"] > 0:
            return 1
        return 0

    def _coins_reward(self, new_state: Dict[str, int]) -> int:
        if new_state["coins"] - self.prior_game_stats["coins"] > 0:
            return 0.2
        else:
            return 0

    def _screen_reward(self, new_state):
        # Reward for moving to the right and thus updating the screen
        return 4 if(new_state["screen"] - self.prior_game_stats["screen"] > 0) else 0
    
    def _stage_reward(self, new_state):
        if new_state["stage"] >= 2:
            return (new_state["stage"] - self.prior_game_stats["stage"]) * 5
        return 0

    def _world_reward(self, new_state):
        if new_state["world"] >= 2:
            return (new_state["world"] - self.prior_game_stats["world"]) * 5
        return 0

    def _game_over_reward(self, new_state):
        return -5 if new_state["game_over"] == 1 else 0
        
    def _stuck_reward(self, new_state):
        # Negative reward if mario stays still or does not move to the right 
        # for a certain amount of time
        if (new_state["screen"] == self.prior_game_stats["screen"] and
            new_state["time"] != self.prior_game_stats["time"]):
            self.screen_stuck += 1
            if(new_state["x_pos"] == self.prior_game_stats["x_pos"]):
                self.stuck_count += 1
            else:
                self.stuck_count = 0
        else:
            self.screen_stuck = 0

        if self.screen_stuck >= 20:
            return -20
        elif self.stuck_count >= 10:
            return -10
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
        return self._read_m(0xC0AB) - 12

    def _get_x_position(self):
        return self._read_m(0xC202)

    def _get_powerup(self):
        # FFB5 (Does Mario have the Superball (0x00 = no, 0x02 = yes)
        # FF99 0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 
        # 0x03 = shrinking, 0x04 = invincibility blinking (when your big but 
        # get hit you are invincible while shrinking)
        # 3 = starman, 2 = superball, 1 = big, 0 = small
        if self._read_m(0xC0D3) == 0x00:
            if self._read_m(0xFFB5) != 0x02:
                if self._read_m(0xFF99) != 0x02:
                    if self._read_m(0xFF99) != 0x01:
                        return 0
                    return 1
                return 1
            return 2
        return 3

    def _get_time(self):
        # DA00 Timer (frames count down from 0x28 to 0x01 in a loop)
        # Used for stuck reward
        return self._read_m(0xDA00)

    def _get_stuck(self):
        if self.stuck_count >= 10:
            return 1
        return 0

    def _get_airbourne(self):
        return self._read_m(0xC20A)

    def _get_boundaries(self, x_distance, y_distance):
        # Grabs an area from the screen that is a certain distance beyond mario
        # Is postprocessed to ensure the boundaries are within the screen area.
        top_boundary = self.mario_y_position - y_distance
        bot_boundary = self.mario_y_position + y_distance + self.mario_height
        left_boundary = self.mario_x_position - x_distance
        right_boundary = self.mario_x_position + x_distance + self.mario_width       
        
        if self.mario_x_position - x_distance <= 0:
            left_boundary = self.mario_x_position
        elif self.mario_x_position + x_distance + self.mario_width >= 20:
            right_boundary = self.mario_x_position    
            
        if self.mario_y_position - y_distance <= 0:
            top_boundary = self.mario_y_position
        elif self.mario_y_position + y_distance + self.mario_height >= 16:
            bot_boundary = self.mario_y_position
        
        return (top_boundary, bot_boundary, left_boundary, right_boundary)    

    def _get_enemies(self, top_boundary, bot_boundary, left_boundary, right_boundary):
        game_area_array = self.game_area()
        for i in range (top_boundary, bot_boundary):
            for j in range(left_boundary, right_boundary):
                # game area variable not added yet so add it in
                if game_area_array[i][j] in self.stompable_enemies:
                    return 1
                elif game_area_array[i][j] in self.unstompable_enemies:
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
    
    def _get_land(self, x_distance):
        # This function checks the land x distance in front and behind mario to discern if 
        # there are blocks there or not.
        
        if self._get_airbourne() == 0:
            return (0,0)

        # gets the specified left and right boundaries to all the way down to bottom of screen
        (_, _, left_boundary, right_boundary) = self._get_boundaries(x_distance, 1)
        
        mario_bot = self.mario_y_position + self.mario_height
        mario_left = self.mario_x_position
        mario_right = self.mario_x_position + self.mario_width

        game_area_array = self.game_area()
        # 0 if floor offscreen, 1 if there is floor, -1 if there is no floor
        if mario_left == left_boundary:
            floor_behind = -1
        else:
            floor_behind = 1
            for b in range(left_boundary, left_boundary + 2):
                if game_area_array[15][b] not in self.neutral_blocks:
                    floor_behind = -1
                    break
    
        if mario_right == right_boundary:
            floor_front = -1
        else:
            floor_front = 1
            for y in range(right_boundary - 2, right_boundary):
                if game_area_array[15][y] not in self.neutral_blocks:
                    floor_front = -1
                    break

        return(floor_behind, floor_front)
    
    def _get_close_land(self):
        return self._get_land(2)
    
    def _get_far_land(self):
        return self._get_land(4)

    def _get_front_projectiles(self):
        # This function checks a 4x4 box that is 1 tile above mario for projectiles.
        # Binary yes/no of whether there are projectiles.

        if self.mario_y_position == 0 or self.mario_x_position == 0:
            return 0

        y_distance = 5
        x_distance = 4

        top_boundary = self.mario_y_position - y_distance
        bot_boundary = self.mario_y_position - 1
        left_boundary = self.mario_x_position + self.mario_width
        right_boundary = self.mario_x_position + self.mario_width + x_distance      
        
        # 0 if detection box offscreen, 1 if there is projectile, -1 if no projectile

        if self.mario_x_position + self.mario_width + x_distance >= 20:
            right_boundary = 20
        
        if self.mario_y_position - y_distance <= 0:
            top_boundary = 0

        game_area_array = self.game_area()
        for i in range(top_boundary, bot_boundary):
            for j in range(left_boundary, right_boundary):
                if game_area_array[i][j] in self.projectiles:
                    return 1
        
        return -1

    def _get_x_scroll(self):
        return np.ceil(self._read_m(0xFF43)/8)
    
    def _get_died(self):
        if self._read_m(0xDFE9) == 0x02:
            return 1
        return 0
        
    # @override
    def game_area(self) -> np.ndarray:
        # shape = (20, 18)
        shape = (20, 16)
        game_area_section = (0, 2) + shape

        # x scroll is a value between 0-256 that follows rightward movement of the 20x16 screen
        # through the 32x32 tile layout and reaches 0 when it does a complete loop of the layout
        horizontal_edge = 32
        left_screen = int(self._get_x_scroll())
        right_screen = left_screen + 20

        if right_screen > horizontal_edge:
            right_screen = right_screen % 32
        
        mario_seen = False

        xx = game_area_section[0]
        yy = game_area_section[1]
        width = game_area_section[2]
        height = game_area_section[3]

        tilemap_background = self.pyboy.botsupport_manager().tilemap_background()

        # If 20x16 screen is halfway through borders of tile layout, split into two and combine
        if right_screen < left_screen:
            first_half = tilemap_background[left_screen : horizontal_edge, yy : yy + height]
            second_half = tilemap_background[xx : xx + right_screen, yy : yy + height]

            for i in range(len(first_half)):
                first_half[i].extend(second_half[i])

            game_area = np.asarray(first_half, dtype=np.int32)
        else:
            game_area = np.asarray(
                tilemap_background[left_screen : right_screen, yy : yy + height], dtype=np.int32
            )

        ss = self._get_sprites()
        for s in ss:
            _x = (s.x // 8) - xx
            _y = (s.y // 8) - yy
            if 0 <= _y < height and 0 <= _x < width:
                if  s.tile_identifier in self.mario_tiles:
                    if mario_seen:
                        if _x < self.mario_x_position:
                            self.mario_x_position = _x
                        if _y < self.mario_y_position:
                            self.mario_y_position = _y
                    else:
                        self.mario_y_position = _y
                        self.mario_x_position = _x
                    
                    mario_seen = True
                game_area[_y][_x] = s.tile_identifier

        return game_area

    def _search_array(self, search_area, search_size, mario_added, searching_below_mario):
        new_tile_selection = []
        for i in range(search_size):
            # Adds mario only if not added yet
            if search_area[i] in self.mario_tiles and not mario_added:
                # get powerup returns 0 = Mario, 1 = big, 2 = flower, 3 = star
                new_tile_selection.append(self._get_powerup())
            elif search_area[i] in self.projectiles:
                new_tile_selection.append(Tiles.PROJECTILE.value)
            elif search_area[i] in self.unstompable_enemies:
                new_tile_selection.append(Tiles.UNSTOMPABLE.value)
            elif search_area[i] in self.stompable_enemies:
                new_tile_selection.append(Tiles.STOMPABLE.value)
            elif search_area[i] in self.powerups:
                new_tile_selection.append(Tiles.POWERUP.value)
            elif search_area[i] in self.mario_projectiles:
                new_tile_selection.append(Tiles.MARIO_PROJECTILE.value)
            elif search_area[i] == 255:
                new_tile_selection.append(Tiles.LEVER.value)
            elif search_area[i] in self.neutral_blocks:
                new_tile_selection.append(Tiles.BLOCK.value)
            else:
                # Ideally find all the air tiles but works for now
                new_tile_selection.append(Tiles.AIR.value)

        # Returns lowest value because 0 = highest priority
        new_tile = min(new_tile_selection)

        # Prioritises blocks if searching above mario, air if searching below mario
        if not searching_below_mario:
            if new_tile == Tiles.AIR.value and Tiles.BLOCK.value in new_tile_selection:
                new_tile = Tiles.BLOCK.value
        return new_tile


    def game_area_red(self):
        # Reduces game area to 1/4 of original size
        # 0 = Mario, 1 = big mario, 2 = flower power, 3 = starman, 
        # 4 = projectile, 5 = unstompable enemy, 6 = stompable enemy, 7 = powerups
        # 8 = air, 9 = block
        game_area_array = self.game_area()

        rows = game_area_array.shape[0]
        cols = game_area_array.shape[1]

        new_area = np.zeros((int(rows/2), int(cols/2)), dtype=np.int32)

        search_size = 4

        mario_added = False

        for i in range(0, rows, 2):
            for j in range(0, cols, 2):
                search_area = [
                    game_area_array[i][j],
                    game_area_array[i][j+1],
                    game_area_array[i+1][j],
                    game_area_array[i+1][j+1]
                    ]
                
                searching_below_mario = False

                if i >= self.mario_y_position + 1:
                    searching_below_mario = True

                new_tile = self._search_array(search_area, search_size, mario_added, searching_below_mario)
                
                if not mario_added and new_tile == 0:
                    mario_added = True

                new_area[int(i/2)][int(j/2)] = new_tile

        return new_area
    
    def simplify_game_area(self):
        game_area_array = self.game_area()

        rows = game_area_array.shape[0]
        cols = game_area_array.shape[1]

        new_area = np.zeros((rows, cols), dtype=np.int32)

        for i in range(rows):
            for j in range(cols):
                if game_area_array[i][j] in self.mario_tiles:
                # get powerup returns 0 = Mario, 1 = big, 2 = flower, 3 = star
                    new_area[i][j] = self._get_powerup()
                elif game_area_array[i][j] in self.projectiles:
                    new_area[i][j] = Tiles.PROJECTILE.value
                elif game_area_array[i][j] in self.unstompable_enemies:
                    new_area[i][j] = Tiles.UNSTOMPABLE.value 
                elif game_area_array[i][j] in self.stompable_enemies:
                    new_area[i][j] = Tiles.STOMPABLE.value
                elif game_area_array[i][j] in self.powerups:
                    new_area[i][j] = Tiles.POWERUP.value
                elif game_area_array[i][j] in self.mario_projectiles:
                    new_area[i][j] = Tiles.MARIO_PROJECTILE.value
                elif game_area_array[i][j] == 255:
                    new_area[i][j] = Tiles.LEVER.value
                elif game_area_array[i][j] in self.neutral_blocks:
                    new_area[i][j] = Tiles.BLOCK.value
                else:
                    # Ideally find all the air tiles but works for now
                    new_area[i][j] = Tiles.AIR.value
        
        return new_area
    
