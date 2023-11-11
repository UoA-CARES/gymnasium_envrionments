import logging

import cv2

from util.configurations import GymEnvironmentConfig
from envrionments.pokemon.PokemonEnvironment import PokemonEnvironment, PokemonImage

def key_to_action(key):
    map = {
        115: 0,
        97: 1,
        100: 2,
        119: 3,
        122: 4,
        120: 5,
        32: 6, 
    }
    if key in map.keys():
        return map[key]
    else:
        return -1

if __name__ == '__main__':
    args = {
        'gym' : 'pokemon',
        'task' : 'pokemon',
    }
    config = GymEnvironmentConfig(**args)
    env = PokemonImage(config)

    state = env.reset()
    image = env.grab_frame()
    
    while True:
        cv2.imshow("State", image)
        key = cv2.waitKey(0)
        action = key_to_action(key)
        if action == -1:
            break

        state, reward, done, _  = env.step(action)
        image = env.grab_frame()

        stats = env._generate_game_stats()
        logging.info(stats)