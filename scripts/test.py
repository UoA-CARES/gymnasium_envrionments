import logging

import cv2

import pandas as pd

from util.configurations import GymEnvironmentConfig
from envrionments.pyboy.pokemon.Pokemon import Pokemon, PokemonImage
from envrionments.pyboy.mario.Mario import Mario, MarioImage


def key_to_action(key):
    map = {
        115: 0,  # s - down
        97: 1,  # a - left
        100: 2,  # d - right
        119: 3,  # w - up
        122: 4,  # z - A
        120: 5,  # x - B
        # 32: 6,  #space - start
    }
    logging.info(f"Key: {key}")
    if key in map.keys():
        logging.info(f"Map: {map[key]}")
        return map[key]
    else:
        return -1


if __name__ == "__main__":
    args = {
        "gym": "pyboy",
        # 'task' : 'pokemon',
        "task": "mario",
    }
    config = GymEnvironmentConfig(**args)
    # env = PokemonImage(config)
    env = MarioImage(config)

    state = env.reset()
    image = env.grab_frame()

    while True:
        cv2.imshow("State", image)
        key = cv2.waitKey(0)
        action = key_to_action(key)
        if action == -1:
            break

        state, reward, done, _ = env.step(action, discrete=True)
        image = env.grab_frame()

        stats = env._generate_game_stats()
        # logging.info(stats)

        game_area = env.game_area()

        area = pd.DataFrame(game_area)

        print(area)
        logging.info(game_area.shape)
