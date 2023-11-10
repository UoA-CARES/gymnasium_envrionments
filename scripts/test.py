import logging

import cv2

from util.configurations import GymEnvironmentConfig
from envrionments.pokemon.PokemonEnvironment import PokemonEnvironment, PokemonImage

def step(env, action):
    state, reward, done, _  = env.step(action)

    image = env.grab_frame()

    stats = env._generate_game_stats()
    logging.info(stats)
    cv2.imshow("State", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = {
        'gym' : 'pokemon',
        'task' : 'pokemon',
    }
    config = GymEnvironmentConfig(**args)
    env = PokemonImage(config)

    state = env.reset()
    image = env.grab_frame()
    cv2.imshow("State", image)
    cv2.waitKey(0)
    
    step(env, 0)
    step(env, 1)
    step(env, 2)
    step(env, 3)