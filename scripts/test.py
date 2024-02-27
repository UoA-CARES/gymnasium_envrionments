import logging

import cv2
import pandas as pd
from envrionments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig


def key_to_action(button: int):
    key_map = {
        115: 0,  # s - down
        97: 1,  # a - left
        100: 2,  # d - right
        119: 3,  # w - up
        122: 4,  # z - A
        120: 5,  # x - B
        # 32: 6,  #space - start
    }
    logging.info(f"Key: {button}")
    if button in key_map:
        logging.info(f"Map: {key_map[button]}")
        return key_map[button]
    return -1


if __name__ == "__main__":
    args = {
        "gym": "pyboy",
        # 'task' : 'pokemon',
        "task": "mario",
    }
    env_config = GymEnvironmentConfig(**args)

    env_factory = EnvironmentFactory()

    env = env_factory.create_environment(
        env_config
    )  # This line should be here for seed consistency issues

    state = env.reset()
    image = env.grab_frame()

    while True:
        cv2.imshow("State", image)
        key = cv2.waitKey(0)
        action = key_to_action(key)
        if action == -1:
            break

        # TODO - action needs to be made continuous
        state, reward, done, _ = env.step(action)
        image = env.grab_frame()

        stats = env._generate_game_stats()

        game_area = env.game_area()

        area = pd.DataFrame(game_area)

        print(area)
        logging.info(game_area.shape)
