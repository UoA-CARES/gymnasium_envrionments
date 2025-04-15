import logging
import time

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_value_network(
    env,
    agent: Algorithm,
    number_eval_episodes: int,
    record=None,
    total_steps: int = 0,
):
    state = env.reset()

    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps + 1, frame)

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1

            action = agent.select_action_from_policy(state, evaluation=True)

            state, reward, done, truncated = env.step(action)
            episode_reward += reward

            if eval_episode_counter == 0 and record is not None:
                frame = env.grab_frame()
                record.log_video(frame)

            if done or truncated:
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                    )

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    record.stop_video()


def value_based_train(
    env,
    env_eval,
    agent: Algorithm,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display=False,
):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    number_steps_per_evaluation = train_config.number_steps_per_evaluation

    batch_size = alg_config.batch_size
    G = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated = env.step(action)
        if display:
            env.render()

        memory.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        info = {}

        for _ in range(G):
            info = agent.train_policy(memory, batch_size, total_step_counter)

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")
            evaluate_value_network(
                env_eval,
                agent,
                number_eval_episodes=train_config.number_eval_episodes,
                record=record,
                total_steps=total_step_counter,
            )
            logging.info("--------------------------------------------")

        if done or truncated:
            episode_time = time.time() - episode_start
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                **info,
                display=True,
            )

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
