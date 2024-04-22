import logging
import time
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_mbrl_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0
):
    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps + 1, frame)

    number_eval_episodes = int(config.number_eval_episodes)

    state = env.reset()

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

            state, reward, done, truncated = env.step(action_env)
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


def policy_based_mbrl_train(
    env,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    # Algorthm specific attributes - e.g. NaSA-TD3 dd
    intrinsic_on = (
        bool(alg_config.intrinsic_on) if hasattr(alg_config, "intrinsic_on") else False
    )

    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    batch_size = alg_config.batch_size
    G = alg_config.G
    G_model = alg_config.G_model

    # Variables for the training loop.
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    evaluate = False

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )

            action_env = env.sample_action()

            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            action = hlp.normalize(
                action_env, env.max_action_value, env.min_action_value
            )
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            # algorithm range [-1, 1]
            action = agent.select_action_from_policy(state, noise_scale=noise_scale)
            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

        next_state, reward_extrinsic, done, truncated = env.step(action_env)

        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(state, action, next_state)

        total_reward = reward_extrinsic + intrinsic_reward

        memory.add(
            state,
            action,
            total_reward,
            next_state,
            done,
        )

        state = next_state
        episode_reward += reward_extrinsic  # Note we only track the extrinsic reward for the episode for proper comparison

        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            # MBRL: First time to compute the statisics.
            if total_step_counter == max_steps_exploration:
                statistics = memory.get_statistics()
                agent.set_statistics(statistics)
            # MBRL: Sample and train with different functions.
            # If G_model < 1, means it will skip a few training of world model.
            if G_model < 1.0:
                interval = int(1 / G_model)
                if total_step_counter % interval == 0:
                    agent.train_world_model(memory, batch_size)
            else:
                for _ in range(int(G_model)):
                    agent.train_world_model(memory, batch_size)

            # General training
            for _ in range(G):
                agent.train_policy(memory, batch_size)

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            evaluate = True

        if done or truncated:
            episode_time = time.time() - episode_start
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                display=True,
            )

            if evaluate:
                logging.info("*************--Evaluation Loop--*************")
                evaluate_mbrl_policy_network(
                    env,
                    agent,
                    train_config,
                    record=record,
                    total_steps=total_step_counter,
                )
                logging.info("--------------------------------------------")
                evaluate = False

            # Reset environment
            state = env.reset()
            # MBRL: Update the statistics.
            if len(memory) > 0:
                statistics = memory.get_statistics()
                agent.set_statistics(statistics)
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
