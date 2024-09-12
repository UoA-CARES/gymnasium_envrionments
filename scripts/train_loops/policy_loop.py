import copy
import logging
import time

from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0, normalisation=True
):
    # debug-log logging.info("Logging32")
    state = env.reset()

    # debug-log logging.info("Logging33")
    if record is not None:
        # debug-log logging.info("Logging34")
        frame = env.grab_frame()
        record.start_video(total_steps + 1, frame)
        # debug-log logging.info("Logging35")

    # debug-log logging.info("Logging36")
    number_eval_episodes = int(config.number_eval_episodes)
    # debug-log logging.info("Logging37")

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        # debug-log logging.info("Logging38")
        while not done and not truncated:
            # debug-log logging.info("Logging39")
            episode_timesteps += 1
            normalised_action = agent.select_action_from_policy(state, evaluation=True)
            
            denormalised_action = hlp.denormalize(
                normalised_action, env.max_action_value, env.min_action_value
            ) if normalisation else normalised_action

            # debug-log logging.info("Logging41")
            state, reward, done, truncated = env.step(denormalised_action)
            episode_reward += reward

            # debug-log logging.info("Logging42")
            if eval_episode_counter == 0 and record is not None:
                # debug-log logging.info("Logging44")
                frame = env.grab_frame()
                record.log_video(frame)
                # debug-log logging.info("Logging45")

            # debug-log logging.info("Logging43")
            if done or truncated:
                # debug-log logging.info("Logging46")
                if record is not None:
                    # debug-log logging.info("Logging47")
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                    )
                    # debug-log logging.info("Logging48")

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    # debug-log logging.info("Logging49")
    record.stop_video()
    # debug-log logging.info("Logging50")


def policy_based_train(
    env,
    env_eval,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display=False,
    normalisation=True,
):
    # debug-log logging.info("Logging9")
    start_time = time.time()

    # debug-log logging.info("Logging10")
    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    # debug-log logging.info("Logging11")
    # Algorthm specific attributes - e.g. NaSA-TD3 dd
    intrinsic_on = (
        bool(alg_config.intrinsic_on) if hasattr(alg_config, "intrinsic_on") else False
    )

    # debug-log logging.info("Logging12")
    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    batch_size = alg_config.batch_size
    G = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state = env.reset()

    # debug-log logging.info("Logging13")
    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        # debug-log logging.info("Logging14")
        episode_timesteps += 1

        # debug-log logging.info("Logging15")
        if total_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )

            denormalised_action = env.sample_action()

            # debug-log logging.info("Logging16")
            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            if normalisation:
                normalised_action = hlp.normalize(
                    denormalised_action, env.max_action_value, env.min_action_value
                )
            else:
                normalised_action = denormalised_action
            # debug-log logging.info("Logging17")
        else:
            # debug-log logging.info("Logging18")
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            # debug-log logging.info("Logging19")
            # algorithm range [-1, 1]
            normalised_action = agent.select_action_from_policy(
                state, noise_scale=noise_scale
            )
            # debug-log logging.info("Logging20")
            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            if normalisation:
                denormalised_action = hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
            else:
                denormalised_action = normalised_action
            # debug-log logging.info("Logging21")

        # debug-log logging.info("Logging22")
        next_state, reward_extrinsic, done, truncated = env.step(denormalised_action)
        # debug-log logging.info("Logging23")
        if display:
            # debug-log logging.info("Logging128")
            env.render()

        # debug-log logging.info("Logging23")
        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(
                state, normalised_action, next_state
            )

        # debug-log logging.info("Logging24")
        total_reward = reward_extrinsic + intrinsic_reward

        # debug-log logging.info("Logging25")
        memory.add(
            state,
            normalised_action,
            total_reward,
            next_state,
            done,
        )
        # debug-log logging.info("Logging26")

        state = next_state
        episode_reward += reward_extrinsic  # Note we only track the extrinsic reward for the episode for proper comparison
        # debug-log logging.info("Logging27")

        info = {}
        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            # debug-log logging.info("Logging28")
            for _ in range(G):
                # debug-log logging.info("Logging29")
                info = agent.train_policy(memory, batch_size)

        if intrinsic_on:
            info["intrinsic_reward"] = intrinsic_reward

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")
            evaluate_policy_network(
                env_eval,
                agent,
                train_config,
                record=record,
                total_steps=total_step_counter,
                normalisation=normalisation,
            )
            logging.info("--------------------------------------------")

        if done or truncated:
            episode_time = time.time() - episode_start
            # debug-log logging.info("Logging30")
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                info=info,
                display=True,
            )
            # debug-log logging.info("Logging31")

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
