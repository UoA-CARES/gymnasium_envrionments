import logging
import time

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.record import Record
from util.overlay import overlay_info


def evaluate_policy_network(
    env,
    agent: Algorithm,
    number_eval_episodes: int,
    record: Record | None = None,
    total_steps=0,
    normalisation=True,
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

            result = agent.select_action_from_policy(state, evaluation=True)
            normalised_action, *_ = (
                (result,) if not isinstance(result, tuple) else result
            )

            denormalised_action = (
                hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
                if normalisation
                else normalised_action
            )

            state, reward, done, truncated = env.step(denormalised_action)
            episode_reward += reward

            if eval_episode_counter == 0 and record is not None:
                frame = env.grab_frame()
                overlay = overlay_info(
                    frame, reward=f"{episode_reward:.1f}", **env.get_overlay_info()
                )
                record.log_video(overlay)

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


def policy_based_train(
    env,
    env_eval,
    agent: Algorithm,
    memory: MemoryBuffer,
    record: Record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display: bool = False,
    apply_action_normalisation: bool = True,
):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    batch_size = alg_config.batch_size

    # pylint: disable-next=invalid-name
    G = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state = env.reset()

    episode_start = time.time()
    for train_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        # Data from taking actions for memory buffer - e.g. log_prob from PPO
        action_data = []

        if train_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {train_step_counter + 1}/{max_steps_exploration}"
            )

            denormalised_action = env.sample_action()

            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            normalised_action = denormalised_action
            if apply_action_normalisation:
                normalised_action = hlp.normalize(
                    denormalised_action, env.max_action_value, env.min_action_value
                )
        else:
            # algorithm range [-1, 1])
            result = agent.select_action_from_policy(state)
            normalised_action, *action_data = (
                (result,) if not isinstance(result, tuple) else result
            )

            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            denormalised_action = normalised_action
            if apply_action_normalisation:
                denormalised_action = hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )

        next_state, reward_extrinsic, done, truncated = env.step(denormalised_action)

        if display:
            env.render()

        intrinsic_reward = 0
        if train_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(
                state, normalised_action, next_state
            )

        total_reward = reward_extrinsic + intrinsic_reward

        memory.add(
            state,
            normalised_action,
            total_reward,
            next_state,
            done,
            *action_data,
        )

        state = next_state

        # Note we only track the extrinsic reward for the episode for proper comparison
        episode_reward += reward_extrinsic

        info = {}
        if (
            train_step_counter >= max_steps_exploration
            and (train_step_counter + 1) % number_steps_per_train_policy == 0
        ):
            for _ in range(G):
                info = agent.train_policy(memory, batch_size, train_step_counter)

        info["intrinsic_reward"] = intrinsic_reward

        if (train_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")
            evaluate_policy_network(
                env_eval,
                agent,
                number_eval_episodes=train_config.number_eval_episodes,
                record=record,
                total_steps=train_step_counter,
                normalisation=apply_action_normalisation,
            )
            logging.info("--------------------------------------------")

        if done or truncated:
            episode_time = time.time() - episode_start
            record.log_train(
                total_steps=train_step_counter + 1,
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
