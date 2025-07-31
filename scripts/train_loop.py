import logging
import time

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.record import Record
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from environments.gym_environment import GymEnvironment
from environments.image_wrapper import ImageWrapper
from util.overlay import overlay_info
from util.log_in_place import InPlaceLogger


def evaluate_agent(
    env: GymEnvironment | ImageWrapper,
    agent: Algorithm,
    number_eval_episodes: int,
    record: Record | None = None,
    total_steps: int = 0,
    normalisation: bool = True,
):
    state = env.reset(training=False)

    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps + 1, frame)

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        episode_states = []
        episode_actions = []
        episode_rewards: list[float] = []

        while not done and not truncated:
            episode_timesteps += 1

            normalised_action = agent.select_action_from_policy(state, evaluation=True)

            denormalised_action = (
                hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
                if normalisation
                else normalised_action
            )

            state, reward, done, truncated, step_info = env.step(denormalised_action)
            episode_reward += reward

            # For Bias Calculation
            episode_states.append(state)
            episode_actions.append(normalised_action)
            episode_rewards.append(reward)

            if eval_episode_counter == 0 and record is not None:
                frame = env.grab_frame()
                overlay = overlay_info(
                    frame, reward=f"{episode_reward:.1f}", **env.get_overlay_info()
                )
                record.log_video(overlay)

            if done or truncated:
                # Calculate bias
                info = agent.calculate_bias(
                    episode_states,
                    episode_actions,
                    episode_rewards,
                )

                # incorporate custom data
                info |= step_info

                # Log evaluation information
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                        **info,
                    )

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    record.stop_video()


def train_agent(
    env: GymEnvironment | ImageWrapper,
    env_eval: GymEnvironment | ImageWrapper,
    agent: Algorithm,
    memory: MemoryBuffer,
    record: Record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display: bool = False,
    apply_action_normalisation: bool = True,
):
    logging.setLoggerClass(InPlaceLogger)
    exploration_logger = logging.getLogger("exploration")

    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    # TODO potentially push these into the algorithm itself
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

        if train_step_counter < max_steps_exploration:
            exploration_logger.info(
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
            normalised_action = agent.select_action_from_policy(state)

            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            denormalised_action = normalised_action
            if apply_action_normalisation:
                denormalised_action = hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )

        next_state, reward_extrinsic, done, truncated, step_info = env.step(
            denormalised_action
        )

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
        info |= step_info

        if (train_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")
            evaluate_agent(
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

            agent.epsiode_done()

            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
