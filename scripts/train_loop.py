import logging
import time

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.record import Record
from cares_reinforcement_learning.util.repetition import EpisodeReplay
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)
from environments.gym_environment import GymEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
from util.log_in_place import InPlaceLogger
from util.overlay import overlay_info


def evaluate_usd(
    env: GymEnvironment | MultiModalWrapper,
    agent: Algorithm,
    record: Record | None = None,
    total_steps: int = 0,
    normalisation: bool = True,
):
    state = env.reset(training=False)

    for skill_counter, skill in enumerate(range(agent.num_skills)):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        agent.set_skill(skill, evaluation=True)

        if record is not None:
            frame = env.grab_frame()
            record.start_video(f"{total_steps+1}-{skill}", frame)

            log_path = record.current_sub_directory
            env.set_log_path(log_path, total_steps + 1)

        while not done and not truncated:
            episode_timesteps += 1

            available_actions = env.get_available_actions()
            action_context = ActionContext(
                state=state, evaluation=True, available_actions=available_actions
            )
            normalised_action = agent.select_action_from_policy(action_context)

            denormalised_action = (
                hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
                if normalisation
                else normalised_action
            )

            state, reward, done, truncated, env_info = env.step(denormalised_action)
            episode_reward += reward

            if record is not None:
                frame = env.grab_frame()
                overlay = overlay_info(
                    frame, reward=f"{episode_reward:.1f}", **env.get_overlay_info()
                )
                record.log_video(overlay)

            if done or truncated:
                # Log evaluation information
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=skill_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                        **env_info,
                    )

                    record.stop_video()

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                agent.episode_done()


def evaluate_agent(
    env: GymEnvironment | MultiModalWrapper,
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

        log_path = record.current_sub_directory
        env.set_log_path(log_path, total_steps + 1)

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

            available_actions = env.get_available_actions()
            action_context = ActionContext(
                state=state, evaluation=True, available_actions=available_actions
            )
            normalised_action = agent.select_action_from_policy(action_context)

            denormalised_action = (
                hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
                if normalisation
                else normalised_action
            )

            state, reward, done, truncated, env_info = env.step(denormalised_action)
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
                bias_data = agent.calculate_bias(
                    episode_states,
                    episode_actions,
                    episode_rewards,
                )

                # Log evaluation information
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                        **env_info,
                        **bias_data,
                    )

                    record.stop_video()

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                agent.episode_done()


def train_agent(
    env: GymEnvironment | MultiModalWrapper,
    env_eval: GymEnvironment | MultiModalWrapper,
    agent: Algorithm,
    memory: MemoryBuffer,
    record: Record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display: bool = False,
    apply_action_normalisation: bool = True,
    start_training_step: int = 0,
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

    use_episode_repetition = alg_config.repetition_num_episodes > 0
    repetition_num_episodes = alg_config.repetition_num_episodes

    repeating = False
    repetition_counter = 0

    repetition_buffer = EpisodeReplay()
    repeat = False

    episode_repetitions = 0

    # Add in metrics to track how often it has repeated episodes

    for train_step_counter in range(start_training_step, int(max_steps_training)):
        episode_timesteps += 1

        info: dict = {}

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
        elif use_episode_repetition and repeat and repetition_buffer.has_best_episode():
            exploration_logger.info(
                f"Repeating Episode {episode_num} Step {episode_timesteps}/{len(repetition_buffer.best_actions)}"
            )
            denormalised_action = repetition_buffer.replay_best_episode(
                episode_timesteps - 1
            )

            if episode_timesteps >= len(repetition_buffer.best_actions):
                repeat = False
        else:
            # algorithm range [-1, 1])
            available_actions = env.get_available_actions()
            action_context = ActionContext(
                state=state, evaluation=False, available_actions=available_actions
            )
            normalised_action = agent.select_action_from_policy(action_context)

            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            denormalised_action = normalised_action
            if apply_action_normalisation:
                denormalised_action = hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )

        repetition_buffer.record_action(denormalised_action)
        info["repeated"] = episode_repetitions

        next_state, reward_extrinsic, done, truncated, env_info = env.step(
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

        if (
            train_step_counter >= max_steps_exploration
            and (train_step_counter + 1) % number_steps_per_train_policy == 0
        ):

            training_context = TrainingContext(
                memory=memory,
                batch_size=batch_size,
                training_step=train_step_counter,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_done=done or truncated,
            )
            train_info = {}
            for _ in range(G):
                train_info = agent.train_policy(training_context)
            info |= train_info

        info["intrinsic_reward"] = intrinsic_reward

        if (train_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")

            if agent.policy_type == "usd":
                evaluate_usd(
                    env_eval,
                    agent,
                    record=record,
                    total_steps=train_step_counter,
                    normalisation=apply_action_normalisation,
                )
            else:
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
                **env_info,
                **info,
                display=True,
            )

            if repeating:
                repeat = True
                repetition_counter += 1
                if repetition_counter >= repetition_num_episodes:
                    repeat = False
                    repeating = False
                    repetition_counter = 0
            elif train_step_counter > max_steps_exploration:
                repeat = repetition_buffer.finish_episode(episode_reward)
                repeating = repeat
                episode_repetitions = (
                    episode_repetitions + 1
                    if repeat and use_episode_repetition
                    else episode_repetitions
                )
            else:
                repetition_buffer.finish_episode(episode_reward)

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1

            agent.episode_done()

            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
