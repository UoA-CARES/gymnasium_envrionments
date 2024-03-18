import logging
import time
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0
):
    """
    This function evaluate the agent and world model at a fixed interval
    (10000 for now).
    Cumulative rewards are averaged across 10 episodes.

    Mean Square Error for world model is used to evaluate a world model.

    """
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
            next_state, reward, done, truncated = env.step(action_env)
            state = next_state
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
    """
    This function train the agent and world model. It is the major training
    loop. It calls the evaluation function every fixed interval.

    Parameters:
        env -- environment
        agent -- agent to be trained.
        memory -- a list storing all history
        record -- record a evaluation video
        train_config -- training configurations defines interval.
        alg_config -- agent settings from papers and specify hyper-parameters.

    """
    start_time = time.time()
    # Train config
    batch_size = train_config.batch_size
    G = train_config.G
    G_model = train_config.G_model
    max_steps_training = train_config.max_steps_training
    max_steps_exploration = train_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation

    number_steps_per_train_policy = train_config.number_steps_per_train_policy
    # Algorthm specific attributes - e.g. NaSA-TD3 dd
    intrinsic_on = (
        bool(alg_config.intrinsic_on) if hasattr(alg_config, "intrinsic_on") else False
    )

    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    # Algorithm config
    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} "
        f"Evaluation {number_steps_per_evaluation}"
    )

    # Variables for the training loop.
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    evaluate = False

    state = env.reset()
    episode_start = time.time()
    # Using the maximum training steps as the looper.
    # In some work, number of episodes was used.
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1
        # Explore for a certain of period at the begining
        if total_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )
            # action range the env uses [e.g. -2 , 2 for pendulum]
            action_env = env.sample_action()
            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            action = hlp.normalize(
                action_env, env.max_action_value, env.min_action_value
            )
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)
            # algorithm range [-1, 1]
            action = agent.select_action_from_policy(state)
            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )
        # Actual executing of the action.
        next_state, reward_extrinsic, done, truncated = env.step(action_env)

        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(state, action, next_state)
        total_reward = reward_extrinsic + intrinsic_reward

        # Add the transition to the memory.
        memory.add(state, action, total_reward, next_state, done)
        state = next_state
        # Note we only track the extrinsic reward for the episode for proper comparison
        episode_reward += reward_extrinsic

        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            # MBRL: First time to compute the statisics.
            if total_step_counter == max_steps_exploration:
                statistics = memory.get_statistics()
                agent.set_statistics(statistics)
            # MBRL: Sample and train with different functions.
            for _ in range(G_model):
                experience = memory.sample_next(batch_size)
                agent.train_world_model(experience)

            # General training
            for _ in range(G):
                experience = memory.sample(batch_size)
                agent.train_policy(experience)

        # Decide whether to do the evaluation at this time step.
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

            # Do evaluation only when a episode is finished.
            if evaluate:
                logging.info("*************--Evaluation Loop--*************")
                evaluate_policy_network(
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
