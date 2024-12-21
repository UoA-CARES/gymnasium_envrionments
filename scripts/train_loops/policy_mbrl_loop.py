import logging
import time
import torch
import numpy as np


from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0, normalisation=True
):
    state = env.reset()

    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps + 1, frame)

    number_eval_episodes = int(config.number_eval_episodes)

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            normalised_action = agent.select_action_from_policy(state, evaluation=True)
            denormalised_action = normalised_action

            next_state, reward, done, truncated = env.step(denormalised_action)

            episode_reward += reward
            state = next_state

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
    env_eval,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display=False,
    normalisation=True,
):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    batch_size = alg_config.batch_size
    G = alg_config.G
    G_model = alg_config.G_model
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    reward_model_error_t = 0
    world_model_error_t = 0

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )

            denormalised_action = env.sample_action()
            normalised_action = denormalised_action
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            # algorithm range [-1, 1]
            normalised_action = agent.select_action_from_policy(
                state, noise_scale=noise_scale
            )
            denormalised_action = normalised_action

        next_state, reward_extrinsic, done, truncated = env.step(denormalised_action)

        # if len(memory) > max_steps_exploration:
        #     # Converting to tensor
        #     tensor_action = torch.FloatTensor(normalised_action).to(agent.device).unsqueeze(dim=0)
        #     tensor_state = torch.FloatTensor(state).to(agent.device).unsqueeze(dim=0)
        #     pred_ns, _, _, _ = agent.world_model.pred_next_states(observation=tensor_state,
        #                                                           actions=tensor_action)
        #     pred_reward, _ = agent.world_model.pred_rewards(tensor_state, tensor_action, pred_ns)
        #
        #     # MSE Reward
        #     pred_reward = pred_reward.detach().squeeze().cpu().numpy()
        #     l1_one_rwd_error = abs(pred_reward - reward_extrinsic)
        #     reward_model_error_t += l1_one_rwd_error
        #
        #     # MSE. L1 of dynamics
        #     np_pred_ns = pred_ns.detach().squeeze().cpu().numpy()
        #     one_step_mse = (np.square(np_pred_ns - next_state)).mean()
        #     world_model_error_t += one_step_mse


        if display:
            env.render()

        total_reward = reward_extrinsic

        memory.add(
            state,
            normalised_action,
            total_reward,
            next_state,
            done,
        )

        state = next_state
        episode_reward += reward_extrinsic  # Note we only track the extrinsic reward for the episode for proper comparison

        info = {}
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
            # logging.info(f"Training World Model Error {world_model_error_t}, Reward Error {reward_model_error_t}")
            # reward_model_error_t = 0
            # world_model_error_t = 0

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
            # MBRL: Update the statistics.
            if len(memory) > 0:
                statistics = memory.get_statistics()
                agent.set_statistics(statistics)

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
