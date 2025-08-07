import logging
import time

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import PPOConfig, TrainingConfig


def evaluate_ppo_network(
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
            action = agent.select_action_from_policy(state)
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


def ppo_train(env,
              agent,
              memory,
              record,
              train_config: TrainingConfig,
              alg_config: PPOConfig,
              ):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_per_batch = alg_config.max_steps_per_batch
    number_steps_per_evaluation = train_config.number_steps_per_evaluation

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0

    evaluate = False

    # instant
    crucial_episode_num = 0
    crucial_total_reward = 0
    current_seed = 0
    crucial_actions = []
    crucial_steps = False
    save_episode = False
    max_reward = float('-inf')
    RN = 5
    episode_actions = []

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if crucial_episode_num > 0 and crucial_steps:

            action = crucial_actions[episode_timesteps - 1]
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

            if episode_timesteps >= len(crucial_actions):
                crucial_steps = False
                print(
                    f"Reach end of crucial path for {RN-crucial_episode_num} time")

        else:

            action = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value)

        next_state, reward, done, truncated = env.step(action_env)
        memory.short_term_memory.add(
            state,
            action,
            reward,
            next_state,
            done,
            episode_num,
            episode_timesteps
        )
        episode_actions.append(action)
        state = next_state
        episode_reward += reward

        if (total_step_counter + 1) % max_steps_per_batch == 0:
            agent.train_policy(memory)

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
            if crucial_episode_num == 1:
                crucial_steps = False
                crucial_episode_num -= 1
                # explore = True
                # print(f"Explore time!")
                # for i in range (len(cs)):
                #    print(f"state:{s[i]}, seed:{cs[i]}")
                # input()
                print(
                    f"crucial_total_reward:{crucial_total_reward},episode_time_step:{episode_timesteps},experience reward:{reward}, episode_reward:{episode_reward}")
                # input()

            elif crucial_episode_num > 1:

                crucial_episode_num -= 1
                crucial_steps = True
                print(f"reward:{reward}, episode_reward:{episode_reward}")

             # 1. when try to find episode with higher episode reward
            elif (crucial_episode_num == 0 and episode_reward > max_reward):
                print(
                    f"Findddddddd higher reward in episode num:{episode_num}, episode reward:{episode_reward}, current_max_reward:{max_reward}")
                max_reward = episode_reward
                # print(f"total_reward:{total_reward}, min_reward:{memory.long_term_memory.get_min_reward()}")
                # states, actions, rewards, next_states, dones, episode_nums, episode_steps = memory.short_term_memory.sample_complete_episode(
                #     episode_num, episode_timesteps)
                crucial_actions = episode_actions
                crucial_episode_num = RN
                #crucial_steps = True

            if evaluate:
                logging.info("*************--Evaluation Loop--*************")
                evaluate_ppo_network(
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
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()
            episode_actions = []

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))
