import logging
import time
import random

import numpy as np

from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def _safe_count(mem_obj):
    """
    Best-effort way to estimate how many items exist in a memory object.
    If your memory class has a reliable method, replace this with that.
    """
    try:
        return len(mem_obj)
    except Exception:
        pass

    for attr in ["size", "count", "num_items", "length"]:
        try:
            if hasattr(mem_obj, attr):
                v = getattr(mem_obj, attr)
                return int(v() if callable(v) else v)
        except Exception:
            pass

    # Unknown, assume "enough"
    return 10**9


def build_mixed_schedule(RN: int, avail_long: int, avail_total: int):
    """
    Build a schedule of length RN containing 'long' and 'total' entries.

    Goals:
      - RN even: exact half-half, if possible
      - RN odd: random remainder assignment
      - Respect availability in each memory, shift deficit to the other
      - Shuffle order to mix within the RN repeats
    """
    RN = int(RN)
    avail_long = max(0, int(avail_long))
    avail_total = max(0, int(avail_total))

    if RN <= 0:
        return []

    if avail_long == 0 and avail_total == 0:
        return []

    if avail_long == 0:
        return ["total"] * min(RN, avail_total)

    if avail_total == 0:
        return ["long"] * min(RN, avail_long)

    base = RN // 2
    rem = RN % 2

    n_long = base
    n_total = base

    if rem == 1:
        if random.random() < 0.5:
            n_long += 1
        else:
            n_total += 1

    # Respect availability, shift missing quota
    if n_long > avail_long:
        deficit = n_long - avail_long
        n_long = avail_long
        n_total += deficit

    if n_total > avail_total:
        deficit = n_total - avail_total
        n_total = avail_total
        n_long += deficit

    n_long = min(n_long, avail_long)
    n_total = min(n_total, avail_total)

    schedule = (["long"] * n_long) + (["total"] * n_total)
    random.shuffle(schedule)
    return schedule


def evaluate_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0
):
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

            if done or truncated:
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                    )

                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


def policy_based_train(
    env,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
):
    start_time = time.time()
    explore = False
    crucial_steps = False

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy
    number_of_crusial_episodes = 0

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
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    crucial_episode_num = 0
    crucial_total_reward = 0
    crucial_actions = []
    crucial_states = []
    s = []
    cs = []
    save_episode = False

    RF = 10000
    RN = 5

    # NEW: schedule that defines which memory to sample from for the next RN repeats
    crucial_schedule = []

    evaluate = False
    state = env.reset()
    episode_start = time.time()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration or explore:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )

            action_env = env.sample_action()
            explore = False

            action = hlp.normalize(
                action_env, env.max_action_value, env.min_action_value
            )

        elif crucial_steps and number_of_crusial_episodes > 0:
            # CHANGED: do not only check long_term_memory. Check both.
            long_empty = memory.long_term_memory.is_empty()
            total_empty = memory.long_term_memory_total.is_empty()

            if long_empty and total_empty:
                crucial_steps = False
                print("Both long term memories are empty")
                continue

            if episode_timesteps == 1:
                # CHANGED: build or consume from a pre-built schedule for RN repeats
                if len(crucial_schedule) == 0:
                    avail_long = 0 if long_empty else _safe_count(memory.long_term_memory)
                    avail_total = 0 if total_empty else _safe_count(memory.long_term_memory_total)
                    crucial_schedule = build_mixed_schedule(RN, avail_long, avail_total)

                    if len(crucial_schedule) == 0:
                        crucial_steps = False
                        number_of_crusial_episodes = 0
                        continue

                src = crucial_schedule.pop()

                if src == "long":
                    (crucial_actions,
                     crucial_states,
                     crucial_episode_num,
                     episode_steps,
                     crucial_rewards,
                     crucial_total_reward) = memory.long_term_memory.get_crucial_path(1)
                else:
                    (crucial_actions,
                     crucial_states,
                     crucial_episode_num,
                     episode_steps,
                     crucial_rewards,
                     crucial_total_reward) = memory.long_term_memory_total.get_crucial_path(1)

            action = crucial_actions[episode_timesteps - 1]
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

            if episode_timesteps >= len(crucial_actions):
                crucial_steps = False
                print(f"Reach end of crucial path for {number_of_crusial_episodes} time")

        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            action = agent.select_action_from_policy(state, noise_scale=noise_scale)

            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

        if episode_timesteps == 1:
            s.append(state[0:2])

        next_state, reward_extrinsic, done, truncated = env.step(action_env)

        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(state, action, next_state)

        total_reward = reward_extrinsic + intrinsic_reward

        memory.short_term_memory.add(
            state,
            action,
            total_reward,
            next_state,
            done,
            episode_num,
            episode_timesteps
        )

        state = next_state
        episode_reward += reward_extrinsic

        if total_step_counter > batch_size and total_reward > 0:
            if (not memory.long_term_memory_total.is_full()) or \
               (memory.long_term_memory_total.is_full()
                and total_reward > memory.long_term_memory_total.get_min_reward()
                and episode_timesteps > 2):

                states, actions, rewards, next_states, dones, episode_nums, episode_steps = \
                    memory.short_term_memory.sample_complete_episode(episode_num, episode_timesteps)

                memory.long_term_memory_total.add(
                    [episode_num, total_reward, states, actions, rewards,
                     next_states, dones, episode_nums, episode_steps]
                )
                save_episode = True

        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            for _ in range(G):
                agent.train_policy(memory, batch_size)

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            evaluate = True

        if (total_step_counter + 1) % RF == 0 and episode_reward > 0:
            number_of_crusial_episodes = RN + 1

            # NEW: build schedule ONCE per trigger
            long_empty = memory.long_term_memory.is_empty()
            total_empty = memory.long_term_memory_total.is_empty()

            avail_long = 0 if long_empty else _safe_count(memory.long_term_memory)
            avail_total = 0 if total_empty else _safe_count(memory.long_term_memory_total)

            crucial_schedule = build_mixed_schedule(RN, avail_long, avail_total)

            if len(crucial_schedule) == 0:
                number_of_crusial_episodes = 0
                crucial_steps = False

        if done or truncated:
            print(f"episode_timesteps:{episode_timesteps}, len:{len(crucial_actions)},number_of_crusial_episodes:{number_of_crusial_episodes}")

            episode_time = time.time() - episode_start
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                display=True,
            )

            if number_of_crusial_episodes == 1:
                crucial_steps = False
                number_of_crusial_episodes -= 1
                print(f"crucial_total_reward:{crucial_total_reward},episode_time_step:{episode_timesteps},experience reward:{total_reward}, episode_reward:{episode_reward}")

            elif number_of_crusial_episodes > 1:
                number_of_crusial_episodes -= 1
                crucial_steps = True
                print(f"total_reward:{total_reward}, episode_reward:{episode_reward}")

            elif (not memory.long_term_memory.is_full() and not crucial_steps and episode_reward > 0) or \
                 (not crucial_steps and memory.long_term_memory.is_full()
                  and episode_reward > memory.long_term_memory.get_min_reward()
                  and episode_timesteps > 2):

                print(f"addddddddddddddddddd episode num:{episode_num}, episode reward:{episode_reward}, min_reward:{memory.long_term_memory.get_min_reward()}")

                states, actions, rewards, next_states, dones, episode_nums, episode_steps = \
                    memory.short_term_memory.sample_complete_episode(episode_num, episode_timesteps)

                memory.long_term_memory.add(
                    [episode_num, episode_reward, states, actions, rewards,
                     next_states, dones, episode_nums, episode_steps]
                )

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

            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

