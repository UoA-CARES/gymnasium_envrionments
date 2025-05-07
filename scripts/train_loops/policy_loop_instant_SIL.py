import logging
import time
import random
import numpy as np
import math
from cares_reinforcement_learning.memory.sil import SelfImitation


from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env, agent, config: TrainingConfig, record=None, total_steps=0
):
    # if record is not None:
    #     frame = env.grab_frame()
    #     record.start_video(total_steps + 1, frame)

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

            # if eval_episode_counter == 0 and record is not None:
            #     frame = env.grab_frame()
            #     record.log_video(frame)

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

    #record.stop_video()

def policy_based_train(
    env,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
):
    print("Policy Based Training")
    start_time = time.time()
    explore = False
    crucial_steps = False

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
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    crucial_episode_num = 0
    crucial_total_reward = 0
    current_seed = 0
    crucial_actions = []
    crucial_states = []
    save_episode = False
    max_reward = 0
    evaluate = False
    RN = 5
    #RF = 10000
    
    #SIL
    sil = SelfImitation()  # Using the default values
    sil_initiated = False
    
    state = env.reset()
    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1
        
        if total_step_counter < max_steps_exploration or explore:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )
            
            action_env = env.sample_action()
            action = hlp.normalize(
                action_env, env.max_action_value, env.min_action_value
            )
        elif crucial_episode_num > 0 and crucial_steps:
            
            action = crucial_actions[episode_timesteps - 1]  
            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )
            
         
            if episode_timesteps  >= len(crucial_actions):
                crucial_steps = False
                print(f"Reach end of crucial path for {RN-crucial_episode_num} time")
            
            
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
                     
        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            for _ in range(G):
                info = agent.train_policy(memory, batch_size)
                if not sil_initiated:
                        sil = info["sil_buffer"]
                        sil_initiated = True
            sil.step(state, action, reward_extrinsic, next_state, done) 
            mean_adv, num_valid_samples = sil.train()  

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            evaluate = True
        
        if done or truncated:
            print(f"episode_timesteps:{episode_timesteps}, len:{len(crucial_actions)},number_of_crusial_episodes:{crucial_episode_num}")

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
                crucial_episode_num-= 1  
                #explore = True
                #print(f"Explore time!")
                # for i in range (len(cs)):
                #    print(f"state:{s[i]}, seed:{cs[i]}")
                #input()
                print(f"crucial_total_reward:{crucial_total_reward},episode_time_step:{episode_timesteps},experience reward:{total_reward}, episode_reward:{episode_reward}")
                #input()
                
            elif crucial_episode_num > 1 :
                
                crucial_episode_num -= 1  
                crucial_steps = True
                print(f"total_reward:{total_reward}, episode_reward:{episode_reward}")
            
             # 1. when try to find episode with higher episode reward
            elif ( crucial_episode_num == 0 and episode_reward > 0 and episode_reward > max_reward  ) :
                    print(f"Findddddddd higher reward in episode num:{episode_num}, episode reward:{episode_reward}, current_max_reward:{max_reward}")
                    max_reward = episode_reward
                    # print(f"total_reward:{total_reward}, min_reward:{memory.long_term_memory.get_min_reward()}")
                    states, actions,rewards, next_states, dones, episode_nums, episode_steps =memory.short_term_memory.sample_complete_episode(episode_num,episode_timesteps)
                    crucial_actions = actions
                    crucial_episode_num = RN
                    
            
                
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
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time() 

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
