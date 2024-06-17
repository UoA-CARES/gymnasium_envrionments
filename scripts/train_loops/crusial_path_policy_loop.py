import logging
import time

import numpy as np

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
    start_time = time.time()
    explore = False
    crucial_steps = False

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy
    number_of_crusial_episodes = 0

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
    explore_time = 10000
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

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
            # print(f"action_env:{action_env}")
            # input()

            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            action = hlp.normalize(
                action_env, env.max_action_value, env.min_action_value
            )
        elif crucial_steps and number_of_crusial_episodes > 0:
            
            if episode_timesteps == 1:
                crucial_actions,episode_n, episode_s= memory.long_term_memory.get_crucial_path(1)
                #print(f"actions:{len(crucial_actions)},episode_numbers:{len(episode_n)}, episode_steps:{episode_s}")
                # print(actions[0])
                # input()
            print(f"episode_timesteps:{episode_timesteps}, len:{len(crucial_actions)},number_of_crusial_episodes:{number_of_crusial_episodes}")
                        
             # Check that the index is within bounds
            action_env = crucial_actions[episode_timesteps - 1]
            if episode_timesteps == len(crucial_actions):
                print(f"Reach end of crucial path for {6-number_of_crusial_episodes} time")
                # input()
            if episode_timesteps  >= len(crucial_actions):
               
                crucial_steps = False
                print(f"Reach end of crucial path for {6-number_of_crusial_episodes} time")
                explore = True
            
            
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
        
        if total_step_counter > batch_size:
            #print(f" is full: {memory.long_term_memory.is_full()},total_reward:{total_reward}, min_reward:{memory.long_term_memory.get_min_reward()}, episode_timesteps:{episode_timesteps}")
            if (not memory.long_term_memory.is_full() ) or \
            (memory.long_term_memory.is_full() and total_reward > memory.long_term_memory.get_min_reward() and episode_timesteps > 2):
                   # print(f"total_reward:{total_reward}, min_reward:{memory.long_term_memory.get_min_reward()}")
                    states, actions,rewards, next_states, dones, episode_nums, episode_steps =memory.short_term_memory.sample_complete_episode(episode_num,episode_timesteps)
                   
                    memory.long_term_memory.add([episode_num,total_reward, states, actions,rewards, next_states, dones, episode_nums, episode_steps])
                    
                    
         
        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            for _ in range(G):
                agent.train_policy(memory, batch_size)

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            evaluate = True
        
        if (total_step_counter +1) % explore_time == 0 :#and episode_reward>0:and total_step_counter >10000:
                 number_of_crusial_episodes = 6   
                 #crucial_steps = False
                 #print(f"crucial steps:{crucial_steps}")
              
        if done or truncated:
            #print(f"dones:{done}, truncated:{truncated}")
            #last_episode_experience = [episode_num, total_reward, state, action, reward_extrinsic, episode_num, episode_timesteps]
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
                explore = True
                print(f"Explore time!")
                
            elif number_of_crusial_episodes > 0 :
                
                number_of_crusial_episodes -= 1  
                crucial_steps = True
            
                
           
           
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
