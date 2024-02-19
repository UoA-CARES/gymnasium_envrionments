## Research Report

This document serves as a technical documentation and explaination of approaches which may help for future developers.

### Introduction

Training game agents using Reinforcement learning has always been an interesting field of study, whilst there were not many published and successful attempt on Pokemon Red running on GBA enviroment. In this project, we train the agent with NASATD3, a vision based TD3 approach on continuous frames of game states. Initially, the result of training was not ideal and the agent does not seem to improve. Hence, more work has been done, mainly focusing on improving the accuracy on vision based approach, where computer vision techniques such as template matching, abstraction, contours, edge detection ad layer seperation has been tested. Reward functions has also been explored and shaped such that it encourages the agent to explore the map. As a result, the model seem to have improved after the changes compare to the beginning, making the first step, whilst there is much more to do to address the complex game system of pokemon in the future.

### Related works

There was a well made project on successfully training pokemon red on youtube, and this is the link to the github repository: https://github.com/PWhiddy/PokemonRedExperiments.

During the exploration of vision based approach, https://github.com/google-deepmind/pysc2?tab=readme-ov-file was also a inspiration.

## Method

Simply put, the `pokemon_enviroment.py` acts as the main enviroment to train Pokemon, where the reward and game states information was handled. The `image_wrapper` is used for extracting game frames, stack (3) frames together and pass into NASaTD3. This is where CV preprocessing takes place. Other files, such as `enviroment_factory`, `gym_enviroment` is made to assit the construction of overall trainig enviroment. The actual loops, exploration and evaluation are done in `train.py` and `policy_loop.py`

Here, we focus mainly on the first two scripts mentioned. 

### States and Reward

States, in the context of NAsaTD3, will be autoencoded images, in later stage when we fall back to use TD3, will be the x,y coordinate and map id of the player position. Where other states such as number of pokemon, level etc was not immidiately helpful in the beginning. 

Rewards, are based on the states of the game, tailored to the goal of encourage agent to go out and explore the map. Below are the reward functions and explaination.

The reward functions within the `PokemonEnvironment` are designed to incentivize specific behaviors in an agent navigating a Pokémon Red game environment emulated by Pyboy. These rewards guide the agent towards desirable game outcomes, such as exploring new areas, staying mobile, and engaging with the game's mechanics (like battling, catching Pokémon, or managing resources). Below, we detail the logic, target, and explanation of each specified reward function:

### 1. Stuck Reward (`_stuck_reward`)

**Logic**: This function penalizes the agent for remaining in the same location over multiple steps, indicating it's stuck. If the agent's current location matches its previous location for more than 10 consecutive steps, a penalty of -5 is applied.

**Target**: The goal is to discourage the agent from repetitive non-productive behaviors, such as running into walls or objects that do not lead to progress within the game.

**Explanation**: By monitoring the `location` field in the game state, the function counts how many consecutive times the agent has not moved. If this count exceeds a threshold (10 steps), it's assumed the agent is stuck, triggering a penalty to encourage exploration and movement.

### 2. Location Reward (`_location_reward`)

**Logic**: This function rewards the agent for exploring new locations by checking if the current `map_id` has been seen before. If the `map_id` is not in the `seen_locations` set, the agent receives a reward of 50, and the `map_id` is added to the set.

**Target**: The aim is to encourage the agent to explore as much of the game world as possible, discovering new areas and thereby engaging with a wider range of the game's content.

**Explanation**: Exploration is a key aspect of Pokémon games, with new locations often bringing opportunities for battles, items, and new Pokémon encounters. This reward function directly incentivizes broad exploration.

### 3. Distance Travelled Reward (`_distance_travelled_reward`)

**Logic**: This function calculates the Euclidean distance the agent has moved from its initial position within the same map and rewards movement. If the agent moves to a new map, the initial position is reset. The reward is equal to the distance moved, encouraging the agent to roam widely within each map.

**Target**: To incentivize the agent to explore within maps thoroughly, not just to move between them.

**Explanation**: Continuous exploration within each map can uncover hidden items, encounters, and areas crucial for game progression. The function uses the Euclidean distance as a straightforward metric for movement, with larger movements rewarded more.

### 4. Grass Reward ( `_grass_reward`)

**Logic and Target**: A grass reward function would incentivize the agent for engaging with grassy areas where Pokémon encounters are likely. This could involve a simple check to see if the agent is on a tile classified as 'grass' and rewarding accordingly, encouraging the agent to seek out and engage in Pokémon battles.

**Explanation**: Grass tiles in Pokémon games are where wild Pokémon encounters occur. Engaging in battles is essential for leveling up the player's Pokémon, catching new ones, and progressing in the game. A reward for moving onto grass tiles would promote these interactions.

### 5. Outside Reward (`_outside_reward`)

**Logic**: The function differentiates between being indoors and outdoors based on the `map_id`. It increments counters for being outside or inside consecutively. Rewards or penalties are given when the agent has been in one state for more than 10 consecutive steps, encouraging a balance between exploring indoor locations (like gyms, Pokécenters) and the outdoor world.

**Target**: To ensure the agent does not overly prefer one environment over the other, balancing exploration of both indoor and outdoor areas (currently, punish indoor).

**Explanation**: This function aims to keep the agent engaging with outdoor by rewarding sustained periods in outdoor environment.

## CV based approach

Initially as reward function does not seem to be working. An assumption on NaSaTD3 might not be the best solution was made along with attempts to modify the image frames used for training. 

Edge Detection

We applied edge detection to the game frames to highlight boundaries, obstacles, and important features within the game environment. By splitting the frame into its RGB components and applying the Canny edge detector to each, we synthesized a composite edge map that emphasizes edges from all channels. This process aids the agent in recognizing paths, barriers, and other significant elements that influence navigation and strategy.

Contour is also used alongside edge detection, and in fact the result seems better than template matching. 

```python

def apply_edge_detection(self, frame):
    b, g, r = cv2.split(frame)
    edges_b = cv2.Canny(b, 200, 600)
    edges_g = cv2.Canny(g, 200, 600)
    edges_r = cv2.Canny(r, 200, 600)
    edges_rgb = cv2.merge([edges_b, edges_g, edges_r])
    return edges_rgb
```

Template Matching for Object Detection

To detect specific objects (e.g., player character, enemies, items, obstacles) within the game frames, we employed template matching. This method involves sliding template images over the frame to find matches based on a similarity measure. By doing so, we could abstract the game frame into a simplified representation where each detected object is marked with a colored rectangle, transforming the complex visual scene into a manageable set of categorical and positional data.

I also thought about getting obstacle distance information through calculation, but had given up half way.

## Experiment

Both TD3 and NaSATD3 approach were carried out, in which TD3 was much more efficient and shows better learning. 

The reward function of distance was a game changer, that encouraged the agent to walk out of the house and walk towards the edges of the map. However, I observed that, when going through the doors, map id changes in the first frame, but the position remains relative to the previous map, making the reward calculation when going through the doors inaccurate and overly rewarded. Hence, the variable `buffer_frame` was used to pause reward for a few frames when map id changes, leavig time for the actual position to catch up.  

The walkable matrix is also crucial, although not yet proven by the time the report was written, to the state of the game and for agent to understand the surroundings.

## Conclusion

After careful design of reward functions, testing, debugging and modify them to suit the game enviroment, the agent now has learnt to walk out of the house and is likely to walk for a longer distance, for both TD3 and NaSATD3 approach. Comparing to the initial state of not moving at all, it has been a impressive improvement. However, much more work has to be done for the agent to learn to play the game. Thoughtful design of rewards, and exploring more state information, as well and high performance machine is needed.  
