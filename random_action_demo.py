import copy

import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite

import utils

DOMAIN_NAME = "acrobot"
TASK_NAME = "swingup"

# function for viewing all domain and tasks
utils.all_env()

# Load the environment
random_state = np.random.RandomState(42)
env = suite.load(DOMAIN_NAME, TASK_NAME, task_kwargs={"random": random_state})

# Simulate episode with random actions
duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()


while env.physics.data.time < duration:

    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

    camera0 = env.physics.render(camera_id=0, height=200, width=200)
    camera1 = env.physics.render(camera_id=1, height=200, width=200)
    frames.append(np.hstack((camera0, camera1)))
    rewards.append(time_step.reward)
    observations.append(copy.deepcopy(time_step.observation))
    ticks.append(env.physics.data.time)

    # Plot reward every second
    if (round(env.physics.data.time % 1, 2)) <= 0.01:
        plt.scatter(ticks, rewards)
        plt.pause(0.001)


# Save Video of animation
video_name = DOMAIN_NAME + "_" + TASK_NAME + ".mp4"
utils.display_video(video_name, frames, framerate=1.0 / env.control_timestep())

# Plot reward and observations
num_sensors = len(time_step.observation)

_, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
ax[0].plot(ticks, rewards)
ax[0].set_ylabel("reward")
ax[-1].set_xlabel("time")

for i, key in enumerate(time_step.observation):
    data = np.asarray([observations[j][key] for j in range(len(observations))])
    ax[i + 1].plot(ticks, data, label=key)
    ax[i + 1].set_ylabel(key)

plt.show()
