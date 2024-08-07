# gymnasium_environment
We have created a standardised general purpose gym that wraps the most common simulated environments used in reinforcement learning into a single easy to use place. This package serves as an example of how to develop and setup new environments - perticularly for the robotic environments. This package utilises the algorithms implemented in the repository https://github.com/UoA-CARES/cares_reinforcement_learning/ - consult that repository for algorithm implementations. 

## Installation Instructions
If you want to utilise the GPU with Pytorch install CUDA first - https://developer.nvidia.com/cuda-toolkit

Install Pytorch following the instructions here - https://pytorch.org/get-started/locally/

Follow the instructions at https://github.com/UoA-CARES/cares_reinforcement_learning/ to first install the CARES RL dependency.

`git clone` this repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

Install the environments dependent on pyboy here https://github.com/UoA-CARES/pyboy_environment

# Usage
This package is a basic example of running the CARES RL algorithms on OpenAI/DMCS. 

`train.py` takes in hyperparameters that allow you to customise the training gym enviromment – see options below - or RL algorithm. Use `python3 train.py -h` for help on what parameters are available for customisation.

## Gym Environments
This package contains wrappers for the following gym environments:

#### Deep Mind Control Suite
The standard Deep Mind Control suite: https://github.com/google-deepmind/dm_control

```
python3 train.py run --gym dmcs --domain ball_in_cup --task catch TD3
```

<p align="center">
    <img src="./media/dmcs.png" style="width: 80%;"/>
</p>

#### OpenAI Gymnasium
The standard OpenAI Gymnasium: https://github.com/Farama-Foundation/Gymnasium 

```
python train.py run --gym openai --task CartPole-v1 DQN

python train.py run --gym openai --task HalfCheetah-v4 TD3
```

<p align="center">
    <img src="./media/openai.jpg" style="width: 80%;" />
</p>

#### Game Boy Emulator
Environment running Gameboy games utilising the pyboy wrapper: https://github.com/UoA-CARES/pyboy_environment 

```
python3 train.py run --gym pyboy --task mario NaSATD3
```

<p align="center">
    <img src="./media/mario.png" style="width: 40%;" />
    <img src="./media/pokemon.png" style="width: 40%;"/>
</p>

# Data Outputs
All data from a training run is saved into '~/cares_rl_logs'. A folder will be created for each training run named as 'ALGORITHM/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS', e.g. 'TD3-HalfCheetah-v4-23_10_11_08:47:22'. This folder will contain the following directories and information saved during the training session:

```
├─ALGORITHM/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS/
├─ SEED
|  ├─ env_config.py
|  ├─ alg_config.py
|  ├─ train_config.py
|  ├─ data
|  |  ├─ train.csv
|  |  ├─ eval.csv
|  ├─ figures
|  |  ├─ eval.png
|  |  ├─ train.png
|  ├─ models
|  |  ├─ model.pht
|  |  ├─ CHECKPOINT_N.pht
|  |  ├─ ...
|  ├─ videos
|  |  ├─ STEP.mp4
|  |  ├─ ...
├─ SEED...
├─ ...
```

# Plotting
The plotting utility in https://github.com/UoA-CARES/cares_reinforcement_learning/ will plot the data contained in the training data. An example of how to plot the data from one or multiple training sessions together is shown below. Running 'python3 plotter.py -h' will provide details on the plotting parameters.

```
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```
