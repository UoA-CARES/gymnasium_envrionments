# gymnasium_envrionment

This repository includes a script that allows you to run any OpenAI Gymnasium (https://github.com/Farama-Foundation/Gymnasium) or Deep Mind Control Suite (https://github.com/google-deepmind/dm_control) environment – provided you comply with all the dependencies for that environment. 

This package serves as an example of how to develop and setup new environments - perticularly for the robotic environments. Consult the repository https://github.com/UoA-CARES/cares_reinforcement_learning/ for a guide on how to use the general RL package

## Installation Instructions
If you want to utilise the GPU with Pytorch install CUDA first - https://developer.nvidia.com/cuda-toolkit

Install Pytorch following the instructions here - https://pytorch.org/get-started/locally/

Follow the instructions at https://github.com/UoA-CARES/cares_reinforcement_learning/ to first install the CARES RL dependency.

`git clone` this repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

## Usage
This package is a basic example of running the CARES RL algorithms on OpenAI/DMCS. 

`train.py` takes in hyperparameters that allow you to customise the training run enviromment – OpenAI or DMCS Environment - or RL algorithm. Use `python3 train.py -h` for help on what parameters are available for customisation.

An example is found below for running on the OpenAI, DMCS, and pyboy environments with TD3/NaSATD3 through console
```
python train.py run --gym openai --task CartPole-v1 DQN

python train.py run --gym openai --task HalfCheetah-v4 TD3

python3 train.py run --gym dmcs --domain ball_in_cup --task catch TD3

python3 train.py run --gym pyboy --task pokemon NaSATD3 --image_observation=1
```

An example is found below for running using pre-defined configuration files - note these directories will need to exist on your machine for this to function. Examples of each configuration can be found in the `configs` folder and under https://github.com/UoA-CARES/cares_reinforcement_learning/
```
python example_training_loops.py config --env_config ~/cares_rl_configs/env_dmcs_config.json --training_config ~/cares_rl_configs/training_config.json --algorithm_config ~/cares_rl_configs/algorithm_config.json
```

### Data Outputs
All data from a training run is saved into '~/cares_rl_logs'. A folder will be created for each training run named as 'ALGORITHM-TASK-YY_MM_DD:HH:MM:SS', e.g. 'TD3-HalfCheetah-v4-23_10_11_08:47:22'. This folder will contain the following directories and information saved during the training session:

```
ALGORITHM-TASK-YY_MM_DD:HH:MM:SS/
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

### Plotting
The plotting utility in https://github.com/UoA-CARES/cares_reinforcement_learning/ will plot the data contained in the training data. An example of how to plot the data from one or multiple training sessions together is shown below. Running 'python3 plotter.py -h' will provide details on the plotting parameters.

```
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```