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
This package is a basic example of running the CARES RL algorithms on OpenAI/DMCS/pyboy. 

## Running Training and Evaluation
The packagage is called using `run.py`. This takes in specific commands list below for training and evaluation purposes.

Use `python3 run.py -h` for help on what parameters are available for customisation.

### Train
The train command in the run.py script is used to initiate the training process for reinforcement learning models within specified gym environments. This command can be customized using various hyperparameters to tailor the training environment and the RL algorithm. You can use python "run.py train cli -h" to view all available options for customization and start a run directly through the terminal. This flexibility enables users to experiment with different settings and optimize their models effectively.

Specific and larger configuration changes can be loaded using python "run.py train config --data_path <PATH_TO_TRAINING_CONFIGS>", allowing for a more structured and repeatable training setup through configuration files. 

```
python run.py train cli -h
python run.py train config --data_path <PATH_TO_TRAINING_CONFIGS>
```

### Evaluate
The evaluate command is used to re-run the evaluation loops on a trained reinforcement learning model within a specified gym environment. By running python run.py evaluate --data_path <PATH_TO_TRAINING_DATA>, users can load the trained model and the corresponding training data to evaluate how well the model performs on the given task. 

```
python run.py evaluate --data_path <PATH_TO_TRAINING_DATA>
```

## Gym Environments
This package contains wrappers for the following gym environments:

#### Deep Mind Control Suite
The standard Deep Mind Control suite: https://github.com/google-deepmind/dm_control

```
python3 run.py train cli --gym dmcs --domain ball_in_cup --task catch TD3
```

<p align="center">
    <img src="./media/dmcs.png" style="width: 80%;"/>
</p>

#### OpenAI Gymnasium
The standard OpenAI Gymnasium: https://github.com/Farama-Foundation/Gymnasium 

```
python run.py train cli --gym openai --task CartPole-v1 DQN

python run.py train cli --gym openai --task HalfCheetah-v4 TD3
```

<p align="center">
    <img src="./media/openai.jpg" style="width: 80%;" />
</p>

#### Game Boy Emulator
Environment running Gameboy games utilising the pyboy wrapper: https://github.com/UoA-CARES/pyboy_environment 

```
python3 run.py train cli --gym pyboy --task mario SACAE
```

<p align="center">
    <img src="./media/mario.png" style="width: 40%;" />
    <img src="./media/pokemon.png" style="width: 40%;"/>
</p>

# Data Outputs
All data from a training run is saved into the directory specified in the `CARES_LOG_BASE_DIR` environment variable. If not specified, this will default to `'~/cares_rl_logs'`.

You may specify a custom log directory format using the `CARES_LOG_PATH_TEMPLATE` environment variable. This path supports variable interpolation such as the algorithm used, seed, date etc. This defaults to `"{algorithm}/{algorithm}-{domain_task}-{date}"`.

This folder will contain the following directories and information saved during the training session:

```text
├─ <log_path>
|  ├─ env_config.json
|  ├─ alg_config.json
|  ├─ train_config.json
|  ├─ *_config.json
|  ├─ ...
|  ├─ SEED_N
|  |  ├─ data
|  |  |  ├─ train.csv
|  |  |  ├─ eval.csv
|  |  ├─ figures
|  |  |  ├─ eval.png
|  |  |  ├─ train.png
|  |  ├─ models
|  |  |  ├─ model.pht
|  |  |  ├─ CHECKPOINT_N.pht
|  |  |  ├─ ...
|  |  ├─ videos
|  |  |  ├─ STEP.mp4
|  |  |  ├─ ...
|  ├─ SEED_N
|  |  ├─ ...
|  ├─ ...
```

# Plotting
The plotting utility in https://github.com/UoA-CARES/cares_reinforcement_learning/ will plot the data contained in the training data based on the format created by the Record class. An example of how to plot the data from one or multiple training sessions together is shown below.

Running 'python3 plotter.py -h' will provide details on the plotting parameters and control arguments. You can custom set the font size and text for the title, and axis labels - defaults will be taken from the data labels in the csv files.

```sh
python3 plotter.py -h
```

Plot the results of a single training instance

```sh
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM/ALGORITHM-TASK-YY_MM_DD:HH:MM:SS
```

Plot and compare the results of two or more training instances

```sh
python3 plotter.py -s ~/cares_rl_logs -d ~/cares_rl_logs/ALGORITHM_A/ALGORITHM_A-TASK-YY_MM_DD:HH:MM:SS ~/cares_rl_logs/ALGORITHM_B/ALGORITHM_B-TASK-YY_MM_DD:HH:MM:SS
```
