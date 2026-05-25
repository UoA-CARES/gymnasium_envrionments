# Gymnasium Environments for CARES RL

This repository provides training orchestration, experiment execution, environment wrappers, plotting utilities, and batch evaluation pipelines for reinforcement learning (RL) experiments using the CARES RL framework.

Repository:

https://github.com/UoA-CARES/gymnasium_envrionments

The framework supports:

- OpenAI Gymnasium environments
- DeepMind Control Suite environments
- TD3 and SAC training
- multi-seed experiments
- automated batch execution
- plotting and evaluation
- custom network architecture experiments
- fractional activation experiments

This repository works together with the CARES RL repository:

https://github.com/UoA-CARES/cares_reinforcement_learning

---

# Related CARES RL Fractional Activation Branch

The fractional activation implementations used in this repository are provided in the CARES RL repository branch:

Repository:

```text
cares_reinforcement_learning
```

Branch:

```text
feature/fractional-swish-gelu
```

That branch extends the CARES RL framework with:

- fractional-order activations
- Grünwald-Letnikov-inspired activations
- residual fractional GELU activations
- smooth fractional Swish variants
- RL-safe fractional nonlinearities
- configurable actor/critic activation placement

The `main` branch of `cares_reinforcement_learning` does not contain these fractional activation implementations.

To run fractional activation experiments correctly, the following setup should be used:

| Repository | Branch |
|---|---|
| `cares_reinforcement_learning` | `feature/fractional-swish-gelu` |
| `gymnasium_envrionments` | current working branch for experiment orchestration |

---

# Repository Structure

```text
gymnasium_envrionments/
│
├── scripts/
│   ├── batch_coordinator.py
│   ├── execution_coordinator.py
│   └── ...
│
├── plotting/
│
├── configs/
│
├── run.py
├── plotter.py
└── README.md
```

---

# Main Components

| File / Folder | Purpose |
|---|---|
| `run.py` | Main training entry point |
| `batch_coordinator.py` | Batch experiment generation and orchestration |
| `execution_coordinator.py` | Experiment execution management |
| `plotter.py` | Plotting and evaluation utilities |
| `configs/` | Experiment configuration files |
| `plotting/` | Plotting helpers and visualisation tools |

---

# Supported Algorithms

- TD3
- SAC

Additional algorithms may be available depending on the CARES RL installation.

---

# Supported Environments

## OpenAI Gymnasium

Examples:

- HalfCheetah-v4
- Hopper-v4
- Ant-v4
- Humanoid-v4

Example command:

```bash
python run.py train cli --gym openai --task HalfCheetah-v4 TD3
```

---

## DeepMind Control Suite

Examples:

- cheetah-run
- walker-walk
- cartpole-swingup
- finger-spin

Example command:

```bash
python3 run.py train cli --gym dmcs --domain cheetah --task run TD3
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/UoA-CARES/gymnasium_envrionments.git
```

Clone CARES RL repository:

```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
```

Checkout the fractional activation branch:

```bash
cd cares_reinforcement_learning
git checkout feature/fractional-swish-gelu
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Optional PyBoy environments:

https://github.com/UoA-CARES/pyboy_environment

---

# Fractional Activation Experiments

Fractional activation experiments are configured using environment variables.

| Variable | Description |
|---|---|
| `ACTIVATION` | Activation class name |
| `ALGORITHM` | TD3 or SAC |
| `LAYERS` | Number of hidden layers |
| `PLACEMENT` | Activation placement strategy |
| `ALPHAS` | Fractional-order sweep values |

---

# Supported Fractional Activations

Implemented in:

```text
cares_reinforcement_learning/networks/fractional_activations.py
```

Supported activations include:

- FractionalSwish
- FractionalSwishBeta
- FALU
- FractionalGELU
- SafeFractionalSwish
- SafeFALU
- SafeFractionalGELU
- SafeGLFractionalGELU
- ResidualFractionalGELU
- AdaptiveResidualFractionalGELU

---

# Placement Strategies

## 1-Layer Networks

Only:

```text
PLACEMENT=all_both
```

is supported.

---

## 2-Layer Networks

| Placement | Description |
|---|---|
| `all_both` | Fractional activation in all actor and critic hidden layers |
| `all_actor` | Fractional activation only in actor hidden layers |
| `all_critic` | Fractional activation only in critic hidden layers |
| `first_both` | Fractional activation only in first actor and critic hidden layers |
| `first_actor` | Fractional activation only in first actor hidden layer |
| `first_critic` | Fractional activation only in first critic hidden layer |

---

# Fractional Alpha Sweeps

Fractional activations support alpha sweeps.

Example:

```bash
ALPHAS=0.1,0.2,0.3,0.4,0.5
```

Interpretation:

| Alpha | Behaviour |
|---|---|
| `0.1` | Strong nonlinear fractional behaviour |
| `0.3` | Moderate fractional modulation |
| `0.5` | Smoother weaker fractional effects |
| `1.0` | Minimal fractional scaling |

---

# Example Fractional Commands

## Standard GELU Baseline

```bash
ACTIVATION=GELU \
ALGORITHM=TD3 \
LAYERS=1 \
PLACEMENT=all_both \
python3 run.py train cli \
--gym openai \
--task HalfCheetah-v4 \
--batch 1 \
TD3 \
--seeds 10 20 30 40 50 \
--max_workers 5
```

---

## Residual Fractional GELU Sweep

```bash
ACTIVATION=ResidualFractionalGELU \
ALPHAS=0.1,0.2,0.3,0.4,0.5 \
ALGORITHM=TD3 \
LAYERS=1 \
PLACEMENT=all_both \
python3 run.py train cli \
--gym openai \
--task HalfCheetah-v4 \
--batch 1 \
TD3 \
--seeds 10 20 30 40 50 \
--max_workers 5
```

---

## Adaptive Residual Fractional GELU

```bash
ACTIVATION=AdaptiveResidualFractionalGELU \
ALPHAS=0.1,0.2,0.3 \
ALGORITHM=TD3 \
LAYERS=2 \
PLACEMENT=first_actor \
python3 run.py train cli \
--gym openai \
--task Hopper-v4 \
--batch 1 \
TD3 \
--seeds 10 20 30 40 50 \
--max_workers 5
```

---

# Batch Mode

Batch execution is controlled through:

```text
scripts/batch_coordinator.py
```

Example:

```bash
python run.py train cli --gym openai --task HalfCheetah-v4 TD3 --batch 1
```

Parallel workers:

```bash
--max_workers N
```

---

# Plotting

Single experiment:

```bash
python3 plotter.py \
-s ~/cares_rl_logs \
-d <TRAINING_PATH>
```

Compare multiple experiments:

```bash
python3 plotter.py \
-s ~/cares_rl_logs \
-d <RUN_A> <RUN_B>
```

---

# Training Outputs

Training outputs are saved to:

```text
~/cares_rl_logs
```

or overridden using:

```bash
CARES_LOG_BASE_DIR
```

Saved outputs include:

```text
logs/
├── configs
├── csv
├── figures
├── trained_models
├── checkpoints
└── videos
```

---

# Docker Usage

Docker image used for fractional activation experiments:

```bash
docker run -it --gpus all hoda511/fractional-rl:new-fractional_swish_gelu
```

Open another shell inside container:

```bash
docker exec -it <container_name> bash
```

Copy files from container to host machine:

```bash
docker cp <container_name>:<path_inside_container> <host_path>
```

Example:

```bash
docker cp fractional_container:/root/cares_rl_logs ~/Desktop/cares_rl_logs
```

---

# Notes

- Activation names must exactly match class names in `fractional_activations.py`
- Activations are dynamically loaded through `common.py`
- Fractional activations can be independently applied to actor and critic networks
- Alpha sweeps enable systematic fractional-order sensitivity analysis
- The framework supports easy extension with additional activations and environments

---

# Research Motivation

This project investigates whether fractional-order nonlinear transformations can improve:

- RL sample efficiency
- optimisation stability
- gradient propagation
- representation smoothness
- actor-critic learning dynamics
- continuous-control RL performance

compared to standard activations including:

- ReLU
- GELU
- Swish / SiLU

with a focus on modern off-policy actor-critic algorithms such as TD3 and SAC.
