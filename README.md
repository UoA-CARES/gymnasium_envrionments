# Fractional Activation Experiments for CARES RL

This branch extends the `gymnasium_envrionments` training framework with fractional and smooth nonlinear activation function experiments for reinforcement learning.

The repository provides training orchestration, environment wrappers, batch execution, and experiment configuration for evaluating custom activation functions in actor-critic reinforcement learning algorithms.

The underlying reinforcement learning algorithms are provided by the CARES RL repository:

https://github.com/UoA-CARES/cares_reinforcement_learning

---

# Implemented Fractional Activations

- FractionalSwish
- FractionalSwishBeta
- FALU
- FractionalGELU
- SafeFractionalSwish
- SafeFALU
- SafeFractionalGELU
- SafeGLFractionalGELU

Custom activations are implemented in:

```text
cares_reinforcement_learning/networks/fractional_activations.py
```

and dynamically loaded through:

```text
cares_reinforcement_learning/networks/common.py
```

---

# Installation

First install the CARES RL dependency:

```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
```

Clone this repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Install requirements:

```bash
pip3 install -r requirements.txt
```

Optional pyboy environments:

https://github.com/UoA-CARES/pyboy_environment

---

# Supported Algorithms

- TD3
- SAC

---

# Supported Environments

## OpenAI Gymnasium

```bash
python run.py train cli --gym openai --task HalfCheetah-v4 TD3
```

## DeepMind Control Suite

```bash
python3 run.py train cli --gym dmcs --domain cheetah --task run TD3
```

---

# Fractional Activation Experiment Configuration

Experiments are configured using environment variables.

| Variable | Description |
|---|---|
| `ACTIVATION` | Fractional activation class name |
| `ALGORITHM` | TD3 or SAC |
| `LAYERS` | 1 or 2 |
| `PLACEMENT` | Activation placement strategy |

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
| `first_both` | Fractional activation only in the first hidden layer of actor and critic |
| `first_actor` | Fractional activation only in the first actor hidden layer |
| `first_critic` | Fractional activation only in the first critic hidden layer |

---

# Example Commands

## TD3 1-Layer Example

```bash
ACTIVATION=FractionalSwish \
ALGORITHM=TD3 \
LAYERS=1 \
PLACEMENT=all_both \
python3 run.py train cli \
--gym openai \
--task HalfCheetah-v4 \
--batch 1 \
TD3 \
--seeds 10 \
--max_workers 1
```

---

## TD3 2-Layer Example

```bash
ACTIVATION=SafeFractionalGELU \
ALGORITHM=TD3 \
LAYERS=2 \
PLACEMENT=all_actor \
python3 run.py train cli \
--gym openai \
--task HalfCheetah-v4 \
--batch 1 \
TD3 \
--seeds 10 \
--max_workers 1
```

---

## SAC Example

```bash
ACTIVATION=FALU \
ALGORITHM=SAC \
LAYERS=2 \
PLACEMENT=first_both \
python3 run.py train cli \
--gym openai \
--task Hopper-v4 \
--batch 1 \
SAC \
--seeds 10 \
--max_workers 1
```

---

# Batch Mode

Batch mode enables automated execution of multiple experiment configurations.

Example:

```bash
python run.py train cli --gym openai --task HalfCheetah-v4 TD3 --batch 1
```

Experiment configurations are defined in:

```text
scripts/batch_coordinator.py
```

Parallel execution across seeds is controlled using:

```bash
--max_workers N
```

---

# Training Outputs

Training outputs are saved to:

```text
~/cares_rl_logs
```

unless overridden using:

```bash
CARES_LOG_BASE_DIR
```

Saved outputs include:

```text
logs/
├── configs
├── csv data
├── figures
├── trained models
├── checkpoints
└── videos
```

---

# Plotting

Plot single experiment:

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

# Docker Usage

Run container:

```bash
docker run -it --gpus all oculux314/cares:base
```

Open another terminal inside the container:

```bash
docker exec -it <container_name> bash
```

Copy files from container:

```bash
docker cp <container_name>:<path> <host-path>
```

---

# Repository Structure

```text
gymnasium_envrionments/
│
├── scripts/
│   ├── batch_coordinator.py
│   └── ...
│
├── cares_reinforcement_learning/
│   ├── networks/
│   │   ├── common.py
│   │   ├── fractional_activations.py
│   │   └── ...
│   └── ...
│
├── run.py
└── README.md
```

---

# Notes

- Activation names must exactly match class names in `fractional_activations.py`
- Custom activations are dynamically loaded through `common.py`
- The framework supports easy extension with additional activation functions

---

# Research Motivation

This project investigates whether fractional and smooth nonlinear transformations can improve optimisation stability, representation learning, and actor-critic training dynamics compared to standard activations such as ReLU, GELU, and Swish.
