# docker build -t oculux314/cares:base . (use --no-cache to rebuild from start)
# docker run -it --gpus all oculux314/cares:base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV MUJOCO_GL=osmesa
ENV CARES_LOG_BASE_DIR=/app/cares_rl_logs
WORKDIR /app

# -------------------------------------------------------------------
# Installation
# -------------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    python-is-python3 \
    python3-venv \
    python3-pip \
    git \
    # This is needed for mujoco
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    libosmesa6-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    nano

# -------------------------------------------------------------------
# Clone repos
# -------------------------------------------------------------------

# gymnasium_envrionments - training engine and core environments
RUN git clone https://github.com/UoA-CARES/gymnasium_envrionments.git

# cares_reinforcement_learning - RL algorithms
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git

# -------------------------------------------------------------------
# Setup cares_reinforcement_learning
# -------------------------------------------------------------------

WORKDIR /app/cares_reinforcement_learning
RUN pip install -r requirements.txt
RUN pip install -e .

# -------------------------------------------------------------------
# Setup gymnasium_envrionments
# -------------------------------------------------------------------

WORKDIR /app/gymnasium_envrionments
RUN pip install -r requirements.txt

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

ENV CARES_LOG_PATH_TEMPLATE="{algorithm}/{run_name}{algorithm}-{date}"
WORKDIR /app/gymnasium_envrionments/scripts
CMD ["bash", "-c", "echo '======================================================================\nRun `python run.py train cli --gym openai --task HalfCheetah-v4 SAC` to start a training run.\n======================================================================' && \
    bash"]
