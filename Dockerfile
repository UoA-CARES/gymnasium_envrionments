# docker build -t oculux314/cares:base .
# docker run -it --gpus all oculux314/cares:base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV MUJOCO_GL=osmesa
ENV CARES_LOG_BASE_DIR=/app/cares_rl_logs
WORKDIR /app

# -------------------------------------------------------------------
# Installation
# -------------------------------------------------------------------

RUN apt-get update
RUN apt-get install -y python-is-python3 python3-venv python3-pip git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

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
RUN git checkout nwil508
RUN git pull
RUN pip install -r requirements.txt
RUN pip install -e .

# -------------------------------------------------------------------
# Setup gymnasium_envrionments
# -------------------------------------------------------------------

WORKDIR /app/gymnasium_envrionments
RUN git checkout nwil508
RUN git pull
RUN pip install -r requirements.txt

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

WORKDIR /app/gymnasium_envrionments/scripts
CMD [ "bash" ]
