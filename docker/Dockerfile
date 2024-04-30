# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install Cares Reinforcement Learning
RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
WORKDIR /workspace/cares_reinforcement_learning
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .

WORKDIR /workspace

# Install Pyboy Environments
RUN git clone https://github.com/UoA-CARES/pyboy_environment.git
WORKDIR /workspace/pyboy_environment
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .

WORKDIR /workspace

RUN git clone https://github.com/UoA-CARES/gymnasium_envrionments.git
WORKDIR /workspace/gymnasium_environments
RUN pip3 install -r requirements.txt

# We don't have GUI capabilities
RUN pip3 uninstall opencv-python
RUN pip3 install opencv-python-headless

# Incase someone doesn't mount volume at runtime
VOLUME /root/cares_rl_logs

WORKDIR /workspace/gymnasium_environments/scripts
