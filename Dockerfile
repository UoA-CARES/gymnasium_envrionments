
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
RUN git checkout action-info-loggin
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .

WORKDIR /workspace

# Install Pyboy Environments
RUN git clone https://github.com/UoA-CARES/pyboy_environment.git
WORKDIR /workspace/pyboy_environment
RUN git checkout lvl-up-task
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .

WORKDIR /workspace

RUN git clone https://github.com/UoA-CARES/gymnasium_envrionments.git
WORKDIR /workspace/gymnasium_envrionments
RUN git checkout p4p-pokemon-docker
RUN pip3 install -r requirements.txt

# We don't have GUI capabilities
RUN pip3 uninstall opencv-python
RUN pip3 install opencv-python-headless

# Incase someone doesn't mount volume at runtime
VOLUME /root/cares_rl_logs

WORKDIR /workspace/gymnasium_envrionments/scripts