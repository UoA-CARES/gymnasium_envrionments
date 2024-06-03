# Building the Docker Image

Build the docker image locally
```bash
# Linux
docker build -t caresrl/gymnasium_environments .

# Mac
docker build --platform linux/amd64 -t caresrl/gymnasium_environments .
```

# Run the docker image
```bash
# Linux
docker run -it --gpus all -v ./logs:/root/cares_rl_logs caresrl/gymnasium_environments bash 

# Mac
docker run -it --platform linux/amd64 -v ./logs:/root/cares_rl_logs caresrl/gymnasium_environments bash 
```

# Development
Any changes to `Record.py` saving directory will affect where we should mount our data volume.

This pulls main from both `cares_reinforcement_learning` and `pyboy_environments`, and `gymnasium_environments`. Meaning all are git repositories.

# Common Docker commands

Exit and Kill docker container
```
exit
```

Exit without killing docker container
```
# Typed in succession
Ctrl P + Ctrl Q
```

View running containers
```
docker ps
```

Attach to running container
```
docker attach <container-id>
```