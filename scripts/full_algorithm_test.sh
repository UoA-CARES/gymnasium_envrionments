#!/bin/bash

# Ensure $1 is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <required_arg> <run_name> [optional_arg]"
  exit 1
fi

# Prepare arguments
required_arg="$1"

# Input to feed into the Python script
input_name="$2"  # If $2 is empty, this will default to an empty string

optional_arg=()
if [ $# -ge 3 ]; then
  shift 2  # remove $1 and $2
  # Join all remaining args into a single string, then split by space
  joined="$*"
  IFS=' ' read -r -a optional_arg <<< "$joined"
fi

# List of all domain/task pairs in DMC Suite
declare -A dmcs_tasks

dmcs_tasks=(
  [acrobot]="swingup"
  [ball_in_cup]="catch"
  [cartpole]="swingup"
  [cheetah]="run"
  [finger]="turn_hard"
  [hopper]="hop"
  [humanoid]="run"
  [reacher]="hard"
  [walker]="walk"
)

# Loop over all domain/task pairs
for domain in "${!dmcs_tasks[@]}"; do
  for task in ${dmcs_tasks[$domain]}; do
    echo "Running: domain=$domain, task=$task"
    python3 run.py train cli --gym dmcs --domain "$domain" --task "$task" "$required_arg" --seeds 10 20 30 40 50 "${optional_arg[@]}" <<< "$input_name"
  done
done

