# !/bin/bash
cmd="python run.py train config --data_path ../../rl_corrective_gym/rl_corrective_gym/space_configs"
env_file="../../rl_corrective_gym/rl_corrective_gym/space_configs/env_config.json"
file "$env_file"

# change the env_config
# {0,1}
jq -c '.action_config = 2 | .dyn_rew = 0 | .effort_rew = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

# {1,0}
jq -c '.action_config = 0 | .dyn_rew = 1 | .effort_rew = 0' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 2' $env_file > tmp.json && mv tmp.json $env_file
$cmd


# {1,1}
jq -c '.action_config = 0  | .effort_rew = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 2' $env_file > tmp.json && mv tmp.json $env_file
$cmd

# {2,0}
jq -c '.action_config = 0 | .dyn_rew = 2 | .effort_rew = 0' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 2' $env_file > tmp.json && mv tmp.json $env_file
$cmd

# {2,1}
jq -c '.action_config = 0 | .effort_rew = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 1' $env_file > tmp.json && mv tmp.json $env_file
$cmd

jq -c '.action_config = 2' $env_file > tmp.json && mv tmp.json $env_file
$cmd

