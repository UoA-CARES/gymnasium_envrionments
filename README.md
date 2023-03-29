# dm_control_suite_setup

### Quick setup for deepmind control suite

```
pip3 install -r requirements.txt
sudo apt-get install libglfw3 libglew2.1
```

### To run test code with random action:
```
python3 random_action_example.py
```
DOMAIN_NAME, TASK_NAME can be set to a range of avaliable options. All basic options can be view by running the utils.all_env() function.

A reward graph is currently updated and displayed every second. Training video and other result plots can be saved after training finishes.

### To run viewer with random action policy:
```
python3 viewer_example.py
```
DM control suite have a built-in viewer function(viewer_launch), providing visualization during training. The function can run a policy and display the trianing process. See https://github.com/deepmind/dm_control/tree/main/dm_control/viewer for more information.


### To train a env policy:

TODO instructions to be added here