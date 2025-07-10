# CaRL

eval model:

python /home/jaeger/ordnung/internal/CaRL/CARLA/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py 

environment variables:
DEBUG_ENV_AGENT 1 # Produce debug outputs
HIGH_FREQ_INFERENCE 0 # Can run model at 20 Hz, didn't see any benefit
NO_CARS 0 # Removes all other cars for debugging, do not use during evaluation.
RECORD 0 # Record infraction clips
SAMPLE_TYPE # How to make action deterministic. Options: mean, sample, roach. Use mean for CaRL and roach for roach
ROUTES debug  # Name for visualization folder
SAVE_PATH  # Folder to save debug output in
SAVE_PNG  # Save higher quality individual debug frames in PNG. Otherwise video is saved. 
SCENARIO_RUNNER_ROOT  # Set to the scenario runner root. Important to set otherwise scenarios don't work.
UPSCALE_FACTOR 6  # Render higher resolution debug for paper
CPP 0  # Set to 1 when evaluating a C++ model.
CPP_PORT 5555 # Port over which to do communication with C++ code
PPO_CPP_INSTALL_PATH /path/to/build_folder # Path to folder containing c++ binary (ppo_carla_inference)
PATH_TO_SINGULARITY /path/to/ppo_cpp.sif # Path to singularity container file ppo_cpp.sif
PYTORCH_KERNEL_CACHE_PATH ~/.cache/torch # Path to pytorch cache