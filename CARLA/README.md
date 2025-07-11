# CaRL - CARLA

This folder contains the code to train and evaluate RL agents with the CARLA leaderboard 2.0.
In general, we recommend reading the Appendix of the paper if you want to use the code, since it explains many technical details, necessary to understand the code.

### Structure:
* [custom_leaderboard](custom_leaderboard) contains a modified version of the CARLA leaderboard 2.0 that is much faster for RL training than the original. It is used to train models.
* [original_leaderboard](original_leaderboard) contains the original version of the CARLA leaderboard 2.0. It is used for evaluating models.
* [results](results) contains model files
* [team_code](team_code) contains the training and evaluation files.
* [tools](tools) contains various scripts for generating training routes or aggregating results.


## Contents

1. [Setup](#setup)
2. [Pre-Trained Models](#pre-trained-models)
3. [Local Debugging](#local-debugging)
4. [Benchmarking](#benchmarking)
6. [Training](#training)
7. [C++ code](#CPP-Training-and-Evaluation-Code)


## Setup

Clone the repo, setup CARLA 0.9.15, and build the conda environment:
```Shell
git clone https://github.com/autonomousvision/CaRL.git
cd CaRL
chmod +x setup_carla.sh
./setup_carla.sh
conda env create --file=environment.yml
conda activate carl
```

Before running the code, you will need to add the following paths to PYTHONPATH on your system:
```Shell
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/CaRL/CARLA
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```
You can add this in your shell scripts or directly integrate it into your favorite IDE. \
E.g. in PyCharm: Settings -> Project -> Python Interpreter -> Show all -> garage (need to add from existing conda environment first) -> Show Interpreter Paths -> add all the absolute paths above (without pythonpath).

## Pre-Trained Models
We provide a set of [pretrained model weights](results).
For CaRL we provide 2 seeds for a python pytorch model and 1 seed for a [C++ libtorch](#CPP-Training-and-Evaluation-Code) model.
For Roach we provide 5 python seeds. The last number indicates the seed.
All models are licensed under the same license as the code.

Each folder has an `config.json` containing all hyperparameters which will automatically be loaded and override default hyperparameters. The folder also contains a `model_final.pth` which is the model from the last PPO iteration, and an `optimizer_final.pth` which are the adam parameters from the last iteration.

TODO Think2Drive and PlanT

## Local Debugging

To debug evaluation or training, you need to start a CARLA server (or multiple):
```Shell
cd /path/to/CARLA/root
./CarlaUE4.sh
```
CaRL does not use sensor data, so if you do not want to use the spectator camera of CARLA you can start CARLA in CPU mode to save compute using the `-nullrhi` option.
Additionally, you might want to set the `-carla-rpc-port=2000`, `-nosound` `-carla-streaming-port=6000` options in particular if you run multiple servers or non-default ports.

### Evaluation Debugging
To evaluate a model, run [leaderboard_evaluator.py](original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py) as the main python file.

Set the `--agent-config` option to a folder containing a `config.json` and `model_final.pth` files, [e.g. CaRL](results/CaRL_PY_00). <br>
Set the `--agent` to [eval_agent.py](team_code/eval_agent.py). <br>
The `--routes` option should be set to a route file, for example [debug.xml](custom_leaderboard/leaderboard/data/debug.xml). <br>
Set `--checkpoint ` to `/path/to/results/result.json`


The inference model code can be configured using the following environment variables.
The default values are set for CaRL, to evaluate Roach set `SAMPLE_TYPE=roach`. The CPP and singularity values are only needed when running a C++ model.
```Shell
export DEBUG_ENV_AGENT=0 # Produce debug outputs
export SAVE_PATH=/path/to/save/dir # Folder to save debug output in
export RECORD=0 # Record infraction clips
export SAVE_PNG=0 # Save higher quality individual debug frames in PNG. Otherwise video is saved. 
export UPSCALE_FACTOR=1  # Render higher resolution debug for paper
export SCENARIO_RUNNER_ROOT=/path/to/original_leaderboard/scenario_runner # Set to the scenario runner root. Important to set otherwise scenarios don't work.

export HIGH_FREQ_INFERENCE=0 # Can run model at 20 Hz, didn't see any benefit
export NO_CARS=0 # Removes all other cars for debugging, do not use during evaluation.
export SAMPLE_TYPE=mean # How to make action deterministic. Options: mean, sample, roach. Use mean for CaRL and roach for roach
export ROUTES=debug  # Name for visualization folder
export CPP=0  # Set to 1 when evaluating a C++ model.
export CPP_PORT=5555 # Port over which to do communication with C++ code
export PPO_CPP_INSTALL_PATH=/path/to/build_folder # Path to folder containing c++ binary (ppo_carla_inference)
export PATH_TO_SINGULARITY=/path/to/ppo_cpp.sif # Path to singularity container file ppo_cpp.sif
export PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch # Path to pytorch cache
```

### Training Debugging
To train a model you need to start two python processes one for the leaderboard and one for the training code, they communicate via message passing.
CARLA leaderboard:
```Shell
python ${WORK_DIR}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${WORK_DIR}/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios/route_Town03_00.xml.gz --agent ${WORK_DIR}/team_code/env_agent.py --resume 0 --checkpoint ${WORK_DIR}/results/debug_00.json --track MAP --port 2000 --traffic-manager-port 8000 --agent-config /home/jaeger/ordnung/internal/ad_planning/2_carla/results/PPO_debug --gym_port 5555 --debug 0 --repetitions 100 --frame_rate 10.0 --no_rendering_mode False --timeout 900 --skip_next_route False --runtime_timeout 900
```
training script dd_ppo.py:
The training script is highly configurable with many parameters. You can find the documentation at the start of [dd_ppo.py](team_code/dd_ppo.py). The parameters here do not correspond to any particular model and are just mean to get the code running for debugging. TODO change default parameters to CaRL.
```Shell
torch.distributed.run --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 ${WORK_DIR}/team_code/dd_ppo.py --num_envs_per_gpu 1 --use_dd_ppo_preempt False --exp_name DD_PPO_debug --tcp_store_port 7000 --logdir ${WORK_DIR}/results/ --total_batch_size 512 --total_minibatch_size 128 --update_epochs 3 --total_timesteps 10000000 --reward_type simple_reward --debug 1 --debug_type save --ports 5555
```

## Benchmarking

CaRL is currently trained with the 6 scenarios for the longest6 v2 benchmark.
To evaluate CaRL efficiently we parallelize evaluation with multiple GPUs using the [evaluate_routes_slurm.py](evaluate_routes_slurm.py) script. It is build for a SLURM cluster with many cheap consumer GPUs and started using the [run_evaluation_slurm.sh](run_evaluation_slurm.sh). If your cluster has a different structure or job scheduler you can use this script to write your own. In particular for clusters with few expensive GPUs it can be beneficial to evaluate multiple models per GPU at the same time instead.
Since CARLA can be run in CPU mode and CaRL is quite a fast model, longest6 v2 evaluates much faster than is typical with sensorimotor agents.

### Longest6 v2
Longest6 is a benchmark consisting of 36 medium length routes (~1-2 km) from leaderboard 1.0 in towns 1-6. We have adapted the benchmark to the new CARLA version 0.9.15 and leaderboard/scenario runner code. The benchmark features the 7 scenario types from leaderboard 1.0 (now 6 scenarios implemented with the leaderboard 2.0 logic). The scenario descriptions were created by converting the leaderboard 1.0 scenarios with the CARLA route bridge converter. It can serve as a benchmark with intermediate difficulty. Note that the results of models on Longest6 v2 are not directly comparable to the leaderboard 1.0 longest6 numbers. The benchmark can be found [here](custom_leaderboard/leaderboard/data/longest6.xml) and the individual route files [here](custom_leaderboard/leaderboard/data/longest6_split). Unlike the leaderboard 1.0 version, there are no modifications to the CARLA leaderboard code. Longest6 is a training benchmark, so training on Town 01-06 is allowed.

### Result aggregation
To aggregate the results of a parallel evaluation into a single csv we provide the [result_parser.py](tools/result_parser.py) script.

```Shell
python ${WORK_DIR}/tools/result_parser.py --results /path/to/folder/containing_json_files --xml ${WORK_DIR}/custom_leaderboard/leaderboard/data/longest6.xml
```




## Viewing training logs
Start tensorboard with tensorboard --logdir /home/jaeger/ordnung/internal/CaRL/CARLA/results --load_fast=false



## CPP Training and Evaluation Code



