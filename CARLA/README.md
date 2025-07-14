# CaRL - CARLA

This folder contains the code to train and evaluate RL agents with the CARLA leaderboard 2.0.
In general, we recommend reading the Appendix of [the paper](https://arxiv.org/abs/2504.17838) if you want to use the code, since it explains many technical details, necessary to understand the code.

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
5. [Training](#training)
6. [C++ code](#CPP-Training-and-Evaluation-Code)


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

## Training

The main training algorithm is in [dd_ppo.py](team_code/dd_ppo.py). It contains a custom [DD_PPO](https://arxiv.org/abs/1911.00357) implementation that is based on the PPO code in [CleanRL](https://github.com/vwxyzjn/cleanrl/tree/master).
We are using a RL optimized leaderboard in this project but the unmodified CARLA 0.9.15 server.
The CARLA server has various problems for RL training such as that it occasionally crashes due to bugs.
For that reason we use the [train_parallel.py](team_code/train_parallel.py) script to start the training, which starts up the CARLA servers, leaderboard clients and training code. The script monitors the training for CARLA crashes and restarts everything if something crashes. Crashes happen typically a couple of times per run on small scale (10M samples, 8 concurrent servers), but constantly (~every 15 min) at large (300 M samples, 128 concurrent servers) runs.
Additionally, for larger runs we also observed CARLA race conditions on shared cluster file systems. 
The most consistent way for us to get the 300M training run to converge was use a VM/independent node with 128 CPU cores and 8 GPUs and 1TB of RAM (training uses quite a lot of RAM to avoid loading routes from disk during training).
A more principled solution would be to fork the simulator and fix the CARLA bugs. We might do this in the future and update here.

We provide 4 training configs to train different RL models.
The training code is highly configurable, you can find the right hyperparameters for each model in these scripts.

* [train_roach.sh](team_code/train_roach.sh) Reproduces the Roach approach for the CARLA leaderboard 2.0.
* [train_carl_tiny_cpp.sh](team_code/train_carl_tiny_cpp.sh) Reproduces the 10M samples ablation in Table 4. Can be a good start to play around with the repo since it only needs 1 GPU and trains within a day. The script uses the C++ training code described later.
* [train_carl_py.sh](team_code/train_carl_py.sh) Trains the CaRL method (300M samples run) using the python training code.
* [train_carl_cpp.sh](team_code/train_carl_cpp.sh) Trains the CaRL method (300M samples run) using the C++ training code.

Every parameter has a one sentence description about what it does in the [argument parser](team_code/dd_ppo.py).
But in general this is a research repo, so we do not have extensive documentation. I recommend to use Ctrl+Shift+f (or your editors equivalent) to find the parameter in the code and just read the code to learn what it does.

The config system work such that every parameter has a default value that is loaded from [rl_config.py](team_code/rl_config.py). Most relevant parameters can be changed by using command line arguments `--parametername value`. The parameter is then automatically overwritten and the new config is saved in a [config.json](results/CaRL_PY_00/config.json) file alongside the model. During inference or restarting of training the values in the config.json file are automatically loaded overwriting the default values. 
The nice thing about this config system is that one can easily add features while ensuring backwards compatibility with models trained with older code versions.
For that I set the default value in the default config such that the new feature is turned off by default.
If an old model is then run with the new code the default value of the parameter is then loaded because the parameter is not in config.json, ensuring that the new feature is turned off and the old model still runs as intended.

### Viewing training logs
The training logs are stored as tensorboard file for both the C++ and Python code. They can be viewed locally by starting tensorboard with `tensorboard --logdir /path/to/model/dir --load_fast=false`.
The python code additionally implements uploading the log files to [Weights and Biases](https://wandb.ai/) by setting the `--track 1` option.

## CPP Training and Evaluation Code
The training evaluation code for the C++ implementation of CaRL is hosted in the [ppo.cpp](https://github.com/autonomousvision/ppo.cpp) repository.
To use it clone the repository and follow the instructions to build the singularity container and compile the binaries.
The code can also be built to run natively on your system, but I recommend using the singularity container option since it is much more convenient and doesn't seem to affect performance.

Evaluating a model works the same way as with the python code, but you additionally have to specify the following environment variables:

```Shell
export PPO_CPP_INSTALL_PATH=/path/to/folder/with/binaries # The C++ binaries you built
export PATH_TO_SINGULARITY=/path/to/ppo_cpp.sif # The singularity container
export PYTORCH_KERNEL_CACHE_PATH=~/.cache/torch  # Path to the PyTorch cache folder, makes it visible inside the container
```

Training a model works similarly to training with the pytorch code and is started using the [train_parallel.py](team_code/train_parallel.py) script. You additionally have to set the following options, for an example see [here](team_code/train_carl_cpp.sh).

```Shell
--train_cpp 1
--PYTORCH_KERNEL_CACHE_PATH /path/to/.cache
--ppo_cpp_install_path /path/to/folder/with/cpp/binaries
--cpp_singularity_file_path /path/to/ppo_cpp.sif 
--cpp_system_lib_path_1 /path/to/missing/system/libs  # In case there is some system library that the code can't find you can link the path here. You will most likely not need it, everything should be included inside the container. In that case set it to a random folder.
```

The C++ training code is written in a way so that model configuration works mostly the same as the python code.
Both codes store and load parameters from a config.json, they start with default parameters loaded from a default config class which can be overwritten using arguments from the command line with the `--parametername value` and the parameters have the same name.

There are a few differences though. The C++ code implements a subset of the options available in the python code.
All features used in CaRL are implemented, but the python code has some additional features that I did not end up using in CaRL which are not implemented in C++.
Also, I have not implemented the `use_exploration_suggest` feature from the Roach baseline in the C++ code. 

Lastly, PPO has a few hyperparameters that are dependent on each other. For example, you either set `--total_minibatch_size` and `--total_batch_size`, or you set `--num_steps` and `--num_minibatches`. Unfortunately, I have changed my mind about which one is more convenient to set and now the python code implements the former and the C++ code implements the latter.
This has no effect on the algorithm, but you need to familiarize yourself with the parameters and set them correctly.
You can compare [train_carl_py.sh](team_code/train_carl_py.sh) with [train_carl_cpp.sh](team_code/train_carl_cpp.sh) to have an example.









