#!/bin/bash

export git_root=$1
export route_file=$2
export logdir=$3
export index=$4
export client_port=$5
export tm_port=$6
export rl_port=$7
export random_seed=$8
export skip_next_route=$9
export repetitions=${10}

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export NO_RENDER_MODE=False

export WORK_DIR=/home/ste/Documents/CaRL/CARLA
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/custom_leaderboard/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/custom_leaderboard/leaderboard
export CARLA_ROOT=/home/ste/Documents/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
export PPO_CPP_DIR=/home/ste/Documents/ppo.cpp

python -u ${git_root}/custom_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${route_file} --agent ${git_root}/team_code/env_agent.py --checkpoint ${logdir}/route_${index}.json --track MAP --port ${client_port} --traffic-manager-port ${tm_port} --agent-config ${logdir} --gym_port ${rl_port} --traffic-manager-seed ${random_seed} --skip_next_route ${skip_next_route} --frame_rate 10 --resume 1 --no_rendering_mode ${NO_RENDER_MODE} --runtime_timeout 120.0 --timeout 800.0  --repetitions ${repetitions}