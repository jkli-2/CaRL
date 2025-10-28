export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/usr/local/cuda
export WORK_DIR=/data/junkali/CaRL/CARLA
export CARLA_ROOT=/data/junkali/carla

# For CPP
export CPP=0
export PPO_CPP_INSTALL_PATH=/data/junkali/ppo.cpp/build
export PATH_TO_SINGULARITY=/data/junkali/ppo.cpp/tools/ppo_cpp.sif
export PYTORCH_KERNEL_CACHE_PATH=/scratch/junkali/cache/torch
export LD_LIBRARY_PATH=$PPO_CPP_INSTALL_PATH:$LD_LIBRARY_PATH

export ORIGINAL_LEADERBOARD_DIR=${WORK_DIR}/original_leaderboard
export CUSTOM_LEADERBOARD_DIR=${WORK_DIR}/custom_leaderboard

export LEADERBOARD_ROOT=${ORIGINAL_LEADERBOARD_DIR}/leaderboard
export SCENARIO_RUNNER_ROOT=${ORIGINAL_LEADERBOARD_DIR}/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# policy_model_name=CaRL_10M_01
policy_model_name=CaRL_v1_1_PY_01
inference_route=inference_route20
repetition=1

export SAVE_PATH=${WORK_DIR}/inference_save
export SAVE_PNG=1
export RECORD=1
export DEBUG_ENV_AGENT=1
 
python ${ORIGINAL_LEADERBOARD_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${CUSTOM_LEADERBOARD_DIR}/leaderboard/data/${inference_route}.xml --agent ${WORK_DIR}/team_code/eval_agent.py --resume 1 --checkpoint ${WORK_DIR}/results/${policy_model_name}/inference_result_${repetition}.json --track MAP --port 2000 --traffic-manager-port 8000 --agent-config ${WORK_DIR}/results/${policy_model_name}