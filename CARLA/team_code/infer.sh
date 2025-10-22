export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/usr/local/cuda
export WORK_DIR=/data/junkali/CaRL/CARLA
export CARLA_ROOT=/data/junkali/carla

export PPO_CPP_INSTALL_PATH=/data/junkali/ppo.cpp/build
export PATH_TO_SINGULARITY=/data/junkali/ppo.cpp/tools/ppo_cpp.sif
export PYTORCH_KERNEL_CACHE_PATH=/scratch/junkali/cache/torch
export LD_LIBRARY_PATH=$PPO_CPP_INSTALL_PATH:$LD_LIBRARY_PATH

export ORIGINAL_LEADERBOARD_DIR=${WORK_DIR}/original_leaderboard
export LEADERBOARD_ROOT=${ORIGINAL_LEADERBOARD_DIR}/leaderboard
export SCENARIO_RUNNER_ROOT=${ORIGINAL_LEADERBOARD_DIR}/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

MODEL_CPP=${WORK_DIR}/results/CaRL_10M_01
MODEL_PY=${WORK_DIR}/results/Roach_01

python inference_parallel.py \
  --routes ${WORK_DIR}/custom_leaderboard/leaderboard/data/inference_route60.xml \
  --shards 4 \
  --carla-root ${CARLA_ROOT} \
  --leaderboard-root ${LEADERBOARD_ROOT} \
  --agent ${WORK_DIR}/team_code/eval_agent.py \
  --agent-configs $MODEL_CPP $MODEL_PY \
  --agent-backends cpp py \
  --work-dir ${WORK_DIR}/inference_save_parallel \
  --gpu-ids 0 0 1 1 \
  --start-port 2000 \
  --track MAP \
  --ppo-cpp-build /data/junkali/ppo.cpp/build
