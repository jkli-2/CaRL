export CUDA_HOME=/usr/local/cuda
export WORK_DIR=/data/junkali/CaRL/CARLA
export CARLA_ROOT=/data/junkali/carla

export SAVE_PATH=${WORK_DIR}/save_bench2drive
export SAVE_PNG=1
export RECORD=1
export DEBUG_ENV_AGENT=1

export ORIGINAL_LEADERBOARD_DIR=${WORK_DIR}/original_leaderboard
export CUSTOM_LEADERBOARD_DIR=${WORK_DIR}/custom_leaderboard

export LEADERBOARD_ROOT=${ORIGINAL_LEADERBOARD_DIR}/leaderboard
export SCENARIO_RUNNER_ROOT=${ORIGINAL_LEADERBOARD_DIR}/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

training_routes_folder_name=1km_12scen_01
policy_model_name=CaRL_10M_01
 
python ${ORIGINAL_LEADERBOARD_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${CUSTOM_LEADERBOARD_DIR}/leaderboard/data/debug.xml --agent ${WORK_DIR}/team_code/eval_agent.py --resume 1 --checkpoint ${WORK_DIR}/results/${policy_model_name}/config.json --track MAP --port 2000 --traffic-manager-port 8000 --agent-config ${WORK_DIR}/results/${policy_model_name}