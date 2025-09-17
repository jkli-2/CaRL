export SAVE_PATH=${WORK_DIR}/save
export SAVE_PNG=1
export RECORD=1
export DEBUG_ENV_AGENT=1
 
python ${WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py --routes ${WORK_DIR}/custom_leaderboard/leaderboard/data/longest6.xml --agent ${WORK_DIR}/team_code/eval_agent.py --resume 1 --checkpoint ${WORK_DIR}/results/CaRL_PY_01/results2.json --track MAP --port 2000 --traffic-manager-port 8000 --agent-config ${WORK_DIR}/results/CaRL_PY_01