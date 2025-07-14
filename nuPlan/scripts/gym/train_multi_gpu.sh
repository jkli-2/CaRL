export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR="localhost"

# Basic training parameters
TIMESTEPS=10000000
BATCH_SIZE=4096
MINIBATCH_SIZE=1024
NUM_ENVS_PER_GPU=32
# LOAD_FILE="/path/to/model_latest_XXXXXXX.pth"
LOAD_FILE=null

# Overwrites of hydra config
ROUTE_COMPLETION_TYPE="human" # "human", "mean"
COLLISION_TYPE="non_stationary" # "all", "non_stationary", "at_fault"
COMFORT_TYPE="kinematics" # "action_delta", "kinematics", "kinematics_legacy"
ACCUMULATION="nuplan" # "nuplan", "regular", "survival"
AGENT_TYPE="tracks"  # tracks", "idm_agents", "mixed", "no_tracks"
LANE_DISTANCE_TYPE="v1"
OFF_ROUTE_TYPE="v2"

torchrun \
--start-method spawn \
--nproc_per_node=$WORLD_SIZE \
--nnodes=1 \
--max_restarts=0 \
--rdzv-backend=c10d \
$CARL_DEVKIT_ROOT/carl_nuplan/planning/script/run_gym.py \
+py_func=train \
cpu_collect=False \
debug=False \
experiment_name="gym_train" \
job_name="gym_train_multi_gpu" \
load_file=$LOAD_FILE \
reward_builder.config.route_completion_type=$ROUTE_COMPLETION_TYPE \
reward_builder.config.collision_type=$COLLISION_TYPE \
reward_builder.config.comfort_type=$COMFORT_TYPE \
reward_builder.config.reward_accumulation=$ACCUMULATION \
reward_builder.config.lane_distance_type=$LANE_DISTANCE_TYPE \
reward_builder.config.off_route_type=$OFF_ROUTE_TYPE \
simulation_builder.agent_type=$AGENT_TYPE \
total_batch_size=$BATCH_SIZE \
total_minibatch_size=$MINIBATCH_SIZE \
total_timesteps=$TIMESTEPS \
num_envs_per_gpu=$NUM_ENVS_PER_GPU \
cache.cache_path="/mnt/nvme/caches/dev_v5" \
hydra.searchpath="[pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.gym, pkg://carl_nuplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
