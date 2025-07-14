export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

export MASTER_ADDR="localhost"
export MASTER_PORT="4242"

TIMESTEPS=10000000
BATCH_SIZE=2048
MINIBATCH_SIZE=512
NUM_ENVS_PER_GPU=32

CUDA_VISIBLE_DEVICES=0 python $CARL_DEVKIT_ROOT/carl_nuplan/planning/script/run_gym.py \
+py_func=train \
debug=False \
experiment_name="gym_train" \
job_name="gym_train_single_gpu" \
observation_builder=default_observation_builder \
total_batch_size=$BATCH_SIZE \
total_minibatch_size=$MINIBATCH_SIZE \
total_timesteps=$TIMESTEPS \
num_envs_per_gpu=$NUM_ENVS_PER_GPU \
cache.cache_path="/mnt/nvme/caches/dev_v5" \
hydra.searchpath="[pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.gym, pkg://carl_nuplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
