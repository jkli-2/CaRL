export CARLA_PATH=PATH_TO_CARLA

cd dreamerv3
python3 dreamer/main.py  \
    --configs carla \
    --run.script eval_only_carla_parallel \
    --env.carla.carla_installation_path ${CARLA_PATH} \
    --run.num_envs 3 \
    --run.actor_batch 1 \
    --run.actor_threads 3 \
    --env.carla.eval_times 3 \
    --batch_length 3 \
    --batch_length_eval 3 \
    --seed 0 \
    --logdir ../logdir/Think2drive_evaluation_{timestamp} \
    --run.from_checkpoint logdir/Think2drive_posttraining_20250804T101657/checkpoint_1626704.0.ckpt \
    --env.carla.results_directory Think2Drive_evaluation_seed_0