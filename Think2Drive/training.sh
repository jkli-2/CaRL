export CARLA_PATH=PATH_TO_CARLA

cd dreamerv3
python3 dreamer/main.py \
    --configs carla \
    --seed 0 \
    --logdir ../logdir/Think2drive_pretraining_{timestamp} \
    --env.carla.carla_installation_path ${CARLA_PATH} \
    --env.carla.pretraining True # \
    # --run.from_checkpoint logdir/think2drive_posttrain_seed_0_20250304T150744/checkpoint_1536848.0.ckpt