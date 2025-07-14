python $CARL_DEVKIT_ROOT/carl_nuplan/planning/script/run_gym.py \
+py_func=cache \
experiment_name=gym_train \
job_name=cache_dev_v5 \
scenario_builder=nuplan \
scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval" \
scenario_filter=dev_v5 \
scenario_filter.log_names="\${splitter.log_splits.train}" \
cache.cache_path="$NUPLAN_EXP_ROOT/dev_v5" \
hydra.searchpath="[pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.gym, pkg://carl_nuplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
