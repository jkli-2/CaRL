SPLIT=val14_split
CHECKPOINT="$CARL_DEVKIT_ROOT/checkpoints"
CHECKPOINT_NAME=nuplan_51892_1B  # nuplan_51479_1B, nuplan_51892_1B, simple_52971_500M, simple_52972_500M, simple_vm_500M

for CHALLENGE in closed_loop_nonreactive_agents_action closed_loop_reactive_agents_action; do
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval" \
    planner=ppo_planner \
    planner.ppo_planner.checkpoint_path="$CHECKPOINT/$CHECKPOINT_NAME/model_best.pth" \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    callback="[simulation_log_callback]" \
    main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, metric_summary_callback, csv_main_callback]" \
    hydra.searchpath="[pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://carl_nuplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    group="$NUPLAN_EXP_ROOT/$CHECKPOINT_NAME"
done