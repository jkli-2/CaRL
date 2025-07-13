
# Results 🏆

### Performance with non-reactive traffic on Val14 (nuPlan)

### Performance with reactive traffic on Val14 (nuPlan)


# Install 📦

### Code 📄
First, you need to download the [`nuplan-devkit`](https://github.com/motional/nuplan-devkit), create the `nuplan` conda environment, and install the devkit as editable pip package. For instructions, please follow the [nuPlan documentation](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) (Option B).

Next, navigate into the `nuplan` folder of the CaRL repository and install the code in the nuplan conda environment (also as editable pip package), with the following commands:
```bash
cd /path/to/carl-repo/nuplan
conda activate nuplan
pip install -e .
```
> [!NOTE]  
> We use torch version `2.6.0+cu124` (instead the nuPlan default `1.9.0+cu111`) in CaRL. Moreover, we install `gymnasium` and further requirements with this command.

### Dataset 🗃️
> [!IMPORTANT]  
> Before downloading any data, please ensure you have read the [nuPlan license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE).

In order to train and evaluate CaRL on nuPlan, you need to download the nuPlan dataset according to the [official documentation](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html). You can find a bash script for downloading nuPlan in [`/scripts/download/download_nuplan.sh`](https://github.com/autonomousvision/CaRL/nuPlan/scripts/download/download_nuplan.sh) (~2TB). The data needs to be stored in the following format:
```
nuplan
└── dataset
    ├── maps
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north
    │   │   └── ...
    │   ├── us-ma-boston
    │   │   └── ...
    │   ├── us-nv-las-vegas-strip
    │   │   └── ...
    │   └── us-pa-pittsburgh-hazelwood
    │       └── ...
    └── nuplan-v1.1
         ├── splits 
         │     ├── mini 
         │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
         │     │    ├── ...
         │     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
         │     ├── test 
         │     │    ├── 2021.05.25.12.30.39_veh-25_00005_00215.db
         │     │    ├── ...
         │     │    └── 2021.10.06.08.34.20_veh-53_01089_01868.db
         │     └── trainval
         │          ├── 2021.05.12.19.36.12_veh-35_00005_00204.db
         │          ├── ...
         │          └── 2021.10.22.18.45.52_veh-28_01175_01298.db
         └── sensor_blobs (empty)
```

Optionally, if you want to store the complete training dataset, you can download a pre-processed cache we used to train CaRL (see [`/scripts/download/download_cache.sh`](https://github.com/autonomousvision/CaRL/nuPlan/scripts/download/download_nuplan.sh)). The maps are still required for training/evaluation. For evaluation on `val14`, you only need to download the `val` logs. 


### Environment Variables 🌍

Finally, you need to add the following environment variables to your bash scripts or to your `~/.bashrc`:
```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/path/to/nuplan/exp"
export NUPLAN_DEVKIT_ROOT="/path/to/nuplan-devkit/"

export CARL_DEVKIT_ROOT="/path/to/CaRL/nuPlan/"
```


# Training 🏋️
We provide training script in `/scripts/training`.


# Evaluation 🚗
We evaluate the trained policy with the `PPOPlanner` or `PPOEnsemblePlanner`. See `/scripts/simulation` for more information. All checkpoints are provided in the GitHub repository under `/checkpoints`. For example, our best policy can be evaluated with:
```bash
SPLIT=val14_split
CHECKPOINT="$CARL_DEVKIT_ROOT/checkpoints"
CHECKPOINT_NAME=nuplan_51892_1B

for CHALLENGE in closed_loop_nonreactive_agents_action closed_loop_reactive_agents_action; do
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=ppo_planner \
    planner.ppo_planner.checkpoint_path="$CHECKPOINT/$CHECKPOINT_NAME/model_best.pth" \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    callback="[simulation_log_callback]" \
    main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, metric_summary_callback, csv_main_callback]" \
    hydra.searchpath="[pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://carl_nuplan.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    group="$NUPLAN_EXP_ROOT/$CHECKPOINT_NAME"
done
```
Note that this scripts evaluated the reactive and non-reactive simulation of `val14`. You can find the final results in the experiment folder stored in `"$NUPLAN_EXP_ROOT/$CHECKPOINT_NAME"`.

