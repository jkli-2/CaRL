# <h1 align="center">Think2Drive: Reproduction</h1>

<p align="center">
  <strong>An open reproduction of the Think2Drive world model-based autonomous driving system</strong>
</p>
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#setup">Setup</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#tools">Tools</a> •
  <a href="#citation">Citation</a>
</p>

<h2 id="overview">📋 Overview</h2>

This directory contains an independent reproduction of [Think2Drive](https://arxiv.org/abs/2402.16720v2), a reinforcement learning planner for autonomous driving. Think2Drive leverages the model-based RL algorithm [DreamerV3](https://arxiv.org/abs/2301.04104) to learn driving policies through world model simulation on the [CARLA Leaderboard 2.0](https://leaderboard.carla.org/get_started_v2_0/).

## ⚠️ Disclaimer
Since the original Think2Drive code and models are not publicly available, we provide this reproduction based on our best interpretation of the paper. We used Git commit `251910d` of DreamerV3 (the most recent at reproduction time). From the Think2Drive paper we inferred they used an older commit, but our attempts using their code version failed, hence we adopted a newer code version of similar model size in terms of parameters.

**Important:** Due to a bug in DreamerV3, the model must always be JIT-compiled and executed on GPU. CPU execution is not supported.


<h2 id="setup">🛠️ Setup</h2>

### Prerequisites
1. **CARLA Simulator Setup**
 - Complete this [Setup](../CARLA/README.md) first.

2. **CaRL Conda Environment with Numba** Install Numba in the CaRL conda environment for the BEV renderer:
```Shell
conda activate carl
pip install numba
conda deactivate
```

3. **Think2Drive Environment** Create a dedicated environment for DreamerV3:

```Shell
conda env create -f environment.yml
conda activate think2drive
```

4. **Weights & Biases Setup** We use Weights & Biases for experiment tracking:
```Shell
pip install wandb
wandb login
```

5. **Set CARLA simulator path** Set the `carla_installation_path` and `CARLA_PATH` in the following files:
- `dreamerv3/dreamer/config.yaml`
- `carla_agent/start_carla_leaderboard.sh`
- `evaluation.sh`
- `training.sh`

<h2 id="training">🚀 Training</h2>

Think2Drive employs a two-stage curriculum learning approach. We provide trained checkpoints, allowing you to skip the training phase if desired.

### 📊 Training Overview

| Stage | Duration | Steps | Focus |
| - | - | - | - |
| Pretraining | ~2.5 hours | 400K | Basic driving skills without scenarios |
| Post-training | ~9.5 hours | 1.6M additional | Fine-tuning with scenarios |

On A100 GPU with 16 CPU cores

### Stage 1: Pretraining (400K steps)
Train on easy routes without scenarios to learn basic driving skills:


```Shell
# Configure training.sh for pretraining
#   Ensure --env.carla.pretraining is set to True 
#   Comment out --run.from_checkpoint

bash training.sh
```

**Note:** Training continues beyond 400K steps. This implementation saves the model checkpoints every 15 minutes.

### Stage 2: Post-training (1.6M total steps)
Fine-tune with scenarios and increased planner training ratio:

```Shell
# Configure training.sh for post-training
#   Set --env.carla.pretraining to False
#   Set --run.from_checkpoint to a checkpoint close to 400K steps

# Example configuration:
#    --env.carla.pretraining False \
#    --run.from_checkpoint logdir/Think2drive_pretraining_20250803T184851/checkpoint_403024.0.ckpt

bash training.sh
```

<h2 id="evaluation">📊 Evaluation</h2>

Evaluate the model on the CARLA Leaderboard:

```Shell
# Configure evaluation.sh:
#   Set --run.from_checkpoint to a checkpoint close to 1.6M steps

# Example:
#   --run.from_checkpoint logdir/Think2drive_posttraining_20250804T101657/checkpoint_1626704.0.ckpt

bash evaluation.sh
```

**Evaluation Outputs** Results are stored in `carla_leaderboard_checkpoints/[results_directory]`:
- 📹 Videos: Full episode recordings
- 🖼️ Frames: Individual timestep images
- 📊 Metrics: Performance statistics

To aggregate results into a single CSV file, use the [result_parser.py](../CARLA/tools/result_parser.py) script.

<h2 id="tools">🔧 Additional Tools</h2>

### BEV Renderer Data Precomputation
We precompute Bird's Eye View (BEV) renderer data for improved efficiency and cache it in pickle files.

1. **Start CARLA in a separate terminal:**
```Shell
# Use -RenderOffScreen flag for headless operation
bash CarlaUE4.sh [-RenderOffScreen]
```

2. **Run the precomputation script:**
```Shell
python precompute_bev_renderer_data.py
```

<h2 id="citation">📝 Citation</h2>

If you use this reproduction in your research, please cite both the original Think2Drive paper and CaRL:

### Think2Drive
```BibTeX
@article{li2024think2drive,
  title={Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2)},
  author={Li, Qifeng and Jia, Xiaosong and Wang, Shaobo and Yan, Junchi},
  journal={arXiv preprint arXiv:2402.16720},
  year={2024}
}
```

### CaRL

```BibTeX
@article{Jaeger2025ArXiv, 
    author = {Bernhard Jaeger and Daniel Dauner and Jens Beißwenger and Simon Gerstenecker and Kashyap Chitta and Andreas Geiger}, 
    title = {CaRL: Learning Scalable Planning Policies with Simple Rewards}, 
    year = {2025}, 
    journal = {arXiv.org}, 
    volume = {2504.17838}, 
}
```

If you find this repository useful, please consider giving it a star ⭐!

## Acknowledgements

The original code in this repository was written by **Bernhard Jaeger**, **Daniel Dauner**, **Jens Beißwenger** and **Simon Gerstenecker**. **Kashyap Chitta** and **Andreas Geiger** have contributed as technical advisors.

This work builds on the foundations of many open source projects. We particularly thank:

* [leaderboard](https://github.com/carla-simulator/leaderboard)
* [scenario_runner](https://github.com/carla-simulator/scenario_runner)
* [carla_garage](https://github.com/autonomousvision/carla_garage)
* [plant](https://github.com/autonomousvision/plant)
* [roach](https://github.com/zhejz/carla-roach/)
* [tuplan_garage](https://github.com/autonomousvision/tuplan_garage)
* [clean_rl](https://github.com/vwxyzjn/cleanrl/tree/master)

We also thank the creators of the numerous libraries we use. Complex projects like this would not be feasible without your contributions to the open source community.