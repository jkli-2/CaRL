
# Results

## Performance with non-reactive traffic on Val14 (nuPlan)

## Performance with reactive traffic on Val14 (nuPlan)


# Install

## Code 
First, you need to download the [`nuplan-devkit`](https://github.com/motional/nuplan-devkit), create the `nuplan` conda environment, and install the devkit as editable pip package. For instructions, please follow the [nuPlan documentation](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) (Option B).

Next, navigate into the `nuplan` folder of the CaRL repository and install the code in the nuplan conda environment (also as editable pip package), with the following commands:
```bash
cd /path/to/carl-repo/nuplan
conda activate nuplan
pip install -e .
```
NOTE: We use torch version `2.6.0` (instead the nuPlan default `0.0.0`) in CaRL. Moreover, we install `gym` and further requirements with this command.

## Dataset


## Environment Variables

# Training
We provide training script in `/scripts`


# Evaluation
We evaluate the trained policy with the `PPOPlanner` or `PPOEnsemblePlanner`


