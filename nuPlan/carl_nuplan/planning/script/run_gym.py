import logging
import os

import hydra
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from omegaconf import DictConfig

from carl_nuplan.planning.gym.caching import run_caching
from carl_nuplan.planning.gym.training import run_training

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
# set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/gym")

if os.environ.get("NUPLAN_HYDRA_CONFIG_PATH") is not None:
    CONFIG_PATH = os.path.join("../../../../", CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != "gym":
    CONFIG_PATH = os.path.join(CONFIG_PATH, "gym")
CONFIG_NAME = "default_gym"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for gym experiments.
    :param cfg: omegaconf dictionary
    """

    # Configure logger
    build_logger(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    if cfg.py_func == "train":

        logger.info("Starting training...")
        run_training(cfg=cfg)

    elif cfg.py_func == "cache":

        worker = build_worker(cfg)

        logger.info("Starting caching...")
        run_caching(cfg=cfg, worker=worker)
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
