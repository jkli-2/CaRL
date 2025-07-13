import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from carl_nuplan.planning.gym.environment.trajectory_builder.abstract_trajectory_builder import (
    AbstractTrajectoryBuilder,
)

logger = logging.getLogger(__name__)


def build_trajectory_builder(cfg: DictConfig) -> AbstractTrajectoryBuilder:
    """
    Builds trajectory builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of trajectory builder.
    """
    logger.info("Building AbstractTrajectoryBuilder...")
    trajectory_builder = instantiate(cfg.trajectory_builder)
    validate_type(trajectory_builder, AbstractTrajectoryBuilder)
    logger.info("Building AbstractTrajectoryBuilder...DONE!")
    return trajectory_builder
