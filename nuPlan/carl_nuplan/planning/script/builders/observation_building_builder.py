import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from carl_nuplan.planning.gym.environment.observation_builder.abstract_observation_builder import (
    AbstractObservationBuilder,
)

logger = logging.getLogger(__name__)


def build_observation_builder(cfg: DictConfig) -> AbstractObservationBuilder:
    """
    Builds observation builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of observation builder.
    """
    logger.info("Building AbstractObservationBuilder...")
    observation_builder = instantiate(cfg.observation_builder)
    validate_type(observation_builder, AbstractObservationBuilder)
    logger.info("Building AbstractObservationBuilder...DONE!")
    return observation_builder
