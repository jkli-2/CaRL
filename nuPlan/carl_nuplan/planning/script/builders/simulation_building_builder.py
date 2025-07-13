import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from carl_nuplan.planning.gym.environment.simulation_builder.abstract_simulation_builder import (
    AbstractSimulationBuilder,
)

logger = logging.getLogger(__name__)


def build_simulation_builder(cfg: DictConfig) -> AbstractSimulationBuilder:
    """
    Builds simulation builder.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of a simulation builder.
    """
    logger.info("Building AbstractSimulationBuilder...")
    simulation_builder = instantiate(cfg.simulation_builder)
    validate_type(simulation_builder, AbstractSimulationBuilder)
    logger.info("Building AbstractSimulationBuilder...DONE!")
    return simulation_builder
