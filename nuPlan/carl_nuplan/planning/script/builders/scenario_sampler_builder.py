import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from carl_nuplan.planning.gym.environment.scenario_sampler.abstract_scenario_sampler import (
    AbstractScenarioSampler,
)

logger = logging.getLogger(__name__)


def build_scenario_sampler(cfg: DictConfig) -> AbstractScenarioSampler:
    """
    Builds scenario sampler.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of scenario sampler.
    """
    logger.info("Building AbstractScenarioSampler...")
    scenario_sampler = instantiate(cfg.scenario_sampler)
    validate_type(scenario_sampler, AbstractScenarioSampler)
    logger.info("Building AbstractScenarioSampler...DONE!")
    return scenario_sampler
