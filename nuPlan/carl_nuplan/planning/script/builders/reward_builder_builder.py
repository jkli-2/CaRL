import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import validate_type
from omegaconf import DictConfig

from carl_nuplan.planning.gym.environment.reward_builder.abstract_reward_builder import (
    AbstractRewardBuilder,
)

logger = logging.getLogger(__name__)


def build_reward_builder(cfg: DictConfig) -> AbstractRewardBuilder:
    """
    Builds reward manager.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of reward manager.
    """
    logger.info("Building AbstractRewardManager...")
    reward_builder = instantiate(cfg.reward_builder)
    validate_type(reward_builder, AbstractRewardBuilder)
    logger.info("Building AbstractRewardManager...DONE!")
    return reward_builder
