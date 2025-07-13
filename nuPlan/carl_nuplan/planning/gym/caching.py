import gc
import logging
import os
import uuid
from typing import Dict, List, Union
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from carl_nuplan.planning.gym.cache.gym_scenario_cache import GymScenarioCache

logger = logging.getLogger(__name__)


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> None:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[List[AbstractScenario], DictConfig]]]) -> None:
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

        logger.info(f"Extracted {str(len(scenarios))} scenarios for thread_id={thread_id}, node_id={node_id}.")
        cache = GymScenarioCache(cfg.cache.cache_path, cfg.cache.format, cfg.cache.compression_level)

        for idx, scenario in enumerate(scenarios):
            cache.save_scenario(scenario)
            logger.info(f"Processed scenario {idx + 1} / {len(scenarios)} in thread_id={thread_id}, node_id={node_id}")

        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")

    cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return []


def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)  # type: ignore


def run_caching(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the gym scenario caching.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    scenario_builder: NuPlanScenarioBuilder = build_scenario_builder(cfg)
    logger.debug(
        "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
        "accidentally caching each scenario multiple times!"
    )
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)

    data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]
    logger.info(f"Starting dataset caching of {len(data_points)} files...")

    worker_map(worker, cache_scenarios, data_points)
    logger.info("Completed dataset caching!")
