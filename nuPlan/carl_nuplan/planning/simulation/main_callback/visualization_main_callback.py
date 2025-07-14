import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from nuplan.planning.simulation.main_callback.abstract_main_callback import (
    AbstractMainCallback,
)
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from carl_nuplan.planning.simulation.main_callback.render_simulation_log import render_simulation_log


class VisualizationMainCallback(AbstractMainCallback):
    """
    Callback to visualize simulation results.
    NOTE: The function `render_simulation_log_v2` only works for the PPO agents and plots actions.
        Use `render_simulation_log` for general use.
    TODO: Refactor this class.
    """

    def __init__(
        self,
        output_dir: str,
        aggregator_metric_dir: str = "aggregator_metric",
        simulation_log_dir: str = "simulation_log",
        threshold: float = 0.5,
        scenario_tokens: Optional[List[str]] = None,
        worker: Optional[WorkerPool] = None,
    ):
        self._output_dir = Path(output_dir)
        self._aggregator_metric_dir = aggregator_metric_dir
        self._simulation_log_dir = simulation_log_dir
        self._visualization_dir = "visualization"

        self._scenario_tokens = scenario_tokens
        self._threshold = threshold
        self._worker = worker

    @staticmethod
    def _read_scenario_metrics(aggregator_metric_path: Path) -> pd.DataFrame:
        aggregator_metric_file = list(aggregator_metric_path.rglob("*.parquet"))
        assert (
            len(aggregator_metric_file) == 1
        ), f"Found {len(aggregator_metric_file)} files in {aggregator_metric_path}"
        df_aggregator = pd.read_parquet(aggregator_metric_file[0])
        df_aggregator_scenario = df_aggregator[~df_aggregator["num_scenarios"].notnull()]
        return df_aggregator_scenario

    @staticmethod
    def _find_simulation_log_paths(simulation_log_path: Path, scenario_tokens: List[str]) -> List[Path]:
        simulation_log_paths = []
        for planner_dir_folder in simulation_log_path.iterdir():
            for scenario_type_folder in planner_dir_folder.iterdir():
                for log_name_folder in scenario_type_folder.iterdir():
                    for scenario_name_folder in log_name_folder.iterdir():
                        if scenario_name_folder.name in scenario_tokens:
                            for scenario_log_file in scenario_name_folder.iterdir():
                                simulation_log_paths.append(scenario_log_file)
        return simulation_log_paths

    def on_run_simulation_end(self) -> None:

        if self._scenario_tokens is None:
            df_scenario = self._read_scenario_metrics(self._output_dir / self._aggregator_metric_dir)
            scenario_tokens_to_visualize = list(df_scenario[df_scenario["score"] > self._threshold]["scenario"])
        else:
            scenario_tokens_to_visualize = self._scenario_tokens
        simulation_log_paths = self._find_simulation_log_paths(
            self._output_dir / self._simulation_log_dir, scenario_tokens_to_visualize
        )

        if self._worker is None:
            for simulation_log_path in tqdm(simulation_log_paths):
                simulation_log = SimulationLog.load_data(simulation_log_path)
                render_simulation_log(
                    simulation_log, self._output_dir / self._visualization_dir
                )  # TODO: Abstract this function to a renderer class
        else:
            data = [
                {
                    "simulation_log": simulation_log_path,
                    "visualization_dir": self._output_dir / self._visualization_dir,
                }
                for simulation_log_path in simulation_log_paths
            ]
            worker_map(self._worker, render_func, data)


def render_func(args: List[Dict[str, Any]]) -> None:
    def _render_internal(args: List[Dict[str, Any]]) -> None:
        for arg in args:
            simulation_log = SimulationLog.load_data(arg["simulation_log"])
            render_simulation_log(simulation_log, arg["visualization_dir"])

    _render_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()
    return []
