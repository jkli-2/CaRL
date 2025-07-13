from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

useless_columns = [
    "scenario_type",
    "num_scenarios",
    "planner_name",
    "aggregator_type",
    "corners_in_drivable_area",
    "ego_jerk",
    "ego_lane_change",
    "ego_lat_acceleration",
    "ego_lon_acceleration",
    "ego_lon_jerk",
    "ego_yaw_acceleration",
    "ego_yaw_rate",
]

metric_abbreviations = {
    "score": "CLS",
    "no_ego_at_fault_collisions": "NAC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC",
    "ego_is_making_progress": "EIMP",
    "time_to_collision_within_bound": "TTC",
    "ego_progress_along_expert_route": "EPAR",
    "speed_limit_compliance": "SLC",
    "ego_is_comfortable": "EIC",
}

multiplier = {
    "no_ego_at_fault_collisions": -1,
    "drivable_area_compliance": -1,
    "driving_direction_compliance": -1,
    "ego_is_making_progress": -1,
}


weights = {
    "time_to_collision_within_bound": 5,
    "ego_progress_along_expert_route": 5,
    "speed_limit_compliance": 4,
    "ego_is_comfortable": 2,
}


class CSVMainCallback(AbstractMainCallback):
    """
    Callback to save the simulation results in CSV format.
    """

    def __init__(self, output_dir: str, aggregator_metric_dir: str):
        """
        Initializes the CSVMainCallback.
        :param output_dir: Output directory for the CSV files.
        :param aggregator_metric_dir: Folder name for the aggregator metrics.
        """
        self._output_dir = Path(output_dir)
        self._aggregator_metric_dir = aggregator_metric_dir

    def on_run_simulation_end(self) -> None:
        """inherited, see superclass."""

        aggregator_metric_path = self._output_dir / self._aggregator_metric_dir
        aggregator_metric_file = list(aggregator_metric_path.rglob("*.parquet"))
        assert (
            len(aggregator_metric_file) == 1
        ), f"Found {len(aggregator_metric_file)} files in {aggregator_metric_path}"

        df_aggregator = pd.read_parquet(aggregator_metric_file[0])

        # 1. save regular dataframe
        csv_path = aggregator_metric_file[0].with_suffix(".csv")
        df_aggregator.to_csv(csv_path, index=False)

        # 2. save scenarios sorted by score
        scenario_column_order = [0, 1, 10, 7, 2, 3, 5, 9, 6, 8, 4]
        df_scenario = df_aggregator[df_aggregator["num_scenarios"].isna()]
        df_scenario_clean = df_scenario.drop(columns=useless_columns)
        df_scenario_clean = df_scenario_clean.iloc[:, scenario_column_order]
        df_scenario_clean = df_scenario_clean.sort_values(list(metric_abbreviations.keys()))
        df_scenario_clean.to_csv(csv_path.with_name("scenarios.csv"), index=False)

        # 3. save metric overview with importance
        denominator = sum(weights.values())

        def _accumulate(df_scenario):
            """Helper function to accumulate the scores based on the multipliers and weights."""
            scores = []
            for index, row in df_scenario.iterrows():
                multiplier_values = [float(row[metric]) for metric in multiplier.keys()]
                multiplier_score = np.prod(multiplier_values)
                weighted_values = [float(row[metric]) * weight for metric, weight in weights.items()]
                weighted_score = sum(weighted_values) / denominator
                scores.append(multiplier_score * weighted_score)
            return scores

        def _importance_analysis(df, df_score):
            score = df.iloc[-1]["score"]
            _df_score = deepcopy(df_score)
            for metric, _ in multiplier.items():
                _df_scenario = deepcopy(df_scenario)
                _df_scenario[metric] = 1.0
                metric_score = np.mean(_accumulate(_df_scenario))
                _df_score["loss_" + metric_abbreviations[metric]] = metric_score - score
            for metric, _ in weights.items():
                _df_scenario = deepcopy(df_scenario)
                _df_scenario[metric] = 1.0
                metric_score = np.mean(_accumulate(_df_scenario))
                _df_score["loss_" + metric_abbreviations[metric]] = metric_score - score
            return _df_score

        overview_column_order = [8, 5, 0, 1, 3, 7, 4, 6, 2]
        df_overview = df_aggregator.iloc[-1].drop(useless_columns + ["scenario", "log_name"])

        df_overview = df_overview[overview_column_order].rename(index=metric_abbreviations)
        df_overview = pd.DataFrame(_importance_analysis(df_aggregator, df_overview)).T
        df_overview.to_csv(csv_path.with_name("overview.csv"), index=False)
