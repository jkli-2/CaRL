import itertools
import logging
from typing import List, Optional, Tuple, Type

import numpy as np

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver,
    ILQRSolverParameters,
    ILQRWarmStartParameters,
)
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from carl_nuplan.planning.gym.environment.trajectory_builder.action_trajectory_builder import ActionTrajectoryBuilder
from carl_nuplan.planning.simulation.trajectory.action_trajectory import ActionTrajectory

logger = logging.getLogger(__name__)


class LogActionPlanner(AbstractPlanner):
    """
    Planner that uses the human log action to compute the action.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = True

    def __init__(
        self,
        scenario: AbstractScenario,
        num_poses: int,
        future_time_horizon: float,
        iterative: bool = False,
    ):
        """
        Initializes the LogActionPlanner.
        :param scenario: The scenario to use for planning.
        :param num_poses: Number of poses to sample in the future trajectory.
        :param future_time_horizon: Time horizon in seconds for the future trajectory.
        :param iterative: Whether to use iLQR instead of LQR as controller, defaults to False
        """
        self._scenario = scenario

        self._num_poses = num_poses
        self._future_time_horizon = future_time_horizon
        self._iterative = iterative

        self._action: Optional[ActionTrajectory] = None
        self._tracker: Optional[AbstractTracker] = None

        self._trajectory_builder = ActionTrajectoryBuilder()

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """Inherited, see superclass."""

        if self._iterative:
            self._tracker = ILQRTracker(
                n_horizon=40,
                ilqr_solver=ILQRSolver(
                    solver_params=ILQRSolverParameters(
                        discretization_time=0.2,
                        state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
                        input_cost_diagonal_entries=[1.0, 10.0],
                        state_trust_region_entries=[1.0, 1.0, 1.0, 1.0, 1.0],
                        input_trust_region_entries=[1.0, 1.0],
                        max_ilqr_iterations=20,
                        convergence_threshold=1e-6,
                        max_solve_time=0.05,
                        max_acceleration=3.0,
                        max_steering_angle=1.047197,
                        max_steering_angle_rate=0.5,
                        min_velocity_linearization=0.01,
                    ),
                    warm_start_params=ILQRWarmStartParameters(
                        k_velocity_error_feedback=0.5,
                        k_steering_angle_error_feedback=0.05,
                        lookahead_distance_lateral_error=15.0,
                        k_lateral_error=0.1,
                        jerk_penalty_warm_start_fit=1e-4,
                        curvature_rate_penalty_warm_start_fit=1e-2,
                    ),
                ),
            )
        else:
            self._tracker = LQRTracker(
                q_longitudinal=[10.0],
                r_longitudinal=[1.0],
                q_lateral=[1.0, 10.0, 0.0],
                r_lateral=[1.0],
                discretization_time=0.1,
                tracking_horizon=10,
                jerk_penalty=1e-4,
                curvature_rate_penalty=1e-2,
                stopping_proportional_gain=0.5,
                stopping_velocity=0.2,
            )

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        log_current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        ego_state, _ = current_input.history.current_state

        try:
            states = self._scenario.get_ego_future_trajectory(
                current_input.iteration.index,
                self._future_time_horizon,
                self._num_poses,
            )
            trajectory = InterpolatedTrajectory(list(itertools.chain([log_current_state], states)))
            dynamic_state = self._tracker.track_trajectory(current_input.iteration, None, ego_state, trajectory)
            action = self._convert_dynamic_state_to_action(dynamic_state, ego_state)

            self._action = self._trajectory_builder.build_trajectory(action, ego_state)

        except AssertionError:
            logger.warning("Cannot retrieve future ego trajectory. Using previous computed action.")
            if self._action is None:
                raise RuntimeError("Future ego action cannot be retrieved from the scenario!")

        return self._action

    def _convert_dynamic_state_to_action(
        self, dynamic_state: DynamicCarState, ego_state: EgoState
    ) -> Tuple[float, float]:

        max_steering_angle: float = 0.83775804096
        max_acceleration: float = 2.4
        max_deceleration: float = 3.2

        updated_acceleration = dynamic_state.rear_axle_acceleration_2d.x
        updated_steering_angle = ego_state.tire_steering_angle + 0.1 * dynamic_state.tire_steering_rate

        if updated_acceleration >= 0:
            normed_acceleration = np.clip(updated_acceleration / max_acceleration, 0.0, 1.0)
        else:
            normed_acceleration = np.clip(updated_acceleration / max_deceleration, -1.0, 0.0)

        normed_steering_angle = (
            np.clip(updated_steering_angle, -max_steering_angle, max_steering_angle) / max_steering_angle
        )

        return normed_acceleration, normed_steering_angle
