from __future__ import annotations

from typing import Any, List, Tuple, Type

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, TimeDuration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class ActionTrajectory(InterpolatedTrajectory):
    """
    Dummy trajectory that stored the action for the kinetic bicycle model.
    """

    def __init__(
        self,
        acceleration: float,
        steering_rate: float,
        ego_state: EgoState,
        raw_action: List[float] = [],
        store: Any = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(num_poses=80, interval_length=0.1),
    ):
        """
        Initializes the ActionTrajectory.
        :param acceleration: Longitudinal acceleration [m/s^2].
        :param steering_rate: Steering rate [rad/s].
        :param ego_state: Ego state at the start of the trajectory.
        :param raw_action: unnormalized action, stored in simulation log TODO: refactor, defaults to []
        :param store: arbitrary object, stored in simulation log TODO: refactor, defaults to None
        :param trajectory_sampling: sampling for dummy constant velocity trajectory, defaults to TrajectorySampling(num_poses=80, interval_length=0.1)
        """

        super().__init__(trajectory=_get_dummy_trajectory(ego_state, trajectory_sampling))

        self._acceleration = acceleration
        self._steering_rate = steering_rate
        self._ego_state = ego_state
        self._raw_action = raw_action  # TODO: refactor
        self._store = store
        self._trajectory_sampling = trajectory_sampling

    def __reduce__(self) -> Tuple[Type[ActionTrajectory], Tuple[Any, ...]]:
        """
        Helper for pickling.
        """
        return self.__class__, (
            self._acceleration,
            self._steering_rate,
            self._ego_state,
            self._raw_action,
            self._store,
            self._trajectory_sampling,
        )

    @property
    def dynamic_car_state(self) -> DynamicCarState:
        """Helper property to get the dynamic car state."""
        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=self._ego_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=self._ego_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(self._acceleration, 0),
            tire_steering_rate=self._steering_rate,
        )


def _get_dummy_trajectory(ego_state: EgoState, trajectory_sampling: TrajectorySampling) -> List[EgoState]:
    """
    Helper function to create a dummy trajectory with constant velocity.
    :param ego_state: Ego state at the start of the trajectory.
    :param trajectory_sampling: sampling for dummy constant velocity trajectory.
    :return: List of EgoState representing the dummy trajectory.
    """

    time_dot = TimeDuration.from_s(trajectory_sampling.interval_length)
    trajectory: List[EgoState] = [ego_state]

    for time_idx in range(trajectory_sampling.num_poses):
        time_point = ego_state.time_point + time_idx * time_dot
        state = EgoState.build_from_rear_axle(
            rear_axle_pose=ego_state.rear_axle,
            rear_axle_velocity_2d=ego_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            tire_steering_angle=ego_state.tire_steering_angle,
            time_point=time_point,
            vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
            is_in_auto_mode=True,
            angular_vel=ego_state.dynamic_car_state.angular_velocity,
            angular_accel=ego_state.dynamic_car_state.angular_acceleration,
        )
        trajectory.append(state)

    return trajectory
