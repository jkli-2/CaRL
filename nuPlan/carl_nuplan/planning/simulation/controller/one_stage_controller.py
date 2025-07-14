from typing import Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.simulation.controller.abstract_controller import (
    AbstractEgoController,
)
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import (
    AbstractMotionModel,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from carl_nuplan.planning.simulation.trajectory.action_trajectory import (
    ActionTrajectory,
)


class OneStageController(AbstractEgoController):
    """
    Replace the two-stage controller with a single stage controller.
    Instead of using a controller, the ego agent is propagated directly with the action from ActionTrajectory.
    """

    def __init__(self, scenario: AbstractScenario, motion_model: AbstractMotionModel):
        """
        Initializes the OneStageController.
        :param scenario: Scenario interface of nuPlan.
        :param motion_model: Motion model, i.e. a kinematic bicycle model
        """
        self._scenario = scenario
        self._motion_model = motion_model

        #  lazy loaded
        self._current_state: Optional[EgoState] = None

    def get_state(self) -> EgoState:
        """Inherited, see superclass."""
        if self._current_state is None:
            self._current_state = self._scenario.initial_ego_state
        return self._current_state

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._current_state = None

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""

        validate_type(trajectory, ActionTrajectory)
        trajectory: ActionTrajectory = trajectory

        sampling_time = next_iteration.time_point - current_iteration.time_point

        # Compute the dynamic state to propagate the model
        dynamic_state = trajectory.dynamic_car_state

        # Propagate ego state using the motion model
        self._current_state = self._motion_model.propagate_state(
            state=ego_state,
            ideal_dynamic_state=dynamic_state,
            sampling_time=sampling_time,
        )
