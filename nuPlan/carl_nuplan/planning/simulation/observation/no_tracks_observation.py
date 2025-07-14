from typing import Type

from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration


class NoTracksObservation(AbstractObservation):
    """
    Observation that does not use tracks, but only the scenario information.
    Used for debugging and testing purposes. Might be removed in the future.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        :param scenario: The scenario
        """
        self.scenario = scenario
        self.current_iteration = 0

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        return DetectionsTracks(TrackedObjects(None))

    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
