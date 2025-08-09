from collections import deque

import numpy as np

from bird_eye_view_renderer import BirdEyeViewRenderer


class ObservationManager:

    DEFAULT_SCALAR_HISTORY_LENGTH = 16
    DEFAULT_SCALAR_INDICES = [-1, -6, -11, -16]
    SCALAR_FEATURES = ["speed", "steer", "throttle", "brake", "height"]

    def __init__(
        self, scalar_history_length=DEFAULT_SCALAR_HISTORY_LENGTH, scalar_history_indices=None, pixels_per_meter=None
    ):
        # CARLA objects - set during reset()
        self._vehicle = None
        self._carla_world = None
        self._carla_map = None

        # Configuration
        self._scalar_history_length = scalar_history_length
        self._scalar_history_indices = (
            scalar_history_indices if scalar_history_indices is not None else self.DEFAULT_SCALAR_INDICES
        )

        # Components
        self._bev_renderer = BirdEyeViewRenderer(pixels_per_meter=pixels_per_meter)
        self._scalar_value_history = None

    def reset(self, vehicle, carla_world, carla_map):
        # Reset the observer with new CARLA simulation objects.
        self._vehicle = vehicle
        self._carla_world = carla_world
        self._carla_map = carla_map

        # Reset components
        self._bev_renderer.reset(vehicle, carla_world, carla_map)

        # Initialize scalar history with zeros
        initial_values = [0.0] * len(self.SCALAR_FEATURES)
        self._scalar_value_history = deque(
            [initial_values] * self._scalar_history_length, maxlen=self._scalar_history_length
        )

    def _update_scalar_values(self):
        # Update the scalar value history with current vehicle state.

        # Extract vehicle state
        velocity = self._vehicle.get_velocity()
        speed = velocity.length()  # m/s

        control = self._vehicle.get_control()
        steer = control.steer  # [-1, 1]
        throttle = control.throttle  # [0, 1]
        brake = control.brake  # [0, 1]

        location = self._vehicle.get_location()
        height = location.z  # meters

        # Append to history
        current_values = [speed, steer, throttle, brake, height]
        self._scalar_value_history.append(current_values)

    def _get_scalar_input(self):
        # Extract and process scalar features from history.

        # Convert to numpy array and select specific time indices
        scalar_array = np.array(list(self._scalar_value_history))
        selected_values = scalar_array[self._scalar_history_indices]

        # Normalize height relative to current position (most recent frame)
        current_height = selected_values[0, -1]  # Height from most recent frame
        selected_values[:, -1] -= current_height

        return selected_values.flatten()

    def step(self, remaining_route, remaining_lanes, all_other_actors, all_vehicles, all_walkers, all_bicycles):
        # Generate observation for current simulation step.

        # Render bird's-eye view
        bev_image = self._bev_renderer.render(
            remaining_route, remaining_lanes, all_other_actors, all_vehicles, all_walkers, all_bicycles
        )

        # Update and extract scalar features
        self._update_scalar_values()
        scalar_values = self._get_scalar_input()

        observation = {"bev_image": bev_image, "scalars": scalar_values}

        return observation

    def destroy(self):
        pass
