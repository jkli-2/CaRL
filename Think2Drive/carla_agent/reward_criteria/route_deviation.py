import numpy as np
from scipy.spatial.distance import cdist


class RouteDeviation:
    """
    Monitors vehicle deviation from planned routes in CARLA simulation.
    """

    def __init__(self, max_deviation=15):
        """
        max_deviation: float, maximum deviation in meters
        """
        self._max_deviation = max_deviation
        self._vehicle = None
        self._route_planner = None
        self._vehicle_extent_x = 0.0
        self._vehicle_extent_y = 0.0

    def reset(self, vehicle, route_planner):
        """
        vehicle: carla.Vehicle
        """
        self._vehicle = vehicle
        self._route_planner = route_planner

        # Extract vehicle dimensions for potential future use
        bbox = self._vehicle.bounding_box
        self._vehicle_extent_x = bbox.extent.x
        self._vehicle_extent_y = bbox.extent.y

    def step(self, remaining_route):
        """
        Check if vehicle has deviated beyond acceptable limits from the route.
        """

        # Get current vehicle position
        vehicle_location = self._vehicle.get_location()
        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])

        # Find closest point on route (using first waypoint as approximation)
        closest_route_point = remaining_route[0, :2]

        # Calculate Euclidean distance
        deviation_distance = cdist([vehicle_pos], [closest_route_point], "euclidean")[0][0]

        # Check if deviation exceeds threshold
        exceeded_deviation = deviation_distance > self._max_deviation

        if exceeded_deviation:
            print("Exceeded route deviation", flush=True)

        return exceeded_deviation, deviation_distance

    def destroy(self):
        """Clean up resources and reset state."""
        pass
