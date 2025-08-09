import _pickle as pickle
import numpy as np
from scipy.spatial.distance import cdist


class RunStopSign:
    def __init__(self, distance_threshold_for_clearing=5.0, distance_threshold_for_selection=6.0):
        self._distance_threshold_for_clearing = distance_threshold_for_clearing
        self._distance_threshold_for_selection = distance_threshold_for_selection

        # CARLA-specific objects
        self._carla_map = None
        self._vehicle = None

        # Stop sign tracking state
        self._cleared_stop_sign = False
        self._target_stop_sign_location = None
        self._next_stop_sign_by_road_lane = None

        # Map caching
        self._map_name = None

    def reset(self, carla_map, vehicle):
        """
        Reset the monitor with new map and vehicle instances.
        """
        self._carla_map = carla_map
        self._vehicle = vehicle
        map_name = self._carla_map.name.split("/")[-1]

        # Load stop sign data if map changed
        if self._map_name != map_name:
            with open(f"map_data/{map_name}.pkl", "rb") as f:
                data = pickle.load(f)
                # Dictionary mapping road/lane IDs to stop sign locations
                self._next_stop_sign_by_road_lane = data["stop_signs"]["road_lane_id_mapping"]
                self._next_stop_sign_by_road_lane = {
                    key: [loc for (id, loc) in value] for key, value in self._next_stop_sign_by_road_lane.items()
                }
                self._map_name = map_name

        # Reset tracking state
        self._cleared_stop_sign = False
        self._target_stop_sign_location = None

    def step(self, remaining_route, route_wp):
        """
        Process one step of stop sign monitoring.
        """
        run_stop_sign = False
        distance_to_next_stop_sign = float("inf")

        closest_route_location = remaining_route[0]

        # Clear stop sign if vehicle has passed and is far enough away
        if self._cleared_stop_sign:
            distance = cdist([self._target_stop_sign_location], [closest_route_location], "euclidean")[0][0]

            if distance > self._distance_threshold_for_clearing:
                self._cleared_stop_sign = False
                self._target_stop_sign_location = None
        else:
            # Calculate route direction vector
            route_vector = remaining_route[1] - closest_route_location

            # Monitor current target stop sign
            if self._target_stop_sign_location is not None:
                distance = cdist([self._target_stop_sign_location], [closest_route_location], "euclidean")[0][0]
                distance_to_next_stop_sign = distance

                vector_to_stop_sign = self._target_stop_sign_location - closest_route_location
                passed_stop_sign = np.dot(route_vector, vector_to_stop_sign) < 0

                # Check if vehicle ran the stop sign
                if passed_stop_sign and not self._cleared_stop_sign:
                    run_stop_sign = True
                    self._cleared_stop_sign = True  # this is to prevent running the same stop sign multiple times
                    print("Vehicle ran stop sign", flush=True)

                # Check if vehicle properly stopped at stop sign
                speed = self._vehicle.get_velocity().length()
                if distance < 4.0 and speed < 0.1:
                    self._cleared_stop_sign = True
            else:
                # Look for new stop signs ahead
                key = (route_wp.road_id, route_wp.lane_id)
                next_stop_sign_locations = self._next_stop_sign_by_road_lane[key]

                for stop_sign_location in next_stop_sign_locations:
                    vector_to_stop_sign = stop_sign_location - closest_route_location
                    stop_sign_in_front = np.dot(route_vector, vector_to_stop_sign) > 0

                    # Select stop sign if it's ahead and within range
                    if stop_sign_in_front:
                        distance = cdist([stop_sign_location], [closest_route_location], "euclidean")[0][0]

                        distance_to_next_stop_sign = min(distance_to_next_stop_sign, distance)

                        if distance < self._distance_threshold_for_selection:
                            self._target_stop_sign_location = stop_sign_location
                            break

        return run_stop_sign, distance_to_next_stop_sign

    def destroy(self):
        """Clean up resources. Currently no resources need explicit cleanup."""
        pass
