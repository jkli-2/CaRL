# This module provides route planning for CARLA autonomous driving agents. It
# handles route smoothing, densification, lane computation, and real-time
# route tracking with distance calculations.

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import carla


class RoutePlanner:
    DEFAULT_POINT_DISTANCE = 0.1  # meters
    MIN_ROUTE_POINTS = 2

    def __init__(self, point_distance=DEFAULT_POINT_DISTANCE) -> None:
        self._point_distance = point_distance

        self._route_index = 0
        self._route = None
        self._lanes = None
        self._route_rotations = None
        self._route_forward_vectors = None

        self._vehicle = None

    def reset(self, route, vehicle, carla_map):
        """
        route: list of tuples of the form (transform, command)
        vehicle: Vehicle to consider
        """
        self._vehicle = vehicle

        # Extract route components
        locations = [transform[0].location for transform in route]
        rotations = [transform[0].rotation for transform in route]
        forward_vectors = [transform[0].get_forward_vector() for transform in route]

        # Convert to numpy arrays
        sparse_route = np.array([[loc.x, loc.y, loc.z] for loc in locations])
        sparse_rotations = np.array([rot.yaw for rot in rotations])
        sparse_forward_vectors = np.array([[vec.x, vec.y, vec.z] for vec in forward_vectors])

        # Process route
        self._route, self._route_rotations, self._route_forward_vectors = self._smooth_and_densify_route(
            sparse_route, sparse_rotations, sparse_forward_vectors
        )

        # Compute lane information
        self._lanes = self._compute_lanes(carla_map)

        # Reset tracking
        self._route_index = 0

    def _smooth_and_densify_route(self, sparse_route, sparse_rotations, sparse_forward_vectors):
        # Convert sparse route waypoints into smooth, dense trajectory.

        if len(sparse_route) < self.MIN_ROUTE_POINTS:
            return sparse_route, sparse_rotations, sparse_forward_vectors

        # Normalize rotations and handle discontinuities
        smooth_rotations = self._normalize_rotations(sparse_rotations)

        # Calculate cumulative distance along the route
        segment_distances = np.linalg.norm(np.diff(sparse_route, axis=0), axis=1)
        cumulative_distance = np.cumsum(np.insert(segment_distances, 0, 0))

        # Create interpolation functions for each dimension
        position_interp = [interp1d(cumulative_distance, sparse_route[:, i]) for i in range(3)]
        rotation_interp = interp1d(cumulative_distance, smooth_rotations)
        forward_interp = [interp1d(cumulative_distance, sparse_forward_vectors[:, i]) for i in range(3)]

        # Generate dense waypoints
        new_distances = np.arange(0, cumulative_distance[-1], self._point_distance)

        # Interpolate all components
        smooth_route = np.column_stack([func(new_distances) for func in position_interp])
        smooth_rotations = rotation_interp(new_distances) % 360.0
        smooth_forward_vectors = np.column_stack([func(new_distances) for func in forward_interp])

        # Normalize forward vectors
        norms = np.linalg.norm(smooth_forward_vectors, axis=1, keepdims=True)
        smooth_forward_vectors = smooth_forward_vectors / np.maximum(norms, 1e-8)

        return smooth_route, smooth_rotations, smooth_forward_vectors

    def _normalize_rotations(self, rotations):
        """
        Normalize rotation angles to handle 0°/360° discontinuities.
        """
        normalized = rotations.copy() % 360.0
        cumulative_offset = 0.0

        for i in range(1, len(normalized)):
            angle_diff = normalized[i] + cumulative_offset - normalized[i - 1]

            if angle_diff > 180:
                cumulative_offset -= 360.0
            elif angle_diff < -180:
                cumulative_offset += 360.0

            normalized[i] += cumulative_offset

        return normalized

    def _waypoint_to_array(self, waypoint):
        # Convert CARLA waypoint to numpy array.
        return np.array(
            [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z],
            dtype=np.float32,
        )

    def _waypoint_to_list(self, waypoint):
        return np.array([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z], dtype=np.float32)

    def _compute_lanes(self, carla_map):
        # Compute available lanes for each route waypoint.
        lanes = []
        for route_point in self._route:
            location = carla.Location(x=float(route_point[0]), y=float(route_point[1]), z=float(route_point[2]))
            waypoint = carla_map.get_waypoint(location)

            if waypoint is None:
                continue

            # Start with current lane
            lane = [self._waypoint_to_array(waypoint)]

            # waypoints to the left
            curr_wp = waypoint
            while True:
                if curr_wp.lane_change == carla.LaneChange.Left or curr_wp.lane_change == carla.LaneChange.Both:
                    curr_wp = curr_wp.get_left_lane()

                    if curr_wp is not None and curr_wp.lane_type == carla.LaneType.Driving:
                        lane.append(self._waypoint_to_list(curr_wp))
                    else:
                        break
                else:
                    break

            # waypoints to the right
            curr_wp = waypoint
            while True:
                if curr_wp.lane_change == carla.LaneChange.Right or curr_wp.lane_change == carla.LaneChange.Both:
                    curr_wp = curr_wp.get_right_lane()

                    if curr_wp is not None and curr_wp.lane_type == carla.LaneType.Driving:
                        lane.append(self._waypoint_to_list(curr_wp))
                    else:
                        break
                else:
                    break

            lanes.append(lane)

        return lanes

    def step(self):
        # Update route tracking based on current vehicle position.
        vehicle_location = self._vehicle.get_location()
        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z])

        initial_index = self._route_index

        while True:
            # Vector from current route point to vehicle
            route_to_vehicle = vehicle_pos - self._route[self._route_index]

            # Check if vehicle has passed this waypoint
            # (negative dot product means vehicle is behind the waypoint)
            progress = np.dot(route_to_vehicle, self._route_forward_vectors[self._route_index])

            if progress < 0:
                break

            self._route_index += 1

        # Calculate traveled distance
        waypoints_passed = self._route_index - initial_index
        traveled_distance = waypoints_passed * self._point_distance

        remaining_route = self._route[self._route_index :]
        remaining_lanes = self._lanes[self._route_index :]

        return remaining_route, remaining_lanes, traveled_distance

    def destroy(self):
        pass
