import _pickle as pickle
import math

import carla
import numpy as np
import shapely.geometry
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


class RunRedLight:
    """
    Detects traffic light violations for vehicles in CARLA simulator.
    """

    def __init__(self, distance_threshold_for_selection=8.0, distance_light=10.0):
        self._distance_threshold_for_selection = distance_threshold_for_selection
        self._distance_light = distance_light

        # Map and world state
        self._map_name = None
        self._carla_map = None
        self._vehicle = None
        self._world = None

        # Traffic light data
        self._traffic_light_data = None
        self._road_lane_id_mapping = None
        self._traffic_light_id_to_obj = {}
        self._next_traffic_light_locations_by_road_lane = None
        self._traffic_lights_tree = None

        # Current state
        self._target_traffic_light = None
        self._vehicle_extent_x = 0.0

    def reset(self, carla_map, vehicle, world):
        """
        Reset the detector with new map, vehicle, and world instances.
        """
        self._carla_map = carla_map
        self._vehicle = vehicle
        self._world = world
        self._vehicle_extent_x = self._vehicle.bounding_box.extent.x

        # Load map-specific traffic light data
        map_name = self._carla_map.name.split("/")[-1]
        if map_name != self._map_name:
            with open(f"map_data/{map_name}.pkl", "rb") as f:
                data = pickle.load(f)
                traffic_light_data = data["traffic_lights"]

                self._traffic_light_data = traffic_light_data["traffic_light_data"]
                self._road_lane_id_mapping = traffic_light_data["road_lane_id_mapping"]
                self._map_name = map_name

        # Build traffic light object mapping
        self._traffic_light_id_to_obj = {
            self._compute_traffic_light_id(tl): tl for tl in self._world.get_actors().filter("*traffic.traffic_light*")
        }

        # Prepare traffic light location data
        self._next_traffic_light_locations_by_road_lane = {
            key: [(tl_id, loc) for (tl_id, loc) in value] for key, value in self._road_lane_id_mapping.items()
        }

        # Build spatial index for efficient querying
        traffic_light_positions = np.array([x[1] for x in self._traffic_light_data])
        self._traffic_lights_tree = cKDTree(traffic_light_positions)

        self._target_traffic_light = None

    def _compute_traffic_light_id(self, actor):
        """Compute a unique ID for traffic lights based on their location."""
        location = actor.get_location()
        return hash((round(location.x, 2), round(location.y, 2), round(location.z, 2)))

    def step(self, remaining_route, route_wp):
        """
        Check for traffic light violations at the current step.
        """
        run_red_light = False
        distance_to_next_red_traffic_light = float("inf")

        transform = self._vehicle.get_transform()

        # Use route location instead of vehicle location for more accurate detection, otherwise it could drive around red lights
        location = carla.Location(x=remaining_route[0, 0], y=remaining_route[0, 1], z=remaining_route[0, 2])
        location_np = np.array([location.x, location.y, location.z])

        # Calculate distance to next red traffic light
        distance_to_next_red_traffic_light = self._get_distance_to_next_red_light(location_np, route_wp)

        # Check for red light violation
        run_red_light = self._check_red_light_violation(location, transform, location_np)

        return run_red_light, distance_to_next_red_traffic_light

    def _get_distance_to_next_red_light(self, location_np, route_wp):
        """
        Calculate distance to the next red traffic light on the route.
        """
        key = (route_wp.road_id, route_wp.lane_id)
        next_traffic_light_locations = self._next_traffic_light_locations_by_road_lane[key]

        # Filter out green lights
        non_green_lights = [
            loc
            for (tl_id, loc) in next_traffic_light_locations
            if self._traffic_light_id_to_obj[tl_id].state != carla.TrafficLightState.Green
        ]

        if not non_green_lights:
            return float("inf")

        non_green_lights_array = np.array(non_green_lights).reshape((-1, 3))
        return cdist([location_np], non_green_lights_array, "euclidean").min()

    def _check_red_light_violation(self, location, transform, location_np):
        """
        Check if the vehicle is violating a red traffic light.
        """
        # Calculate vehicle tail points for line crossing detection
        tail_close_pt = self._rotate_point(carla.Vector3D(-0.8 * self._vehicle_extent_x, 0, 0), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)
        tail_close_pt_np = np.array([tail_close_pt.x, tail_close_pt.y, tail_close_pt.z])

        tail_far_pt = self._rotate_point(carla.Vector3D(-self._vehicle_extent_x - 1, 0, 0), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)
        tail_far_pt_np = np.array([tail_far_pt.x, tail_far_pt.y, tail_far_pt.z])

        # Find nearby traffic lights
        indices = self._traffic_lights_tree.query_ball_point(location_np, r=self._distance_light)

        for idx in indices:
            if self._is_violating_traffic_light(
                idx, transform, tail_close_pt, tail_far_pt, tail_close_pt_np, tail_far_pt_np
            ):
                print("Red light violation detected", flush=True)
                return True

        return False

    def _is_violating_traffic_light(
        self,
        idx,
        transform,
        tail_close_pt,
        tail_far_pt,
        tail_close_pt_np,
        tail_far_pt_np,
    ):
        """
        Check if the vehicle is violating a specific traffic light.
        """
        traffic_light_data = self._traffic_light_data[idx]
        key, traffic_light_location, wp_road_id, wp_lane_id, forward_vector, lft_lane_wp, rgt_lane_wp = (
            traffic_light_data
        )

        traffic_light = self._traffic_light_id_to_obj[key]
        if traffic_light.state != carla.TrafficLightState.Red:
            return False

        # Check if vehicle is moving towards the traffic light
        vehicle_forward = transform.get_forward_vector()
        vehicle_direction = np.array([vehicle_forward.x, vehicle_forward.y, vehicle_forward.z])

        if np.dot(vehicle_direction, forward_vector) <= 0:
            return False

        # Check if vehicle is in the correct lane
        tail_wp = self._carla_map.get_waypoint(tail_far_pt)
        if tail_wp.road_id != wp_road_id or tail_wp.lane_id != wp_lane_id:
            return False

        # Check if vehicle is crossing the traffic light line
        return self._is_vehicle_crossing_line((tail_close_pt_np, tail_far_pt_np), (lft_lane_wp, rgt_lane_wp))

    def _is_vehicle_crossing_line(self, vehicle_segment, traffic_light_segment):
        """
        Check if vehicle trajectory crosses a traffic light line segment.
        """
        vehicle_line = shapely.geometry.LineString(
            [(vehicle_segment[0][0], vehicle_segment[0][1]), (vehicle_segment[1][0], vehicle_segment[1][1])]
        )
        traffic_light_line = shapely.geometry.LineString(
            [
                (traffic_light_segment[0][0], traffic_light_segment[0][1]),
                (traffic_light_segment[1][0], traffic_light_segment[1][1]),
            ]
        )

        intersection = vehicle_line.intersection(traffic_light_line)
        return not intersection.is_empty

    def _rotate_point(self, point, angle_degrees):
        """
        Rotate a point by a given angle around the Z-axis.
        """
        angle_rad = math.radians(angle_degrees)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        x_rotated = cos_angle * point.x - sin_angle * point.y
        y_rotated = sin_angle * point.x + cos_angle * point.y

        return carla.Vector3D(x_rotated, y_rotated, point.z)

    def destroy(self):
        """Clean up resources."""
        pass
