import carla
import numpy as np
import pickle
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Any

def compute_traffic_light_stop_sign_id(actor: carla.Actor) -> int:
    """Compute a unique ID for traffic lights and stop signs based on their location."""
    location = actor.get_location()
    return hash((location.x, location.y, location.z))

class RoadLaneID:
    def __init__(self, road_id: int, lane_id: int, wp: carla.Waypoint, stop_sign_locations):
        self.road_id = road_id
        self.lane_id = lane_id
        self.key = (road_id, lane_id)
        self.wp = wp
        self.next_stop_sign_locations = stop_sign_locations[self.key].copy()

    def calculate_next_stop_signs(self, stop_sign_locations: Dict[Tuple[int, int], List[Tuple[int, List[float]]]]):
        """Calculate the next stop signs within 100 meters."""
        queue = [(self.wp, 0, True)]  # (wp, distance, still_same_road_lane)

        while queue:
            wp, distance, same_road_lane = queue.pop(0)

            if distance < 100:
                for next_wp in wp.next(1):
                    still_same_road_lane = same_road_lane and (next_wp.road_id, next_wp.lane_id) == (self.road_id, self.lane_id)
                    new_distance = 0 if still_same_road_lane else distance + 1
                    queue.append((next_wp, new_distance, still_same_road_lane))

                    if not still_same_road_lane:
                        key = (next_wp.road_id, next_wp.lane_id)
                        for stop_sign_location in stop_sign_locations[key]:
                            if stop_sign_location not in self.next_stop_sign_locations:
                                self.next_stop_sign_locations.append(stop_sign_location)

def calculate_stop_sign_polygons(stop_sign_waypoints: List[carla.Waypoint]) -> np.ndarray:
    """Calculate polygons for stop signs."""
    polygons = []
    for wp in stop_sign_waypoints:
        yaw = np.radians(wp.transform.rotation.yaw)
        location = wp.transform.location

        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw), np.cos(yaw)]])

        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * wp.lane_width / 2
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([location.x, location.y])
        
        polygons.append(translated_corners)

    return np.array(polygons)

def compute_stop_sign_data(carla_world: carla.World, carla_map: carla.Map) -> Dict[str, Any]:
    """Process a single town and return the collected data."""
    waypoints = carla_map.generate_waypoints(0.1)
    waypoints_locations = np.array([x.transform.location for x in waypoints])

    stop_sign_actors = carla_world.get_actors().filter('*traffic.stop*')
    stop_sign_locations = [x.get_transform().transform(x.trigger_volume.location) for x in stop_sign_actors]
    stop_sign_waypoints = [carla_map.get_waypoint(x) for x in stop_sign_locations]
    stop_sign_coordinates = np.array([[x.transform.location.x, x.transform.location.y, x.transform.location.z] for x in stop_sign_waypoints]).astype('float')

    if not stop_sign_coordinates.shape[0]:
        stop_sign_coordinates = np.empty((0, 3), dtype='float32')

    road_lane_to_stop_sign_locations = defaultdict(list)

    for wp, stop_sign_coord, stop_sign in zip(stop_sign_waypoints, stop_sign_coordinates, stop_sign_actors):
        road_lane_key = (wp.road_id, wp.lane_id)
        hash_code = compute_traffic_light_stop_sign_id(stop_sign)
        road_lane_to_stop_sign_locations[road_lane_key].append((hash_code, stop_sign_coord.tolist()))

    ids_to_obj = {}
    for wp in waypoints:
        key = (wp.road_id, wp.lane_id)
        if key not in ids_to_obj:
            ids_to_obj[key] = RoadLaneID(wp.road_id, wp.lane_id, wp, road_lane_to_stop_sign_locations)

    for item in ids_to_obj.values():
        item.calculate_next_stop_signs(road_lane_to_stop_sign_locations)

    stop_sign_polygons = calculate_stop_sign_polygons(stop_sign_waypoints)

    road_lane_id_mapping = {key: value.next_stop_sign_locations for key, value in ids_to_obj.items()}
    stop_sign_data = {
        'road_lane_id_mapping': road_lane_id_mapping,
        'stop_sign_polygons': stop_sign_polygons
    }

    return stop_sign_data