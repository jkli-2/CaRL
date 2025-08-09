import carla
import numpy as np
from collections import defaultdict
import math
from typing import List, Tuple, Dict, Any


def rotate_point(point: carla.Vector3D, angle: float) -> carla.Vector3D:
    """Rotate a given point by a given angle (in degrees)."""
    rad_angle = math.radians(angle)
    x_ = math.cos(rad_angle) * point.x - math.sin(rad_angle) * point.y
    y_ = math.sin(rad_angle) * point.x + math.cos(rad_angle) * point.y
    return carla.Vector3D(x_, y_, point.z)

def get_traffic_light_waypoints(traffic_light: carla.TrafficLight, carla_map) -> Tuple[carla.Location, List[carla.Waypoint]]:
    """Get the area and waypoints of a given traffic light."""
    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0) # 0.9 to avoid crossing to adjacent lanes

    area = [
        area_loc + carla.Location(x=rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot).x,
                                  y=rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot).y)
        for x in x_values
    ]

    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    for wpx in ini_wps:
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        wps.append(wpx)

    return area_loc, wps



class RoadLaneID:
    def __init__(self, road_id: int, lane_id: int, wp: carla.Waypoint, traffic_light_locations: Dict[Tuple[int, int], List[Tuple[int, List[float]]]]):
        self.road_id = road_id
        self.lane_id = lane_id
        self.key = (road_id, lane_id)
        self.wp = wp
        self.next_traffic_light_locations = traffic_light_locations[self.key].copy()

    def calculate_next_traffic_lights(self, traffic_light_locations: Dict[Tuple[int, int], List[Tuple[int, List[float]]]]):
        """Calculate the next traffic lights within 100 meters."""
        queue = [(self.wp, 0, True)]
        visited = set()

        while queue:
            wp, distance, same_road_lane = queue.pop(0)

            if distance < 100:
                for next_wp in wp.next(1):
                    still_same_road_lane = same_road_lane and (next_wp.road_id, next_wp.lane_id) == (self.road_id, self.lane_id)
                    new_distance = 0 if still_same_road_lane else distance + 1
                    queue.append((next_wp, new_distance, still_same_road_lane))

                    if not still_same_road_lane:
                        key = (next_wp.road_id, next_wp.lane_id)
                        for traffic_light_location in traffic_light_locations[key]:
                            if traffic_light_location not in self.next_traffic_light_locations:
                                self.next_traffic_light_locations.append(traffic_light_location)


def compute_traffic_light_stop_sign_id(actor: carla.Actor) -> int:
    """Compute a unique ID for traffic lights and stop signs based on their location."""
    location = actor.get_location()
    return hash((round(location.x, 2), round(location.y, 2), round(location.z, 2)))

def compute_traffic_light_data(carla_world: carla.World, carla_map: carla.Map) -> Dict[str, Any]:
    traffic_lights = [
        (_actor, *get_traffic_light_waypoints(_actor, carla_map))
        for _actor in carla_world.get_actors()
        if 'traffic_light' in _actor.type_id
    ]

    # Remove duplicates
    tmp = []
    for traffic_light, center, wps in traffic_lights:
        for wp in wps:
            tmp.append((wp, traffic_light))

    ## Remove duplicates, somehow there are duplicates
    locations = [x[0].transform.location for x in tmp]
    locations = np.array([[x.x, x.y] for x in locations])

    unique_waypoints = []
    unique_traffic_lights = []
    for wp, traffic_light in tmp:
        if not unique_waypoints:
            unique_waypoints.append(wp)
            unique_traffic_lights.append(traffic_light)
            continue

        loc = np.array([[loc.transform.location.x, loc.transform.location.y] for loc in unique_waypoints])
        closest_distance = np.linalg.norm(loc - np.array([wp.transform.location.x, wp.transform.location.y]), axis=1).min()

        # Usually the lane width is 3.5 meters but to be sure we use 2 meters as filtering threshold
        if closest_distance >= 2:
            unique_waypoints.append(wp)
            unique_traffic_lights.append(traffic_light)

    traffic_light_data = []
    traffic_light_polygons = []
    road_lane_to_traffic_light_locations = defaultdict(list)

    for wp, traffic_light in zip(unique_waypoints, unique_traffic_lights):
        forward_vector = wp.transform.get_forward_vector()
        forward_vector = np.array([forward_vector.x, forward_vector.y, forward_vector.z])

        yaw_wp = wp.transform.rotation.yaw
        lane_width = wp.lane_width
        location_wp = wp.transform.location

        yaw = np.radians(yaw_wp)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw), np.cos(yaw)]])

        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * lane_width / 2
        translated_corners = (corners @ rotation_matrix.T) + np.array([location_wp.x, location_wp.y])
        traffic_light_polygons.append(translated_corners)

        # somehow the red light detection code in CARLA has a bug and does not detect running some traffic lights, so we need the following 3 lines
        wp_test = carla_map.get_waypoint(carla.Location(location_wp.x, location_wp.y))
        if (wp.road_id, wp.lane_id) != (wp_test.road_id, wp_test.lane_id):
            wp = wp_test

        lft_lane_wp = location_wp + carla.Location(rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp + 90))
        lft_lane_wp = np.array([lft_lane_wp.x, lft_lane_wp.y, lft_lane_wp.z])
        rgt_lane_wp = location_wp + carla.Location(rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp - 90))
        rgt_lane_wp = np.array([rgt_lane_wp.x, rgt_lane_wp.y, rgt_lane_wp.z])
        
        key = compute_traffic_light_stop_sign_id(traffic_light)
        traffic_light_location = (wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)
        if 2 < location_wp.x < 3 and 149 < location_wp.y < 150 and "Town03" in carla_map.name:
            traffic_light_data.append((key, traffic_light_location, 31, wp.lane_id, forward_vector, lft_lane_wp, rgt_lane_wp))
        else:
            traffic_light_data.append((key, traffic_light_location, wp.road_id, wp.lane_id, forward_vector, lft_lane_wp, rgt_lane_wp))
    
    traffic_light_polygons = np.array(traffic_light_polygons)

    traffic_light_coordinates = np.array([[x.transform.location.x, x.transform.location.y, x.transform.location.z] for x in unique_waypoints]).astype('float')

    if not traffic_light_coordinates.shape[0]:
        traffic_light_coordinates = np.empty((0, 3), dtype='float32')

    road_lane_to_traffic_light_locations = defaultdict(list)

    for wp, traffic_light_coord, traffic_light in zip(unique_waypoints, traffic_light_coordinates, unique_traffic_lights):
        road_lane_key = (wp.road_id, wp.lane_id)
        hash_code = compute_traffic_light_stop_sign_id(traffic_light)
        road_lane_to_traffic_light_locations[road_lane_key].append((hash_code, traffic_light_coord.tolist()))

    waypoints = carla_map.generate_waypoints(0.1)

    ids_to_obj = {}
    for wp in waypoints:
        key = (wp.road_id, wp.lane_id)
        if key not in ids_to_obj:
            ids_to_obj[key] = RoadLaneID(wp.road_id, wp.lane_id, wp, road_lane_to_traffic_light_locations)


    for item in ids_to_obj.values():
        item.calculate_next_traffic_lights(road_lane_to_traffic_light_locations)

    road_lane_id_mapping = {key: value.next_traffic_light_locations for key, value in ids_to_obj.items()}

    return {
        'road_lane_id_mapping': road_lane_id_mapping,
        'traffic_light_polygons': traffic_light_polygons,
        'traffic_light_data': traffic_light_data
    }