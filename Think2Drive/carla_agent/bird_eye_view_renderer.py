import carla
import _pickle as pickle
import numpy as np
from scipy.spatial import cKDTree
import math
import cv2
import matplotlib.pyplot as plt
from numba import jit, njit
from collections import deque


class BirdEyeViewRenderer:
    def __init__(
        self,
        bev_size=128,
        ego_vertical_percentage=0.3,
        route_width=3.0,
        pixels_per_meter=None,
        map_dir="map_data",
    ):
        self._bev_size = bev_size
        self._ego_vertical_offset = int(bev_size * (0.5 - ego_vertical_percentage))
        self._map_dir = map_dir
        self._ego_vertical_percentage = ego_vertical_percentage
        self._route_width = route_width

        # Temporal history configuration
        self._history_length = 16
        self._history_indices = [-1, -6, -11, -16]  # Frame indices for multi-temporal rendering

        # Calculated during reset
        self._pixels_per_meter = pixels_per_meter
        self._map_name = None
        self._ego_polygon = None
        self._ego_extent_x = None
        self._ego_extent_y = None
        self._route_thickness = 0
        self._maximum_distance_stop_signs_traffic_lights = 0

        # CARLA references
        self._ego_vehicle = None
        self._carla_world = None
        self._carla_map = None

        # Dynamic object history storage
        # contains lists of with polygons in world coordinates of (vehicle, walker, emergency car, obstacle, green traffic light, yellow traffic light, red traffic light)
        self._dynamic_objects = deque(maxlen=self._history_length)

    def reset(self, ego_vehicle, carla_world, carla_map):
        """
        Reset the renderer with new CARLA environment.
        """
        self._ego_vehicle = ego_vehicle
        self._carla_world = carla_world
        self._carla_map = carla_map

        # Clear history
        self._dynamic_objects.clear()
        for _ in range(self._history_length):
            self._dynamic_objects.append([])

        if self._map_name is None or self._map_name not in self._carla_map.name:
            self._initialize_rendering_parameters(ego_vehicle)
            self._load_map_data()
            self._calculate_detection_distance()
            self._create_ego_polygon()

    def _initialize_rendering_parameters(self, ego_vehicle):
        """Initialize rendering parameters based on ego vehicle and safety constraints."""

        # Vehicle dimensions (x and y assignment is correct for CARLA coordinate system)
        self._ego_extent_x = ego_vehicle.bounding_box.extent.y
        self._ego_extent_y = ego_vehicle.bounding_box.extent.x

        # Calculate pixels per meter based on maximum speed and minimum braking distance
        max_speed_kmh = 72.0  # km/h, also for PDM-Lite we fused 84 km/h and 72 km/h, the two speed classes, being 70 % of two highest speed limits 100 and 120 km/h
        min_braking_distance = (max_speed_kmh / 10) ** 2 / 2  # Formula to compute the min braking distance

        # Formula that a vehicle is fully visible and it's still enough time to brake to 0 km/h + plus safety margin
        if self._pixels_per_meter is None:
            # Ensure vehicle is fully visible with enough braking distance + safety margin
            self._pixels_per_meter = np.float32(
                self._bev_size
                * (1.0 - self._ego_vertical_percentage)
                / (min_braking_distance + 2 * self._ego_extent_y)
                - 0.1
            )
        else:
            self._pixels_per_meter = np.float32(self._pixels_per_meter)

        self._route_thickness = int(self._pixels_per_meter * self._route_width)

    def _load_map_data(self):
        """Load and preprocess map data from pickle file."""
        map_name = self._carla_map.name.split("/")[-1]
        self._map_name = map_name

        with open(f"{self._map_dir}/{map_name}.pkl", "rb") as f:
            data = pickle.load(f)

        # Process road network
        road_network = data["road_network"]
        grid_size = road_network["grid_size"]
        grid = road_network["grid"]

        self._max_render_distance = math.sqrt((grid_size + self._bev_size / self._pixels_per_meter) ** 2 / 2)

        # Pre-compute scaled map elements for efficient rendering
        self._grid_tiles, grid_centers = [], []

        for grid_coord, grid_data in grid.items():
            grid_centers.append(tuple(grid_coord))

            roads, white_lines, yellow_lines = grid_data

            # Pre-scale all map elements
            self._grid_tiles.append(
                (
                    [self._pixels_per_meter * road for road in roads],
                    [self._pixels_per_meter * line for line in white_lines],
                    [self._pixels_per_meter * line for line in yellow_lines],
                )
            )

        self._grid_center_tree = cKDTree(grid_centers)

        # Process stop signs
        stop_sign_data = data["stop_signs"]
        self._stop_signs = stop_sign_data["stop_sign_polygons"].reshape((-1, 4, 2))
        self._stop_signs[:, :, 1] *= -1  # Flip Y coordinate for CARLA
        stop_sign_centers = self._stop_signs.mean(axis=1)
        self._stop_signs *= self._pixels_per_meter
        self._stop_signs_tree = cKDTree(stop_sign_centers)

        # Process traffic lights
        traffic_light_data = data["traffic_lights"]
        self._precompute_traffic_light_polygons(traffic_light_data)

    def _calculate_detection_distance(self) -> None:
        """Calculate maximum distance for detecting traffic lights and stop signs."""
        self._maximum_distance_stop_signs_traffic_lights = (
            math.sqrt((self._bev_size / self._pixels_per_meter) ** 2 / 2) + 3.5 / 2
        )

    def _create_ego_polygon(self) -> None:
        """Create polygon representation of ego vehicle."""
        self._ego_polygon = np.round(
            np.array([self._bev_size / 2 + self._ego_vertical_offset, self._bev_size / 2], dtype=np.float32)
            + self._get_polygon_of_actor(
                np.float32(90),
                self._pixels_per_meter,
                self._ego_extent_x,
                self._ego_extent_y,
                np.array([0, 0], dtype=np.float32),
            )
        ).astype(np.int32)

    def _precompute_traffic_light_polygons(self, traffic_light_data):
        """Precompute traffic light polygons and create mapping to CARLA actors."""
        traffic_lights = self._carla_world.get_actors().filter("*traffic.traffic_light*")

        # Create mapping from location-based ID to traffic light actors
        traffic_light_mapping = {self._compute_traffic_light_stop_sign_id(tl): tl for tl in traffic_lights}

        self._traffic_lights = [traffic_light_mapping[x[0]] for x in traffic_light_data["traffic_light_data"]]

        # Process traffic light polygons
        traffic_light_polygons = traffic_light_data["traffic_light_polygons"].reshape((-1, 4, 2))
        traffic_light_polygons[:, :, 1] *= -1  # Flip Y coordinate
        traffic_light_centers = traffic_light_polygons.mean(axis=1)
        self._traffic_light_polygons = traffic_light_polygons * self._pixels_per_meter
        self._traffic_lights_tree = cKDTree(np.array(traffic_light_centers))

    def _compute_traffic_light_stop_sign_id(self, actor):
        """Compute unique ID for traffic lights and stop signs based on location."""
        location = actor.get_location()
        return hash((round(location.x, 2), round(location.y, 2), round(location.z, 2)))

    @staticmethod
    @jit(nopython=True)
    def _transform_points(points, center, rotation_matrix, bev_size, do_filter=True):
        """Transform points from world coordinates to BEV image coordinates."""
        points = np.round(bev_size / 2 + (points - center) @ rotation_matrix.T).astype(np.int32)

        # Filter out points completely outside the image
        if do_filter and (
            points[:, 0].max() < 0
            or points[:, 1].max() < 0
            or points[:, 0].min() > bev_size
            or points[:, 1].min() > bev_size
        ):
            return None

        return points

    def _transform_points_batched(self, points_list, rotation_matrix, scaled_bev_center, do_filter=True):
        """Apply point transformation to a batch of point lists."""
        return [
            transformed
            for points in points_list
            if (
                transformed := self._transform_points(
                    points, scaled_bev_center, rotation_matrix, self._bev_size, do_filter
                )
            )
            is not None
        ]

    @staticmethod
    @jit(nopython=True)
    def _get_rotation_matrix(yaw_deg):
        """Calculate rotation matrix from yaw angle in degrees."""
        yaw_rad = np.radians(yaw_deg)
        return np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)], [np.sin(yaw_rad), np.cos(yaw_rad)]], dtype=np.float32)

    @staticmethod
    @jit(nopython=True)
    def _transform_route(center, rotation_matrix, ego_route, ppm, bev_size):
        """Transform the ego vehicle's route to BEV image coordinates."""
        points = np.round(
            bev_size / 2 + (ppm * ego_route * np.array([1, -1], dtype=np.float32) - center) @ rotation_matrix.T
        ).astype(np.int32)
        return points

    @staticmethod
    @njit
    def _transform_dynamic_objects(objects, scaled_bev_center, rotation_matrix, bev_size):
        """Transform dynamic objects to BEV coordinates with efficient filtering."""
        N = objects.shape[0]
        transformed_objects = np.empty_like(objects, dtype=np.int32)
        mask = np.full(N, False)

        for i in range(N):
            for j in range(4):
                x, y = objects[i, j] - scaled_bev_center
                a = round(bev_size / 2 + rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y)
                b = round(bev_size / 2 + rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y)

                if 0 < a < bev_size and 0 < b < bev_size:
                    mask[i] = True

                transformed_objects[i, j, 0] = a
                transformed_objects[i, j, 1] = b

        return transformed_objects[mask]

    @staticmethod
    @jit(nopython=True)
    def _get_polygon_of_actor(yaw_deg, pixels_per_meter, extent_x, extent_y, location):
        """Create a rectangle representing an actor in the BEV image."""
        yaw_rad = np.radians(yaw_deg)
        rotation_matrix = np.array(
            [[np.cos(yaw_rad), -np.sin(yaw_rad)], [np.sin(yaw_rad), np.cos(yaw_rad)]], dtype=np.float32
        )

        # Define vehicle corners
        points = np.array(
            [[-extent_x, extent_y], [-extent_x, -extent_y], [extent_x, -extent_y], [extent_x, extent_y]],
            dtype=np.float32,
        )

        # Transform and scale
        points = points @ rotation_matrix.T + location
        points *= pixels_per_meter

        return points

    def _compute_dynamic_object_polygons(
        self, ego_location, all_other_actors, all_vehicles, all_walkers, all_bicycles
    ):
        """Compute polygons for all dynamic objects in the scene."""

        def add_actor(actor, target_list):
            if actor.get_location().distance(self._ego_vehicle.get_location()) > 60:
                return target_list

            transform = actor.get_transform()
            yaw_deg = -transform.rotation.yaw
            location = np.array([transform.location.x, -transform.location.y], dtype=np.float32)

            # Ensure minimum size for visibility
            extent_x = max(actor.bounding_box.extent.x, 1.0)
            extent_y = max(actor.bounding_box.extent.y, 1.0)

            target_list.append(
                self._get_polygon_of_actor(yaw_deg, self._pixels_per_meter, extent_x, extent_y, location)
            )
            return target_list

        # Initialize object lists
        vehicles, walkers, emergency_car, obstacle = [], [], [], []
        green_traffic_light, red_traffic_light = [], []

        # Process different actor types
        for actor in all_walkers:
            walkers = add_actor(actor, walkers)

        for actor in all_bicycles:
            obstacle = add_actor(actor, obstacle)

        for actor in all_vehicles:
            if actor.type_id.startswith("vehicle") and actor.id != self._ego_vehicle.id:  # Also includes trucks
                if "special_type" in actor.attributes and actor.attributes["special_type"] == "emergency":
                    emergency_car = add_actor(actor, emergency_car)
                else:
                    vehicles = add_actor(actor, vehicles)

        for actor in all_other_actors:
            if "role_name" in actor.attributes and "scenario" in actor.attributes["role_name"]:
                obstacle = add_actor(actor, obstacle)

        # Process stop signs within range
        indices = self._stop_signs_tree.query_ball_point(
            ego_location, r=self._maximum_distance_stop_signs_traffic_lights
        )
        stop_signs = self._stop_signs[indices]

        # Process traffic lights within range
        indices = self._traffic_lights_tree.query_ball_point(
            ego_location, r=self._maximum_distance_stop_signs_traffic_lights
        )
        for idx in indices:
            traffic_light = self._traffic_lights[idx]
            traffic_light_polygon = self._traffic_light_polygons[idx]

            if traffic_light.state == carla.TrafficLightState.Green:
                green_traffic_light.append(traffic_light_polygon)
            else:  # Red, Yellow, or Unknown
                red_traffic_light.append(traffic_light_polygon)

        # Convert to numpy arrays with proper shape
        return [
            np.array(vehicles, dtype=np.float32).reshape((-1, 4, 2)),
            np.array(walkers, dtype=np.float32).reshape((-1, 4, 2)),
            np.array(emergency_car, dtype=np.float32).reshape((-1, 4, 2)),
            np.array(obstacle, dtype=np.float32).reshape((-1, 4, 2)),
            np.array(green_traffic_light, dtype=np.float32).reshape((-1, 4, 2)),
            np.array(red_traffic_light, dtype=np.float32).reshape((-1, 4, 2)),
            stop_signs.reshape((-1, 4, 2)),
        ]

    def render(self, remaining_route, remaining_lanes, all_other_actors, all_vehicles, all_walkers, all_bicycles):
        """
        Render the bird's-eye view image.
        """
        # Up to 1,500 FPS with bev_size = 128

        # Subsample route and lanes for performance
        ego_route = remaining_route[::10, :2][:100].astype(np.float32)

        ego_lanes = remaining_lanes[::10][:100]
        ego_lanes = np.array([arr for sublist in ego_lanes for arr in sublist])
        ego_lanes = ego_lanes.astype(np.float32)[:, :2]

        # Calculate ego vehicle transformation
        yaw_deg = self._ego_vehicle.get_transform().rotation.yaw + 180.0
        rotation_matrix = self._get_rotation_matrix(yaw_deg)

        ego_location = self._ego_vehicle.get_location()
        bev_center = np.array([ego_location.x, -ego_location.y], dtype=np.float32)

        # Apply vertical offset
        center_offset = (
            self._get_rotation_matrix(-yaw_deg)
            @ np.array([-self._ego_vertical_offset, 0.0], dtype=np.float32)
            / self._pixels_per_meter
        )
        shifted_bev_center = bev_center + center_offset

        # Find relevant map tiles
        relevant_indices = self._grid_center_tree.query_ball_point(shifted_bev_center, self._max_render_distance)

        # Process map elements
        road_polygons, yellow_lines, white_lines = [], [], []
        scaled_shifted_bev_center = shifted_bev_center * self._pixels_per_meter

        # Process relevant grid tiles
        for idx in relevant_indices:
            roads, whites, yellows = self._grid_tiles[idx]
            road_polygons.extend(self._transform_points_batched(roads, rotation_matrix, scaled_shifted_bev_center))
            yellow_lines.extend(self._transform_points_batched(yellows, rotation_matrix, scaled_shifted_bev_center))
            white_lines.extend(self._transform_points_batched(whites, rotation_matrix, scaled_shifted_bev_center))

        # Transform route and lanes
        transformed_route = self._transform_route(
            scaled_shifted_bev_center, rotation_matrix, ego_route, self._pixels_per_meter, self._bev_size
        )
        transformed_lanes = self._transform_route(
            scaled_shifted_bev_center, rotation_matrix, ego_lanes, self._pixels_per_meter, self._bev_size
        )

        # Update dynamic objects history
        dynamic_objects = self._compute_dynamic_object_polygons(
            shifted_bev_center, all_other_actors, all_vehicles, all_walkers, all_bicycles
        )
        self._dynamic_objects.append(dynamic_objects)

        # Create BEV image
        # Think2Drive uses 34 channels, but since we'll compress them, the number of channels must be a multiple of 8
        bev_image = np.zeros((40, self._bev_size, self._bev_size), dtype=np.uint8)

        # Render static elements
        cv2.fillPoly(bev_image[0], road_polygons, 255)  # Roads
        cv2.polylines(bev_image[1], [transformed_route], False, 255, thickness=self._route_thickness)  # Route
        cv2.fillConvexPoly(bev_image[2], self._ego_polygon, 255)  # Ego vehicle
        cv2.polylines(bev_image[3], [transformed_lanes], False, 255, thickness=self._route_thickness)  # Lanes
        cv2.polylines(bev_image[4], yellow_lines, False, 255)  # Yellow lines
        cv2.polylines(bev_image[5], white_lines, False, 255)  # White lines

        # It seems they use an FPS of 20 and use the last 16 frames
        for t, idx in enumerate(self._history_indices):
            for i, objects in enumerate(self._dynamic_objects[idx]):
                if len(objects):
                    transformed_objects = self._transform_dynamic_objects(
                        objects, scaled_shifted_bev_center, rotation_matrix, self._bev_size
                    )
                    # Since fillPoly sometimes does not fill all boxes of traffic lights, we use fillConvexPoly here
                    for obj in transformed_objects:
                        cv2.fillConvexPoly(bev_image[6 + 7 * t + i], obj, 255)

        return bev_image.swapaxes(0, 2)  # Return as (height, width, channels)

    def visualize_bev(
        self,
        remaining_route,
        remaining_lanes,
        all_other_actors,
        all_vehicles,
        all_walkers,
        all_bicycles,
    ):
        """
        Visualize the bird's-eye view image using matplotlib.
        """
        frames = self.render(
            remaining_route, remaining_lanes, all_other_actors, all_vehicles, all_walkers, all_bicycles
        )
        factor = frames / 255.0

        # Color mapping for different channels
        color_mappings = np.array(
            [
                [50, 50, 50],  # Road
                [150, 150, 150],  # Route
                [255, 255, 255],  # Ego
                [100, 100, 100],  # Lane
                [255, 255, 0],  # Yellow lines
                [255, 0, 255],  # White lines
                # Dynamic objects at different time steps
                # t=0 (current)
                [0, 0, 230],  # Vehicle
                [0, 230, 230],  # Walker
                [230, 0, 0],  # Emergency car
                [230, 0, 0],  # Obstacle
                [0, 230, 0],  # Green traffic light
                [230, 0, 0],  # Red/Yellow traffic light
                [170, 170, 0],  # Stop sign
                # t=-5
                [50, 50, 230],  # Vehicle
                [50, 230, 230],  # Walker
                [230, 50, 50],  # Emergency car
                [230, 50, 50],  # Obstacle
                [50, 230, 50],  # Green traffic light
                [230, 50, 50],  # Red/Yellow traffic light
                [170, 170, 50],  # Stop sign
                # t=-10
                [100, 100, 230],  # Vehicle
                [100, 230, 230],  # Walker
                [230, 100, 100],  # Emergency car
                [230, 100, 100],  # Obstacle
                [100, 230, 100],  # Green traffic light
                [230, 100, 100],  # Red/Yellow traffic light
                [170, 170, 100],  # Stop sign
                # t=-15
                [150, 150, 230],  # Vehicle
                [150, 230, 230],  # Walker
                [230, 150, 150],  # Emergency car
                [230, 150, 150],  # Obstacle
                [150, 230, 150],  # Green traffic light
                [230, 150, 150],  # Red/Yellow traffic light
                [170, 170, 150],  # Stop sign
            ]
        )

        # Composite image with proper layer ordering
        bev_image = factor[:, :, 0, None] * color_mappings[0]

        # Render channels in specific order (background to foreground)
        render_order = [
            0,
            3,
            1,
            4,
            5,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            2,
        ]

        for i in render_order:
            channel_factor = factor[:, :, i, None]
            channel_color = frames[:, :, i, None] / 255.0 * color_mappings[i]
            bev_image = (1 - channel_factor) * bev_image + channel_factor * channel_color

        bev_image = np.round(bev_image).clip(0, 255).astype(np.uint8)

        plt.figure(figsize=(12, 12))
        plt.imshow(bev_image)
        plt.axis("off")
        plt.title("Bird's Eye View - CARLA Simulation")
        plt.show()
