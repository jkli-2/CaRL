import carla
import numpy as np
from shapely import Polygon
from scipy.spatial.distance import cdist


class Collision:
    """
    The collision detection system for a CARLA simulation environment.
    """

    def __init__(self, distance_threshold=15.0):
        self._distance_threshold = distance_threshold

        # Vehicle and world references
        self._vehicle = None
        self._carla_world = None

        # Collision sensor
        self._collision_sensor = None

        # Collision state tracking
        self._collision_happened = False
        self._collision_actor_type = None

    def reset(self, vehicle, carla_world):
        """
        Reset the collision detector with a new vehicle and world.
        """

        self._vehicle = vehicle
        self._carla_world = carla_world

        # Create and attach collision sensor
        blueprint = self._carla_world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._carla_world.spawn_actor(blueprint, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(self._on_collision)

        # Reset collision state
        self._collision_happened = False
        self._collision_actor_type = None

    def _on_collision(self, event):
        """
        Callback function for collision sensor events.
        """

        self._collision_happened = True
        self._collision_actor_type = event.other_actor.type_id

    def step(self, all_non_ego_vehicles, all_pedestrians, all_bicycles):
        # Check geometric bounding box intersections
        bounding_box_collision = self._compute_bounding_box_intersection(
            all_non_ego_vehicles, all_pedestrians, all_bicycles
        )

        # Combine sensor and geometric collision detection
        collision_occurred = self._collision_happened or bounding_box_collision
        collision_type = self._collision_actor_type

        # Reset sensor collision flag for next step
        self._collision_happened = False

        if collision_occurred:
            print(f"Collision detected with actor type: {collision_type}", flush=True)

        return collision_occurred, collision_type

    def _get_polygon_of_actor(self, yaw_deg, extent_x, extent_y, location):
        # Create a 2D polygon representation of an actor's bounding box.
        yaw_rad = np.radians(yaw_deg)
        rotation_matrix = np.array(
            [[np.cos(yaw_rad), -np.sin(yaw_rad)], [np.sin(yaw_rad), np.cos(yaw_rad)]], dtype=np.float32
        )

        # Define bounding box corners relative to center
        corners = np.array(
            [
                [-extent_x, extent_y],  # Front-left
                [-extent_x, -extent_y],  # Rear-left
                [extent_x, -extent_y],  # Rear-right
                [extent_x, extent_y],  # Front-right
            ],
            dtype=np.float32,
        )

        # Apply rotation and translation
        rotated_corners = corners @ rotation_matrix.T + location

        return rotated_corners

    def _compute_bounding_box_intersection(self, all_non_ego_vehicles, all_pedestrians, all_bicycles):
        """
        Check for bounding box intersections between ego vehicle and nearby actors.
        """

        # Get ego vehicle position
        ego_location = self._vehicle.get_location()
        ego_location_2d = np.array([ego_location.x, ego_location.y])

        # Combine all relevant actors
        all_relevant_actors = all_non_ego_vehicles + all_pedestrians + all_bicycles

        if not all_relevant_actors:
            return False

        # Calculate distances to all actors for performance optimization
        actor_locations = np.array([[actor.get_location().x, actor.get_location().y] for actor in all_relevant_actors])

        distances = cdist([ego_location_2d], actor_locations, metric="euclidean")[0]
        nearby_mask = distances < self._distance_threshold

        # Create ego vehicle polygon
        ego_transform = self._vehicle.get_transform()
        ego_extent = self._vehicle.bounding_box.extent
        ego_polygon_points = self._get_polygon_of_actor(
            ego_transform.rotation.yaw, ego_extent.x, ego_extent.y, ego_location_2d
        )
        ego_polygon = Polygon(ego_polygon_points)

        # Check intersections with nearby actors
        for actor, is_nearby in zip(all_relevant_actors, nearby_mask):
            if not is_nearby:
                continue

            # Get actor properties
            actor_transform = actor.get_transform()
            actor_extent = actor.bounding_box.extent
            actor_location = actor.get_location()
            actor_location_2d = np.array([actor_location.x, actor_location.y])

            # Since we render them being at least 1m wide and long, we also check for that
            extent_x = max(actor_extent.x, 1.0)
            extent_y = max(actor_extent.y, 1.0)

            # Create actor polygon
            actor_polygon_points = self._get_polygon_of_actor(
                actor_transform.rotation.yaw, extent_x, extent_y, actor_location_2d
            )
            actor_polygon = Polygon(actor_polygon_points)

            # Check for intersection
            if ego_polygon.intersects(actor_polygon):
                return True

        return False

    def destroy(self):
        """
        Clean up resources and destroy the collision sensor.
        """

        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None

    def __del__(self):
        self.destroy()
