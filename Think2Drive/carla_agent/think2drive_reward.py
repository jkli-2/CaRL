import math
import numpy as np
from scipy.spatial.distance import cdist
import carla

import reward_criteria as criteria


class Think2DriveReward:

    # Configuration constants
    MAX_SPEED_MULTIPLIER = 0.8
    SAFETY_DISTANCES = {
        "vehicle": 8.0,  # meters
        "bicycle": 5.0,  # meters
        "pedestrian": 5.0,  # meters
    }
    STOPPING_DISTANCES = {
        "traffic_light": 4.0,  # meters
        "stop_sign": 2.5,  # meters
    }

    # Reward weights
    REWARD_WEIGHTS = {
        "speed": 1.0,
        "travel": 1.0,
        "deviation": 2.0,
        "steering": 0.5,
    }

    def __init__(self):
        """Initialize the reward system with default parameters."""
        # Initialize criteria evaluators
        self._collision = criteria.Collision()
        self._timeout = criteria.Timeout()
        self._route_deviation = criteria.RouteDeviation()
        self._route_completion = criteria.RouteCompletion()
        self._red_light = criteria.RunRedLight()
        self._stop_sign = criteria.RunStopSign()

        # Vehicle state tracking
        self._vehicle_extent_x = 0
        self._previous_steer = 0

        # CARLA objects
        self._vehicle = None
        self._carla_map = None
        self._world = None
        self._route_planner = None

    def reset(self, vehicle, carla_map, world, route_planner):
        """
        Reset the reward system for a new episode.
            vehicle: carla.Vehicle
            carla_map: carla.Map
            world: carla.World
            route_planner: route.RoutePlanner
        """
        self._vehicle = vehicle
        self._carla_map = carla_map
        self._world = world
        self._route_planner = route_planner

        # Reset all criteria evaluators
        self._collision.reset(vehicle, world)
        self._timeout.reset(vehicle)
        self._route_deviation.reset(vehicle, route_planner)
        self._route_completion.reset()
        self._red_light.reset(carla_map, vehicle, world)
        self._stop_sign.reset(carla_map, vehicle)

        # Cache vehicle properties
        self._vehicle_extent_x = self._vehicle.bounding_box.extent.x
        self._previous_steer = 0

    def step(self, remaining_route, distance_traveled, all_vehicles, all_walkers, all_bicycles):
        """
        Compute reward and episode status for the current simulation step.
        """

        # this operation takes a lot of time, so we execute it only once
        route_location = carla.Location(x=remaining_route[0, 0], y=remaining_route[0, 1], z=remaining_route[0, 2])
        route_wp = self._carla_map.get_waypoint(route_location)

        ego_location = self._vehicle.get_location()
        ego_wp = self._carla_map.get_waypoint(carla.Location(x=ego_location.x, y=ego_location.y, z=ego_location.z))

        # Filter out ego vehicle from other vehicles
        all_non_ego_vehicles = [v for v in all_vehicles if v.id != self._vehicle.id]

        # Evaluate all driving criteria
        violations = self._evaluate_driving_criteria(
            remaining_route, route_wp, ego_wp, all_non_ego_vehicles, all_walkers, all_bicycles
        )

        # Determine episode termination conditions
        # From the paper we can not infere in which cases they terminate the episode and when they truncate it. Hence we
        # estimate this being a logic assignment.
        termination = (
            violations["collision"]
            or violations["red_light"]
            or violations["stop_sign"]
            or violations["route_deviation"]
            or violations["timeout"]
        )
        truncation = violations["route_completed"]

        # Compute reward and target speed
        reward, target_speed = self._compute_reward(
            violations,
            distance_traveled,
            all_non_ego_vehicles,
            all_walkers,
            all_bicycles,
            remaining_route,
        )

        info = np.array(
            [
                violations["red_light"],
                violations["stop_sign"],
                violations["route_deviation"],
                violations["collision"],
                violations["timeout"],
                violations["route_completed"],
            ],
            dtype=bool,
        )

        return (reward, termination, truncation, info, violations["route_deviation_distance"], target_speed)

    def _evaluate_driving_criteria(
        self, remaining_route, route_wp, ego_wp, all_non_ego_vehicles, all_walkers, all_bicycles
    ):
        # Evaluate all driving criteria and return violation flags and distances.

        # Collision detection
        collision_happened, collision_actor_type = self._collision.step(
            all_non_ego_vehicles, all_walkers, all_bicycles
        )

        # Timeout detection
        timeout_reached, ticks_without_movement = self._timeout.step()

        # Route completion
        reached_route_end = self._route_completion.step(remaining_route)

        # Route deviation
        exceeded_route_deviation, route_deviation = self._route_deviation.step(remaining_route)

        # Traffic light violations (check both route and ego waypoints)
        run_red_light_route, dist_traffic_light_route = self._red_light.step(remaining_route, route_wp)
        run_red_light_ego, dist_traffic_light_ego = self._red_light.step(remaining_route, ego_wp)

        # Stop sign violations (check both route and ego waypoints)
        run_stop_sign_route, dist_stop_sign_route = self._stop_sign.step(remaining_route, route_wp)
        run_stop_sign_ego, dist_stop_sign_ego = self._stop_sign.step(remaining_route, ego_wp)

        return {
            "collision": collision_happened,
            "timeout": timeout_reached,
            "route_completed": reached_route_end,
            "route_deviation": exceeded_route_deviation,
            "route_deviation_distance": route_deviation,
            "red_light": run_red_light_route or run_red_light_ego,
            "stop_sign": run_stop_sign_route or run_stop_sign_ego,
            "distance_to_traffic_light": min(dist_traffic_light_route, dist_traffic_light_ego),
            "distance_to_stop_sign": min(dist_stop_sign_route, dist_stop_sign_ego),
        }

    def _compute_distance_to_leading_actors(
        self, all_non_ego_vehicles, all_pedestrians, all_bicycles, remaining_route
    ):
        """
        Compute distances to leading vehicles, pedestrians, and cyclists.
        """
        distance_to_leading_vehicle = float("inf")
        distance_leading_bicycle = float("inf")
        distance_leading_walker = float("inf")

        # Find leading vehicles on the planned route
        if all_non_ego_vehicles:
            vehicle_locations = np.array(
                [[v.get_location().x, v.get_location().y, v.get_location().z] for v in all_non_ego_vehicles]
            )

            # Compute distances to route points (subsample route for efficiency)
            route_subset = remaining_route[::5][:100]
            distances = cdist(vehicle_locations, route_subset, metric="euclidean")

            # Find vehicles close to the planned route
            min_route_distances = np.min(distances, axis=1)
            closest_route_indices = np.argmin(distances, axis=1)
            on_route_mask = min_route_distances < 2.0

            if np.any(on_route_mask):
                closest_indices = closest_route_indices[on_route_mask]
                leading_vehicle_idx = np.argmin(closest_indices)

                # 2 because the distance between route points after [::5] is 0.5m
                distance_to_leading_vehicle = float(closest_indices[leading_vehicle_idx]) / 2.0

        # Find closest pedestrians
        ego_location = self._vehicle.get_location()
        for pedestrian in all_pedestrians:
            pedestrian_distance = pedestrian.get_location().distance(ego_location)
            distance_leading_walker = min(distance_leading_walker, pedestrian_distance)

        # Find closest bicycles
        for bicycle in all_bicycles:
            bicycle_distance = bicycle.get_location().distance(ego_location)
            distance_leading_bicycle = min(distance_leading_bicycle, bicycle_distance)

        return distance_to_leading_vehicle, distance_leading_bicycle, distance_leading_walker

    def _compute_target_speed(self, distance_to_obstacle, required_stopping_distance):
        """
        Compute target speed based on distance to obstacle and required stopping distance.
        """
        # b=(v*3.6/10)^2/2 (Bernhard's formula)
        # Roach's max speed is 6 m/s => b=2.33 m
        # Multiply max speed by np.clip(dist_veh, 0.0, 5.0) / 5.0 => Reduce from 5/2.33 = 2.15 distance on
        # If we assume our highest speed is 50 m/s => b = 12.5 m => Reduce speed from 12.5 * 2.15 = 26.875 m on

        base_target_speed = (self._vehicle.get_speed_limit() / 3.6) * self.MAX_SPEED_MULTIPLIER

        # Apply speed reduction based on proximity to obstacle
        effective_distance = distance_to_obstacle - required_stopping_distance
        speed_reduction_factor = np.clip(effective_distance, 0.0, 12.5) / 12.5

        return base_target_speed * speed_reduction_factor

    def _compute_reward(
        self,
        violations,
        distance_traveled,
        all_non_ego_vehicles,
        all_pedestrians,
        all_bicycles,
        remaining_route,
    ):
        """
        Compute the reward for the current step based on driving performance.
        """
        current_speed = self._vehicle.get_velocity().length()

        # Compute distances to nearby actors
        distance_to_vehicle, distance_to_bicycle, distance_to_walker = self._compute_distance_to_leading_actors(
            all_non_ego_vehicles, all_pedestrians, all_bicycles, remaining_route
        )

        # Compute target speed considering all constraints
        base_speed = (self._vehicle.get_speed_limit() / 3.6) * self.MAX_SPEED_MULTIPLIER

        speed_constraints = [
            self._compute_target_speed(
                violations["distance_to_traffic_light"], self.STOPPING_DISTANCES["traffic_light"]
            ),
            self._compute_target_speed(violations["distance_to_stop_sign"], self.STOPPING_DISTANCES["stop_sign"]),
            self._compute_target_speed(distance_to_vehicle, self.SAFETY_DISTANCES["vehicle"]),
            self._compute_target_speed(distance_to_bicycle, self.SAFETY_DISTANCES["bicycle"]),
            self._compute_target_speed(distance_to_walker, self.SAFETY_DISTANCES["pedestrian"]),
        ]

        target_speed = min(base_speed, *speed_constraints)

        # Compute reward components
        current_steer = self._vehicle.get_control().steer

        # This works because the speed_limit is always 30 km/h at the beginning of each episode
        speed_reward = 1.0 - abs(current_speed - target_speed) / 7.5

        # Travel reward: encourage forward progress
        travel_reward = distance_traveled

        # Route deviation penalty
        deviation_penalty = -violations["route_deviation_distance"] / 8.0

        # Steering smoothness penalty
        steering_penalty = -abs(self._previous_steer - current_steer)
        self._previous_steer = current_steer

        # Combine reward components
        reward = (
            self.REWARD_WEIGHTS["speed"] * speed_reward
            + self.REWARD_WEIGHTS["travel"] * travel_reward
            + self.REWARD_WEIGHTS["deviation"] * deviation_penalty
            + self.REWARD_WEIGHTS["steering"] * steering_penalty
        )

        # Apply terminal state rewards/penalties
        if violations["route_deviation"] or violations["timeout"]:
            reward = -1.0
        elif violations["red_light"] or violations["stop_sign"] or violations["collision"]:
            reward = -1.0 - current_speed  # Larger penalty for higher speed violations
        elif violations["route_completed"]:
            reward = 1.0  # Success bonus

        return reward, target_speed

    def destroy(self):
        """Clean up resources and destroy criteria evaluators."""
        self._collision.destroy()
        self._timeout.destroy()
        self._route_deviation.destroy()
        self._route_completion.destroy()
        self._red_light.destroy()
        self._stop_sign.destroy()
