import os

import carla
import zmq

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import leaderboard.autoagents.agent_wrapper as agent_wrapper

from think2drive_reward import Think2DriveReward
from think2drive_observation_manager import ObservationManager
from route_planner import RoutePlanner


def get_entry_point() -> str:
    """Entry point function required by CARLA leaderboard framework."""
    return "CarlaAgent"


class CarlaAgent(AutonomousAgent):
    """
    This agent manages the CARLA simulation environment and communicates with the Dreamer implemenation. It handles observations, rewards, route planning, and vehicle control.
    """

    def __init__(self, carla_host: str, carla_port: int, debug: bool = False):
        # The constructor is only called once at the very beginning of the
        # first episode
        super().__init__(carla_host, carla_port, debug)
        self.track = Track.MAP

        # ZeroMQ communication
        self._established_connection = False
        self._context = None
        self._socket = None

        self._eval = os.getenv("EVALUATION", "False").lower() == "true"
        self._action_repeat = 2
        self._maximum_height_difference_actors = 8

        # Core components
        self._route_planner = RoutePlanner()
        self._reward = Think2DriveReward()
        self._observation_manager = ObservationManager(pixels_per_meter=2)

        self._carla_steps = -1
        self._control = None
        self._traveled_distance = 0.0

        # CARLA entities (initialized in setup)
        self._vehicle = None
        self._world = None
        self._carla_map = None

    def setup(self, exp_folder, port):
        # This method is called at the beginning of each episode to prepare
        # the agent
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = CarlaDataProvider.get_world()
        self._carla_map = CarlaDataProvider.get_map()

        # Reset all components
        self._route_planner.reset(self.dense_global_plan_world_coord, self._vehicle, self._carla_map)
        self._reward.reset(self._vehicle, self._carla_map, self._world, self._route_planner)
        self._observation_manager.reset(self._vehicle, self._world, self._carla_map)

        self._carla_steps = -1
        self._control = None
        self._traveled_distance = 0.0

    def _establish_connection(self):
        # Establish ZeroMQ connection
        ipc_file = os.getenv("IPC_FILE", "ipc_test")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.connect(f"ipc:///tmp/{ipc_file}")

        self._config_dict = self._socket.recv_pyobj()
        self._socket.send_pyobj("Connected to CarlaAgent client")

        # This action is not used, because we automatically send the first observation to dreamer
        _ = self._socket.recv_pyobj()

        self._established_connection = True

    def sensors(self):
        return []

    def _compute_actor_lists(self):
        # Categorize and filter actors in the world by type and proximity.
        all_actors = self._world.get_actors()
        ego_height = self._vehicle.get_location().z

        all_vehicles, all_walkers, all_bicycles, all_other_actors = [], [], [], []

        for actor in all_actors:
            height_diff = abs(actor.get_location().z - ego_height)
            if height_diff >= self._maximum_height_difference_actors:
                continue

            actor_type = actor.type_id.lower()
            base_type = actor.attributes.get("base_type", "").lower()
            if base_type == "bicycle":
                all_bicycles.append(actor)
            elif actor_type.startswith("walker"):
                all_walkers.append(actor)
            elif actor_type.startswith("vehicle"):  # Also includes trucks
                all_vehicles.append(actor)
            else:
                all_other_actors.append(actor)

        return all_other_actors, all_vehicles, all_walkers, all_bicycles

    def run_step(self, input_data, timestamp, sensors=None, last_step=False):
        # Execute one step and send the control command to the simulator

        # Establish connection if needed
        if not self._established_connection:
            self._establish_connection()

        self._carla_steps += 1

        all_other_actors, all_vehicles, all_walkers, all_bicycles = self._compute_actor_lists()

        remaining_route, remaining_lanes, traveled_distance = self._route_planner.step()
        self._traveled_distance += traveled_distance
        observation = self._observation_manager.step(
            remaining_route, remaining_lanes, all_other_actors, all_vehicles, all_walkers, all_bicycles
        )

        if self._carla_steps % self._action_repeat == 0 or last_step:
            reward, termination, truncation, info, route_deviation, target_speed = self._reward.step(
                remaining_route, self._traveled_distance, all_vehicles, all_walkers, all_bicycles
            )
            self._traveled_distance = 0.0

            # Handle episode termination
            if last_step:
                termination, truncation = True, True
            elif self._eval:
                # In evaluation mode, only leaderboard can terminate episodes
                termination, truncation = False, False

            data = {
                "observation": observation,
                "reward": reward,
                "termination": termination,
                "truncation": truncation,
                "info": info,
            }

            self._socket.send_pyobj(data)

            if not last_step:
                action = self._socket.recv_pyobj()

                if action[0] == "reset":
                    if self._eval:
                        # Do not allow dreamer to end the episode but only the leaderboard evaluator
                        action = ["step", (0, 0, 0)]
                    else:
                        raise agent_wrapper.NextRoute("Episode ended.")

                throttle, brake, steer = action[1]
                self._control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        return self._control

    def destroy(self, results=None):
        if self._eval:
            self.run_step(input_data=None, timestamp=0, last_step=True)

        self._route_planner.destroy()
        self._reward.destroy()
        self._observation_manager.destroy()

    def __del__(self):
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
