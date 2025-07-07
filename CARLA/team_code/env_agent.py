"""
Connects with a carla gym wrapper.
"""
# Python imports
import os
import math
import pathlib
import random
from collections import deque

# Carla leaderboard imports
from leaderboard.autoagents import autonomous_agent
from leaderboard.autoagents.agent_wrapper import NextRoute
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# Pip imports
from PIL import Image
import zmq
import numpy as np
import carla
from pytictoc import TicToc
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# Code imports
from reward.roach_reward import RoachReward
from reward.simple_reward import SimpleReward
from rl_config import GlobalConfig
import rl_utils as rl_u
from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.bev_observation import ObsManager as ObsManager2
from birds_eye_view.run_stop_sign import RunStopSign
from nav_planner import RoutePlanner

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'EnvAgent'


t = TicToc()
init_tictoc = False


class EnvAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def __init__(self, carla_host, carla_port, debug=False):
    """ Init only gets called once during the whole training in custom_leaderboard"""
    super().__init__(carla_host, carla_port, debug)

    self.track = autonomous_agent.Track.MAP
    self.config = GlobalConfig()
    self.initialized_global = False

    # Environment variables
    self.save_path = os.environ.get('SAVE_PATH')

  def setup(self, exp_folder, port):
    """Sets up the agent. is called with every new route."""
    self.step = -1
    self.port = port
    self.exp_folder = exp_folder
    self.termination = False
    self.truncation = False
    self.data = None
    self.last_timestamp = 0.0
    self.last_control = None
    self.list_traffic_lights = []
    self.initialized_route = False

  def sensors(self):
    sensors = []

    return sensors

  def agent_global_init(self):
    #  Socket to talk to server
    print(f'Connecting to gymnasium server, port: {self.port}')
    self.context = zmq.Context()
    conf_socket = self.context.socket(zmq.PAIR)
    current_folder = pathlib.Path(__file__).parent.resolve()
    comm_folder = os.path.join(current_folder, 'comm_files')
    pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
    communication_file = os.path.join(comm_folder, str(self.port))
    # Connect to python process receiving up to date config file.
    conf_socket.connect(f'ipc://{communication_file}.conf_lock')
    json_config = conf_socket.recv_string(
    )  # Overwrite default config with the configured one from the training process
    loaded_config = jsonpickle.decode(json_config)
    self.config.__dict__.update(loaded_config.__dict__)
    conf_socket.send_string(f'Config received port: {self.port}')

    # Connect to env gym to send observations
    self.socket = self.context.socket(zmq.PAIR)
    self.socket.connect(f'ipc://{communication_file}.lock')
    self.socket.send_string(f'Connected to env_agent client. {self.port}')

    self.config.debug = int(os.environ.get('DEBUG_ENV_AGENT', 0)) == 1
    if self.config.use_new_bev_obs:
      self.bev_semantics_manager = ObsManager2(self.config)
    else:
      self.bev_semantics_manager = ObsManager(self.config)

    conf_socket.close()
    self.initialized_global = True

  def agent_route_init(self):
    self.vehicles_all = []
    self.walkers_all = []
    self.vehicle = CarlaDataProvider.get_hero_actor()
    self.world = self.vehicle.get_world()
    settings = self.world.get_settings()
    # If this triggers you started the leaderboard client with a different FPS than specified in the config.
    assert math.isclose(settings.fixed_delta_seconds, 1.0 / self.config.frame_rate)
    self.world_map = CarlaDataProvider.get_map()
    self.stop_sign_criteria = RunStopSign(self.world, self.world_map)
    if self.config.use_new_bev_obs:
      self.bev_semantics_manager.attach_ego_vehicle(self.vehicle, self.stop_sign_criteria, self.world_map,
                                                    self.dense_global_plan_world_coord)
    else:
      self.bev_semantics_manager.attach_ego_vehicle(self.vehicle, self.stop_sign_criteria, self.world_map)

    self.close_traffic_lights = []

    # Preprocess traffic lights
    all_actors = self.world.get_actors()
    all_traffic_lights = all_actors.filter('*traffic_light*')
    for actor in all_traffic_lights:
      center, waypoints = rl_u.get_traffic_light_waypoints(actor, self.world_map)
      self.list_traffic_lights.append((actor, center, waypoints))

    if self.config.reward_type == 'roach':
      self.reward_handler = RoachReward(self.vehicle, self.world_map, self.world, self.config)
    elif self.config.reward_type == 'simple_reward':
      self.reward_handler = SimpleReward(self.vehicle, self.world_map, self.world, self.config,
                                         self.dense_global_plan_world_coord)
    else:
      raise ValueError('Selected reward type is not implemented.')

    self.route_planner = RoutePlanner()
    self.route_planner.set_route(self.dense_global_plan_world_coord)
    self.total_route_len = len(self.dense_global_plan_world_coord)

    # In some towns TL are red for a very long time and green for a short amount of time.
    # To balance this we set traffic lights on a route to green when the agent arrives, with a certain prob. per route
    self.active_green_wave = False
    if self.config.use_green_wave:
      random_number = random.uniform(0, 1)
      if random_number < self.config.green_wave_prob:
        self.active_green_wave = True

    if self.config.use_extra_control_inputs:
      self.last_wheel_angle = 0.0
      self.past_wheel_errors = deque([0.0 for _ in range(int(1.0 * self.config.frame_rate))],
                                     maxlen=int(1.0 * self.config.frame_rate))

    self.initialized_route = True

  def preprocess_observation(self, waypoint_route, timestamp):
    self.stop_sign_criteria.tick(self.vehicle)
    actors = self.world.get_actors()
    self.vehicles_all = actors.filter('*vehicle*')
    self.walkers_all = actors.filter('*walker*')
    self.static_all = actors.filter('*static*')
    # TODO render background vehicles
    # for actor in world.get_environment_objects(carla.CityObjectLabel.Car):
    #   static_vehicles.append(actor)
    bev_semantics = self.bev_semantics_manager.get_observation(waypoint_route,
                                                               self.vehicles_all,
                                                               self.walkers_all,
                                                               self.static_all,
                                                               debug=self.config.debug)
    observations = {'bev_semantics': bev_semantics['bev_semantic_classes']}

    if self.config.debug:
      Image.fromarray(bev_semantics['rendered']).save(self.save_path + (f'/{self.step:04}.png'))

    last_control = self.vehicle.get_control()
    velocity = self.vehicle.get_velocity()
    transform = self.vehicle.get_transform()
    forward_vec = transform.get_forward_vector()

    np_vel = np.array([velocity.x, velocity.y, velocity.z])
    np_fvec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
    forward_speed = np.dot(np_vel, np_fvec)

    np_vel_2d = np.array([velocity.x, velocity.y])
    velocity_ego_frame = rl_u.inverse_conversion_2d(np_vel_2d, np.zeros(2), np.deg2rad(transform.rotation.yaw))

    # acceleration = self.vehicle.get_acceleration()
    # np_acceleration_2d = np.array([acceleration.x, acceleration.y])
    # acc_ego_frame = rl_u.inverse_conversion_2d(np_acceleration_2d, np.zeros(2), np.deg2rad(transform.rotation.yaw))

    speed_limit = self.vehicle.get_speed_limit()
    if isinstance(speed_limit, float):
      # Speed limit is in km/h we compute with m/s, so we convert it by / 3.6
      maximum_speed = speed_limit / 3.6
    else:
      #  Car can have no speed limit right after spawning
      maximum_speed = self.config.rr_maximum_speed

    measurements = [
        last_control.steer, last_control.throttle, last_control.brake,
        float(last_control.gear),
        float(velocity_ego_frame[0]),
        float(velocity_ego_frame[1]),
        float(forward_speed), maximum_speed
    ]

    if self.config.use_extra_control_inputs:
      left_wheel = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
      right_wheel = self.vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
      avg_wheel = 0.5 * (left_wheel + right_wheel)  # They can be quite different, we take the avg to simplify.
      avg_wheel /= self.config.max_avg_steer_angle  # Normalize from range [-60, 60] to [-1, 1]

      measurements.append(avg_wheel)

      last_error = last_control.steer - self.last_wheel_angle

      self.past_wheel_errors.append(last_error)
      # I am omitting the time step because it is constant, normalizes the input automatically in [-1, 1]
      error_derivative = self.past_wheel_errors[-1] - self.past_wheel_errors[-2]
      error_integral = sum(self.past_wheel_errors) / len(self.past_wheel_errors)

      # These inputs should allow the model to learn something like a PID controller for steering.
      measurements.append(last_error)
      measurements.append(error_derivative)
      measurements.append(error_integral)

      self.last_wheel_angle = avg_wheel

    if self.config.use_target_point:
      measurements.append(bev_semantics['target_point'][0])
      measurements.append(bev_semantics['target_point'][1])

    observations['measurements'] = np.array(measurements, dtype=np.float32)
    # Add remaining time till timeout. remaining time till blocked, remaining route to help return prediction
    remaining_time = (float(self.config.eval_time) - timestamp) / float(self.config.eval_time)
    time_till_blocked = self.reward_handler.block_detector.time_till_blocked
    perc_route_left = float(len(waypoint_route)) / 100.0  # 100.0 is just some constant for normalization
    if self.config.use_ttc:
      remaining_ttc_penalty_ticks = self.reward_handler.remaining_ttc_penalty_ticks / self.config.ttc_penalty_ticks
    if self.config.use_comfort_infraction:
      remaining_comfort_penalty_ticks = (self.reward_handler.remaining_comfort_penalty_ticks /
                                         self.config.comfort_penalty_ticks)

    if not self.config.use_value_measurements:
      remaining_time = 0.0
      time_till_blocked = 0.0
      perc_route_left = 0.0
      if self.config.use_ttc:
        remaining_ttc_penalty_ticks = 0.0
      if self.config.use_comfort_infraction:
        remaining_comfort_penalty_ticks = np.zeros(6)

    value_measurements = [remaining_time, time_till_blocked, perc_route_left]
    if self.config.use_ttc:
      value_measurements.append(remaining_ttc_penalty_ticks)
    if self.config.use_comfort_infraction:
      value_measurements.extend(remaining_comfort_penalty_ticks)

    assert self.config.num_value_measurements == len(value_measurements)

    observations['value_measurements'] = np.array(value_measurements, dtype=np.float32)

    collision_with_pedestrian = bev_semantics['collision_px']
    perc_off_road = bev_semantics['percentage_off_road']

    return observations, collision_with_pedestrian, perc_off_road

  def get_waypoint_route(self):
    ego_vehicle_transform = self.vehicle.get_transform()
    pos = ego_vehicle_transform.location
    pos = np.array([pos.x, pos.y])
    waypoint_route = self.route_planner.run_step(pos)

    return waypoint_route

  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument

    global init_tictoc
    if init_tictoc:
      t.toc(msg='Time for reset:')
    init_tictoc = False

    self.step += 1
    self.last_timestamp = timestamp

    if not self.initialized_global:
      self.agent_global_init()
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.last_control = control
      return control

    if not self.initialized_route:
      self.agent_route_init()
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.last_control = control
      return control

    if self.step < self.config.start_delay_frames:
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      return control

    if self.step % self.config.action_repeat != 0:
      return self.last_control

    # In some towns TL are red for a very long time and green for a short amount of time.
    # To balance this we set traffic lights on a route to green when the agent arrives, with a certain prob. per route
    if self.config.use_green_wave and self.active_green_wave:
      affecting_tl = self.vehicle.get_traffic_light()
      if affecting_tl is not None:
        affecting_tl.set_state(carla.TrafficLightState.Green)

    waypoint_route = self.get_waypoint_route()
    obs, collision_with_pedestrian, perc_off_road = self.preprocess_observation(waypoint_route, timestamp)
    reward, termination, truncation, exploration_suggest = self.reward_handler.get(timestamp, waypoint_route,
                                                                                   collision_with_pedestrian,
                                                                                   self.vehicles_all, self.walkers_all,
                                                                                   self.static_all, perc_off_road)
    data = {
        'observation': obs,
        'reward': reward,
        'termination': termination,
        'truncation': truncation,
        'info': exploration_suggest
    }
    if termination or truncation:
      self.termination = termination
      self.truncation = truncation
      self.data = data
      # Will terminate the route, call destroy and start the next one.
      if not init_tictoc:
        t.tic()
        init_tictoc = True
      raise NextRoute('Episode ended by roach reward.')
    # Send observation to training server
    self.socket.send_multipart(
        (data['observation']['bev_semantics'], data['observation']['measurements'],
         data['observation']['value_measurements'], np.array(data['reward'], dtype=np.float32),
         np.array(data['termination'], dtype=bool), np.array(data['truncation'], dtype=bool),
         np.array(data['info']['n_steps'], dtype=np.int32), np.array(data['info']['suggest'], dtype=np.int32)),
        copy=False)
    #  Receive next action from training server
    action = np.frombuffer(self.socket.recv(copy=False), dtype=np.float32)

    control = self.convert_action_to_control(action)
    self.last_control = control

    return control

  def convert_action_to_control(self, action):
    # Convert acceleration to brake / throttle. Acc in [-1,1]. Negative acceleration -> brake
    if action[1] > 0.0:
      throttle = action[1]
      brake = 0.0
    else:
      throttle = 0.0
      brake = -action[1]
    control = carla.VehicleControl(steer=float(action[0]), throttle=float(throttle), brake=float(brake))
    return control

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    """
    if self.termination or self.truncation:
      data = self.data
    else:
      print('Leaderboard ended episode.')
      waypoint_route = self.get_waypoint_route()
      obs, collision_with_pedestrian, perc_off_road = self.preprocess_observation(waypoint_route, self.last_timestamp)
      reward, termination, _, exploration_suggest = self.reward_handler.get(self.last_timestamp, waypoint_route,
                                                                            collision_with_pedestrian,
                                                                            self.vehicles_all, self.walkers_all,
                                                                            self.static_all, perc_off_road)
      # We define leaderboard termination as a truncation for the roach reward.
      term = False
      trunc = True
      if termination:
        term = True
        trunc = False

      data = {
          'observation': obs,
          'reward': reward,
          'termination': term,
          'truncation': trunc,
          'info': exploration_suggest
      }
    # Send observation to training server
    self.socket.send_multipart(
        (data['observation']['bev_semantics'], data['observation']['measurements'],
         data['observation']['value_measurements'], np.array(data['reward'], dtype=np.float32),
         np.array(data['termination'], dtype=bool), np.array(data['truncation'], dtype=bool),
         np.array(data['info']['n_steps'], dtype=np.int32), np.array(data['info']['suggest'], dtype=np.int32)),
        copy=False)
    self.reward_handler.destroy()
    # Cleanup route level variables:
    del self.vehicle
    del self.world
    del self.world_map
    del self.stop_sign_criteria
    del self.close_traffic_lights
    del self.reward_handler
    del self.route_planner
    del self.step
    del self.port
    del self.exp_folder
    del self.data
    del self.last_timestamp
    del self.last_control
    del self.list_traffic_lights
    del self.initialized_route
    del self.vehicles_all
    del self.walkers_all

    self.termination = False
    self.truncation = False
