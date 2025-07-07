'''
Gymnasium class for the CARLAEnv. Establishes communication with the env_agent and serves as gymnasium interface.
'''
import os
import math
import pathlib

import gymnasium as gym
from gymnasium import spaces
import zmq
import numpy as np


class CARLAEnv(gym.Env):
  '''
    Gymnasium environment class interface. Handles communication with env_agent.py
  '''

  metadata = {'render_modes': ['rgb_array']}

  def __init__(self, port, config, render_mode='rgb_array'):  # pylint: disable=locally-disabled, unused-argument

    self.observation_space = spaces.Dict({
        'bev_semantics':
            spaces.Box(0,
                       255,
                       shape=(config.obs_num_channels, config.bev_semantics_height, config.bev_semantics_width),
                       dtype=np.uint8),
        'measurements':
            spaces.Box(-math.inf, math.inf, shape=(config.obs_num_measurements,), dtype=np.float32),
        'value_measurements':
            spaces.Box(-math.inf, math.inf, shape=(config.num_value_measurements,), dtype=np.float32)
    })
    self.action_space = spaces.Box(config.action_space_min,
                                   config.action_space_max,
                                   shape=(config.action_space_dim,),
                                   dtype=np.float32)

    self.metadata['render_fps'] = config.frame_rate
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PAIR)
    self.port = port
    self.initialized = False
    self.config = config

  def reset(self, seed=None, options=None):  # pylint: disable=locally-disabled, unused-argument
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    if not self.initialized:
      # Connect to env_agent.
      current_folder = pathlib.Path(__file__).parent.resolve()
      comm_folder = os.path.join(current_folder, 'comm_files')
      pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
      communication_file = os.path.join(comm_folder, str(self.port))
      self.socket.bind(f'ipc://{communication_file}.lock')
      print(f'Connecting to leaderboard gym, port: {self.port}')

      msg = self.socket.recv_string()
      print(msg)
      self.initialized = True

    data = self.socket.recv_multipart(copy=False)
    observation = {
        'bev_semantics':
            np.frombuffer(data[0],
                          dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                  self.config.bev_semantics_width),
        'measurements':
            np.frombuffer(data[1], dtype=np.float32),
        'value_measurements':
            np.frombuffer(data[2], dtype=np.float32)
    }

    info = {'n_steps': np.frombuffer(data[6], dtype=np.int32), 'suggest': np.frombuffer(data[7], dtype=np.int32)}

    return observation, info

  def step(self, action):
    self.socket.send(action.tobytes(), copy=False)

    data = self.socket.recv_multipart(copy=False)

    observation = {
        'bev_semantics':
            np.frombuffer(data[0],
                          dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                  self.config.bev_semantics_width),
        'measurements':
            np.frombuffer(data[1], dtype=np.float32),
        'value_measurements':
            np.frombuffer(data[2], dtype=np.float32)
    }

    reward = np.frombuffer(data[3], dtype=np.float32).item()
    termination = np.frombuffer(data[4], dtype=bool).item()  # True if agent ended in destroy method.
    truncation = np.frombuffer(data[5], dtype=bool).item()  # True if agent timed out.

    info = {
        'n_steps': np.frombuffer(data[6], dtype=np.int32).item(),
        'suggest': np.frombuffer(data[7], dtype=np.int32).item()
    }

    return observation, reward, termination, truncation, info

  def close(self):
    print('Called close!')
