import embodied
import numpy as np
import subprocess
import os
import time
import socket
import zmq
import pathlib
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools
from functools import wraps


class SimulationTimeoutError(Exception):
    """Custom exception raised when simulation operations exceed timeout limits."""

    pass


def timeout(seconds):
    """
    Decorator to add timeout functionality to methods.

    Args:
        seconds: The timeout duration in seconds.
    """

    def decorator(function):
        @wraps(function)
        def function_with_timeout(*args, **kwargs):
            def raise_timeout_error(signum, frame):
                raise SimulationTimeoutError(f"Function call timed out after {seconds} seconds")

            # Set the signal handler and alarm
            signal.signal(signal.SIGALRM, raise_timeout_error)
            signal.alarm(seconds)

            try:
                result = function(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            return result

        return function_with_timeout

    return decorator


class Carla(embodied.Env):

    # Default training routes directory
    PRETRAINING_ROUTES_DIR = "../../CARLA/custom_leaderboard/leaderboard/data/think2drive_pretrain"
    POSTTRAINING_ROUTES_DIR = "../../CARLA/custom_leaderboard/leaderboard/data/think2drive_posttrain"

    # CARLA control actions: (throttle, brake, steer)
    AVAILABLE_ACTIONS = (
        (0.0, 1.0, 0.0),  # Full brake
        (0.7, 0.0, -0.5),  # Moderate throttle, left steer
        (0.7, 0.0, -0.3),
        (0.7, 0.0, -0.2),
        (0.7, 0.0, -0.1),
        (0.7, 0.0, 0.0),  # Moderate throttle, straight
        (0.7, 0.0, 0.1),
        (0.7, 0.0, 0.2),
        (0.7, 0.0, 0.3),
        (0.7, 0.0, 0.5),  # Moderate throttle, right steer
        (0.3, 0.0, -0.7),  # Low throttle, sharp left steer
        (0.3, 0.0, -0.5),
        (0.3, 0.0, -0.3),
        (0.3, 0.0, -0.2),
        (0.3, 0.0, -0.1),
        (0.3, 0.0, 0.0),  # Low throttle, straight
        (0.3, 0.0, 0.1),
        (0.3, 0.0, 0.2),
        (0.3, 0.0, 0.3),
        (0.3, 0.0, 0.5),
        (0.3, 0.0, 0.7),  # Low throttle, sharp right steer
        (0.0, 0.0, -1.0),  # No throttle, full left steer
        (0.0, 0.0, -0.6),
        (0.0, 0.0, -0.3),
        (0.0, 0.0, -0.1),
        (0.0, 0.0, 0.0),  # No throttle, straight
        (0.0, 0.0, 0.1),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, 0.6),
        (0.0, 0.0, 1.0),  # No throttle, full right steer
    )

    def __init__(
        self,
        task,
        carla_installation_path,
        image_size,
        index=None,
        eval=False,
        results_directory="../carla_leaderboard_checkpoints/training",
        eval_routes_queue=None,
        num_envs=1,
        seed=None,
        render_off_screen=True,
        **kwargs,
    ):

        if index == 0:
            self._print_configuration(
                task, carla_installation_path, image_size, index, eval, results_directory, num_envs, seed
            )

        self.pretraining = kwargs["pretraining"]

        # Initialize process and communication variables
        self._carla_process = None
        self._leaderboard_pid = None
        self._socket = None
        self._context = None

        # Route and evaluation state
        self._route_file = None
        self._result_index = None
        self._save_frames = True

        # Configuration
        self._image_size = image_size
        self._carla_installation_path = carla_installation_path
        self._simulator_instance_index = index
        self._seed = seed
        self._render_option = "-RenderOffScreen -nullrhi" if render_off_screen else ""

        # Evaluation vs training setup
        self._eval = eval
        self._results_directory = results_directory

        if self._eval:
            self._eval_routes_queue = eval_routes_queue
            self._num_envs = num_envs
        else:
            self._training_routes = self._load_training_routes()

        # Communication and connection state
        self._ipc_file = f"inner{np.random.randint(2 ** 32)}"
        self._established_connection = False

    def _print_configuration(
        self,
        task,
        carla_path,
        image_size,
        index,
        eval_mode,
        results_dir,
        num_envs,
        seed,
    ):
        """Print environment configuration."""
        print(f"Task: {task}")
        print(f"CARLA Installation Path: {carla_path}")
        print(f"Image Size: {image_size}")
        print(f"Index: {index}")
        print(f"Evaluation Mode: {eval_mode}")
        print(f"Results Directory: {results_dir}")
        print(f"Number of Environments: {num_envs}")
        print(f"Seed: {seed}")

    def _load_training_routes(self):
        """Load and sort training routes from the default directory."""
        training_routes_directory = self.PRETRAINING_ROUTES_DIR if self.pretraining else self.POSTTRAINING_ROUTES_DIR
        return list(
            sorted(
                [
                    os.path.abspath(os.path.join(training_routes_directory, x))
                    for x in os.listdir(training_routes_directory)
                ]
            )
        )

    @property
    def obs_space(self):
        """
        Define the observation space for the CARLA environment.

        Returns:
            Dictionary mapping observation names to their spaces.
        """
        image_shape = self._image_size + (34,)  # 34-channel semantic BEV image

        return {
            "image": embodied.Space(np.uint8, image_shape),
            "scalars": embodied.Space(np.float32, 20),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @property
    def act_space(self):
        """
        Define the action space for the CARLA environment.

        Returns:
            Dictionary mapping action names to their spaces.
        """
        return {
            "action": embodied.Space(np.int32, (), 0, len(self.AVAILABLE_ACTIONS)),
            "reset": embodied.Space(bool),
        }

    def _create_default_sample(self):
        """
        Create a default sample for error cases (e.g., crashes, timeouts).
        """
        default_sample = {key: item.sample() for key, item in self.obs_space.items()}
        default_sample["image"] = np.zeros((128, 128, 40), dtype=np.uint8)
        default_sample["scalars"] *= 0
        default_sample["reward"] = np.float32(0.0)
        default_sample["is_first"] = False
        default_sample["is_last"] = True
        default_sample["is_terminal"] = True
        default_sample["log/carla_crashed"] = True

        # Initialize all CARLA-specific log flags
        carla_events = [
            "carla_run_red_light",
            "carla_run_stop_sign",
            "carla_exceed_route_dev",
            "carla_collision",
            "carla_timeout",
            "carla_route_end",
        ]
        for event in carla_events:
            default_sample[f"log/{event}"] = False

        # Add evaluation-specific metadata
        if self._eval:
            default_sample["log/route_file"] = self._route_file
            default_sample["log/seed"] = self._seed
            default_sample["log/result_index"] = self._result_index
            default_sample["log/save_frames"] = self._save_frames

        return default_sample

    @timeout(120)
    def _start_carla_with_timeout(self):
        """Start CARLA server and leaderboard processes with timeout protection."""
        # Clean up existing processes
        if self._leaderboard_pid:
            self._terminate_process_and_children(self._leaderboard_pid)
        if self._carla_process:
            self._terminate_process_and_children(self._carla_process.pid)

        # Start CARLA server
        carla_rpc_port = self._get_free_port()
        print(f"Starting CARLA on port: {carla_rpc_port}")

        carla_command = (
            f"bash {self._carla_installation_path}/CarlaUE4.sh "
            f"-carla-rpc-port={carla_rpc_port} -nosound "
            f"-carla-streaming-port=0 {self._render_option} -graphicsadapter=0"
        )

        with open(os.devnull, "w", encoding="utf8") as devnull:
            self._carla_process = subprocess.Popen(
                carla_command,
                shell=True,
                start_new_session=True,
                stdout=devnull,  # Redirect stdout to devnull
                stderr=devnull,  # Redirect stderr to devnull
            )

        if self._eval:
            # Wait 20 seconds to ensure the CARLA simulator has started
            # This is only required during evaluation. During training this is handled by our modified CARLA leaderboard module.
            time.sleep(20)

        # Start leaderboard process
        tm_port = self._get_free_port()  # Traffic Manager port
        leaderboard_command = self._generate_leaderboard_command(carla_rpc_port, tm_port)
        # bash start_leaderboard_own.sh /home/jens/Desktop/master_thesis/master_thesis_rl/custom_leaderboard/leaderboard/data/roach_preprocessed_routes6/route_Town01_00.xml.gz results 0 2000 8000 42 inner1 False
        print(f"Leaderboard command: {leaderboard_command}", flush=True)

        with open(os.devnull, "w", encoding="utf8") as devnull:
            leaderboard_process = subprocess.Popen(
                leaderboard_command,
                shell=True,
                start_new_session=True,
                cwd="../carla_agent",
                # stdout=devnull,  # Redirect stdout to devnull
                # stderr=devnull,  # Redirect stderr to devnull
            )
            self._leaderboard_pid = leaderboard_process.pid

    def _generate_leaderboard_command(self, carla_rpc_port, tm_port):
        """
        Generate the command to start the leaderboard evaluation.
        """
        if self._eval:
            if not self._eval_routes_queue.empty():
                route_file, seed, result_index = self._eval_routes_queue.get()
                self._route_file = route_file
                self._seed = seed
                self._result_index = result_index
                self._save_frames = True

                checkpoint_path = (
                    f"../carla_leaderboard_checkpoints/{self._results_directory}/route_{self._result_index}.json"
                )
            else:
                # All routes have been evaluated or are currently being evaluated so we stop saving the frames
                self._save_frames = False

                checkpoint_path = (
                    f"../carla_leaderboard_checkpoints/{self._results_directory}/route_{self._result_index}_todl.json"
                )
        else:
            self._route_file = self._training_routes[self._simulator_instance_index]
            self._result_index = str(self._simulator_instance_index)

        pathlib.Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)

        carla_tm_port = self._get_free_port()

        return (
            f"bash start_carla_leaderboard.sh {self._route_file} "
            f"{checkpoint_path} {carla_rpc_port} {carla_tm_port} {self._seed} "
            f"{self._ipc_file} {self._eval}"
        )

    def _convert_observation(self, observation):
        """
        Convert raw CARLA observation to the format expected by the RL agent.
        """
        image = observation["bev_image"]
        scalars = observation["scalars"].astype(np.float32)
        return image, scalars

    # Timeout for the servers used by dreamer is 300 seconds, so we must choose a smaller timeout value
    @timeout(120)
    def _establish_connection_with_timeout(self):
        """Establish ZMQ connection with the leaderboard process."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"ipc:///tmp/{self._ipc_file}")

        # Send initial configuration (Right now not used)
        config_dict = {}
        self._socket.send_pyobj(config_dict)

        # Wait for acknowledgment
        message = self._socket.recv_pyobj()
        print(f"Connection established: {message}")

    def step(self, action):
        """
        Execute one environment step.
        """
        if not self._established_connection:
            try:
                self.close()
                self._ipc_file = f"inner{np.random.randint(2 ** 32)}"
                self._start_carla_with_timeout()
                self._establish_connection_with_timeout()
                self._established_connection = True
            except SimulationTimeoutError:
                print("Simulation timed out during connection establishment")
                self._established_connection = False
                return self._create_default_sample()

        try:
            return self._step_with_timeout(action)
        except SimulationTimeoutError:
            print("Simulation timed out during step execution")
            self._established_connection = False
            return self._create_default_sample()

    @timeout(50)
    def _step_with_timeout(self, action):
        """Execute a single step with timeout protection."""
        if action["reset"]:
            command = ("reset",)
            is_first = True
        else:
            control_action = self.AVAILABLE_ACTIONS[action["action"]]
            command = ("step", control_action)
            is_first = False

        # Send command and receive response
        self._socket.send_pyobj(command)
        data = self._socket.recv_pyobj()

        observation = data["observation"]
        reward = data["reward"]
        termination = data["termination"]
        truncation = data["truncation"]
        info = data["info"]

        # Handle evaluation episode completion
        if self._eval and (termination or truncation):
            time.sleep(10)  # Allow leaderboard evaluator to finish
            raise SimulationTimeoutError()

        # Process observation
        processed_image, scalars = self._convert_observation(observation)

        # Create sample dictionary
        sample = {
            "image": processed_image,
            "scalars": scalars,
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": termination or truncation,
            "is_terminal": termination,
            "log/carla_crashed": False,
            "log/carla_run_red_light": info[0],
            "log/carla_run_stop_sign": info[1],
            "log/carla_exceed_route_dev": info[2],
            "log/carla_collision": info[3],
            "log/carla_timeout": info[4],
            "log/carla_route_end": info[5],
        }

        # Add evaluation metadata
        if self._eval:
            sample.update(
                {
                    "log/route_file": self._route_file,
                    "log/seed": self._seed,
                    "log/result_index": self._result_index,
                    "log/save_frames": self._save_frames,
                }
            )

        return sample

    def _get_free_port(self):
        """
        Find a free port for the CARLA simulator.

        Returns:
            Available port number.
        """

        def is_port_free(port: int):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    return True
                except socket.error:
                    return False

        while True:
            port = int(np.random.randint(1024, 65536))
            if is_port_free(port):
                return port

    def _terminate_process_and_children(self, pid):
        """
        Terminate a process and all its child processes.
        """
        if not pid:
            return

        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for graceful termination
            gone, alive = psutil.wait_procs(children, timeout=3)

            # Force kill remaining processes
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass

            # Terminate parent process
            try:
                parent.terminate()
                parent.wait(3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    def close(self):
        """Clean up all resources and terminate processes."""
        if self._socket:
            self._socket.close(linger=0)

        if self._carla_process:
            self._terminate_process_and_children(self._carla_process.pid)

        if self._leaderboard_pid:
            self._terminate_process_and_children(self._leaderboard_pid)

        if self._context:
            self._context.destroy(linger=0)
            self._context.term()

        # Reset all process references
        self._context = None
        self._socket = None
        self._carla_process = None
        self._leaderboard_pid = None

    def __del__(self):
        """Ensure cleanup when object is garbage collected."""
        self.close()

    # Abstract methods from embodied.Env (not used but required for interface compliance)
    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        raise NotImplementedError()

    def _reset(self):
        raise NotImplementedError()
