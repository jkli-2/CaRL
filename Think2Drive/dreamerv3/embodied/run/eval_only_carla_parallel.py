import sys
import os
import gc
import glob
import time
import pathlib
from collections import defaultdict
from functools import partial as bind
from multiprocessing import Queue

import cloudpickle
import embodied
import numpy as np
import cv2
import ujson


def prefix(d, p):
    """Add prefix to dictionary keys."""
    return {f"{p}/{k}": v for k, v in d.items()}


class EvaluationCompletedException(Exception):
    """Exception raised when an evaluation process is completed."""

    pass


def eval_only_carla_parallel(make_agent, make_env, args, config):
    """
    Main function to set up and run the parallel CARLA evaluation pipeline.

    Args:
        make_agent: Function to create the agent
        make_replay: Function to create the replay buffer
        make_env: Function to create the environment
        make_logger: Function to create the logger
        args: Configuration arguments
    """
    # Validate environment and batch sizes
    if args.num_envs:
        assert args.actor_batch <= args.num_envs, (
            f"Actor batch size ({args.actor_batch}) cannot exceed " f"number of environments ({args.num_envs})"
        )

    # Automatically assign ports if needed
    for key in ("actor_addr",):
        if "{auto}" in args[key]:
            port = embodied.distr.get_free_port()
            args = args.update({key: args[key].format(auto=port)})

    eval_routes_queue = Queue()
    final_result_files = []
    eval_routes_directory = os.path.abspath(config["env.carla.eval_routes_directory"])
    results_directory = config["env.carla.results_directory"]

    # Populate evaluation queue with incomplete routes
    _populate_evaluation_queue(eval_routes_queue, final_result_files, eval_routes_directory, results_directory, config)

    # Serialize functions for distribution
    make_env = cloudpickle.dumps(make_env)
    make_agent = cloudpickle.dumps(make_agent)

    # Create worker processes
    workers = [
        embodied.distr.Process(parallel_env, make_env, i, args, config, eval_routes_queue)
        for i in range(args.num_envs)
    ]

    # Add agent process or thread
    if args.agent_process:
        workers.append(
            embodied.distr.Process(parallel_agent, make_agent, args, config, eval_routes_queue, final_result_files)
        )
    else:
        workers.append(
            embodied.distr.Thread(parallel_agent, make_agent, args, config, eval_routes_queue, final_result_files)
        )

    # Run all workers
    embodied.distr.run(workers, args.duration, exit_after=True)


def _populate_evaluation_queue(
    eval_routes_queue, final_result_files, eval_routes_directory, results_directory, config
):
    """Populate the evaluation queue with incomplete routes."""
    for dataset_evaluation_idx in range(config["env.carla.eval_times"]):
        for route_idx, route_file in enumerate(sorted(glob.glob(f"{eval_routes_directory}/*"))):
            seed = int(hash((dataset_evaluation_idx, route_idx, config["seed"])) % (2**32))
            result_index = f"{dataset_evaluation_idx}_{route_idx}"

            result_file = os.path.abspath(
                f"../carla_leaderboard_checkpoints/{results_directory}/route_{result_index}.json"
            )

            if not _is_result_file_complete(result_file):
                eval_routes_queue.put((route_file, seed, result_index))
                final_result_files.append(result_file)


def _is_result_file_complete(result_file):
    """Check if a result file contains complete evaluation data."""
    try:
        with open(result_file, "r") as file:
            data = ujson.load(file)

        infractions = (
            data.get("_checkpoint", {})
            .get("global_record", {})
            .get("infractions", {})
            .get("min_speed_infractions", None)
        )
        return infractions is not None
    except (FileNotFoundError, ValueError, KeyError):
        return False


def parallel_agent(make_agent, args, config, eval_routes_queue, final_result_files):
    """
    Run the agent in parallel with actor threads.

    Args:
        make_agent: Function to create the agent
        args: Configuration arguments
    """
    # Deserialize agent creation function if needed
    if isinstance(make_agent, bytes):
        make_agent = cloudpickle.loads(make_agent)

    agent = make_agent()

    # Create actor thread
    workers = [embodied.distr.Thread(parallel_actor, agent, args, config, eval_routes_queue, final_result_files)]

    # Run the workers
    embodied.distr.run(workers, args.duration)


class ColorConfig:
    """Configuration for color mappings in visualization."""

    color_mappings = np.array(
        [
            [50, 50, 50],  # Road
            [150, 150, 150],  # Route
            [255, 255, 255],  # Ego
            [100, 100, 100],  # Lane
            [255, 255, 0],  # Yellow lines
            [255, 0, 255],  # White lines
            # Dynamic objects at different time steps
            [0, 0, 230],  # Vehicle at t=0
            [0, 230, 230],  # Walker at t=0
            [0, 0, 230],  # Emergency car at t=0
            [230, 0, 0],  # Obstacle at t=0
            [0, 230, 0],  # Green traffic light at t=0
            [230, 0, 0],  # Yellow & Red traffic light at t=0
            [170, 170, 0],  # Stop sign at t=0
            [50, 50, 230],  # Vehicle at t=-5
            [50, 230, 230],  # Walker at t=-5
            [50, 50, 230],  # Emergency car at t=-5
            [230, 50, 50],  # Obstacle at t=-5
            [50, 230, 50],  # Green traffic light at t=-5
            [230, 230, 50],  # Yellow & Red traffic light at t=-5
            [170, 170, 50],  # Stop sign at t=-5
            [100, 100, 230],  # Vehicle at t=-10
            [100, 230, 230],  # Walker at t=-10
            [100, 100, 230],  # Emergency car at t=-10
            [230, 100, 100],  # Obstacle at t=-10
            [100, 230, 100],  # Green traffic light at t=-10
            [230, 230, 100],  # Yellow & Red traffic light at t=-10
            [170, 170, 100],  # Stop sign at t=-10
            [150, 150, 230],  # Vehicle at t=-15
            [150, 230, 230],  # Walker at t=-15
            [150, 150, 230],  # Emergency car at t=-15
            [230, 150, 150],  # Obstacle at t=-15
            [100, 230, 100],  # Green traffic light at t=-15
            [230, 230, 150],  # Yellow & Red traffic light at t=-15
            [170, 170, 150],  # Stop sign at t=-15
        ]
    )

    @classmethod
    def get_color_mappings(cls):
        """Get the color mapping array."""
        return cls.color_mappings


def convert_frames_to_images(frames):
    """
    Convert multi-channel or RGB frames to displayable images.
    """
    if frames.shape[-1] > 3:
        return _process_multi_channel_frames(frames)
    else:
        return _process_rgb_frames(frames)


def _process_multi_channel_frames(frames):
    """Process multi-channel frames using color mappings."""
    factor = frames / 255.0
    color_mappings = ColorConfig.get_color_mappings()

    video_frames = factor[:, :, :, 0, None] * color_mappings[0]

    # Apply color mappings for each channel
    channel_order = [
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

    for i in channel_order:
        channel_factor = factor[:, :, :, i, None]
        channel_color = frames[:, :, :, i, None] / 255.0 * color_mappings[i]
        video_frames = (1 - channel_factor) * video_frames + channel_factor * channel_color

    return np.round(video_frames).clip(0, 255).astype(np.uint8)


def _process_rgb_frames(frames):
    """Process RGB frames."""
    video_frames = frames.copy()
    if video_frames.shape[-1] == 1:
        video_frames = video_frames.squeeze(axis=-1)
    return video_frames


def _define_available_actions():
    """
    Define the available actions in the CARLA environment.

    Returns:
        tuple: A tuple of available actions as (throttle, brake, steer) tuples.
    """
    return (
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


def _add_action_visualization(frame, action_index, actions):
    """Add action visualization bars to the frame."""
    if action_index < 0 or action_index >= len(actions):
        return frame

    throttle, brake, steer = actions[action_index]

    # Colors
    YELLOW = (200, 200, 0)
    GREEN = (0, 200, 0)
    RED = (200, 0, 0)
    BLUE = (50, 50, 200)
    WHITE = (150, 150, 150)

    # Draw frame borders
    cv2.line(frame, (5, 50 + 128 + 45), (5, 50 + 256 - 40), WHITE, 2)
    cv2.line(frame, (256 - 5, 50 + 128 + 45), (256 - 5, 50 + 256 - 40), WHITE, 2)
    cv2.line(frame, (5, 50 + 256 + 45), (5, 50 + 384 - 40), WHITE, 2)
    cv2.line(frame, (256 - 5, 50 + 256 + 45), (256 - 5, 50 + 384 - 40), WHITE, 2)

    # Draw steering visualization
    steer_color = BLUE if steer < 0 else GREEN if steer > 0 else YELLOW
    cv2.rectangle(frame, (128, 50 + 128 + 60), (int(round(128 + steer * 123, 0)), 50 + 256 - 55), steer_color, -1)

    # Draw throttle/brake visualization
    if brake > 0:
        cv2.rectangle(frame, (128, 50 + 256 + 55), (int(round(128 - brake * 123, 0)), 50 + 384 - 50), RED, -1)
    else:
        cv2.rectangle(frame, (128, 50 + 256 + 55), (int(round(128 + throttle * 123, 0)), 50 + 384 - 50), GREEN, -1)

    # Draw center lines
    cv2.line(frame, (128, 50 + 128 + 45), (128, 50 + 256 - 40), WHITE, 2)
    cv2.line(frame, (128, 50 + 256 + 45), (128, 50 + 384 - 40), WHITE, 2)

    # Add labels
    _put_text(frame, "steering", (2, 200))
    _put_text(frame, "throttle & brake", (2, 328))

    return frame


def _add_text_to_frame(frame, observations, actions):
    """Add telemetry text overlays to the frame."""
    text_configs = [
        ("Steer", (5, 25), f'{observations["scalars"][0][2]:.2f}'),
        ("Throt", (5, 65), f'{observations["scalars"][0][3]:.2f}'),
        ("Brake", (5, 105), f'{observations["scalars"][0][4]:.2f}'),
        ("Speed", (182, 25), f'{observations["scalars"][0][0]:.2f}'),
        ("SpLim", (182, 65), f'{observations["scalars"][0][1]:.2f}'),
        ("Reward", (182, 105), f'{observations["reward"][0]:.2f}'),
        ("Conti", (360, 25), f'{observations["estimated_cont"][0]:.2f}'),
        ("Value", (360, 65), f'{observations["estimated_value"][0]:.2f}'),
        ("EstRew", (360, 105), f'{observations["estimated_reward"][0]:.2f}'),
    ]

    for label, position, value in text_configs:
        _put_text(frame, f"{label}: {value}", position)

    action_index = observations["action"][0]
    frame = _add_action_visualization(frame, action_index, actions)

    # Add scalar values
    locations = [
        (5, 140),
        (60, 140),
        (115, 140),
        (170, 140),
        (225, 140),
        (280, 140),
        (335, 140),
        (390, 140),
        (445, 140),
        (5, 170),
    ]
    colors = np.array([255, 255, 255, 190, 255, 255, 255, 190, 255, 190], dtype=np.uint8)
    colors = colors[:, None] * np.ones((1, 3))

    for scalar, position, color in zip(observations["scalars"][0][5:], locations, colors):
        _put_text(frame, str(round(scalar, 1)), position, color=color)

    return frame


def _put_text(frame, text, position, color=(255, 255, 255)):
    """Put text on frame with specified color."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def generate_video_frame(observations):
    """
    Generate a single video frame from observations.
    """
    frame = np.zeros((690, 512, 3), dtype=np.uint8)
    frame[:178, :, :] = 20

    # Process main image
    image = convert_frames_to_images(observations["image"])[0]  # (128, 128, 3)

    # Process saliency map
    saliency = (observations["saliency"][0] * 255).max(axis=-1).astype(np.uint8)[:, :, None]

    # Process estimated decoder output
    estimated_dec = (observations["estimated_dec"] * 255).astype(np.uint8)
    estimated_dec = convert_frames_to_images(estimated_dec)[0]

    # Resize all components
    resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    resized_saliency = cv2.resize(saliency, (256, 256), interpolation=cv2.INTER_CUBIC)
    resized_dec = cv2.resize(estimated_dec, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Create saliency overlay
    saliency_overlay = np.zeros_like(resized_image)
    saliency_overlay[:, :, 0] = resized_saliency.squeeze()

    # Compose final frame
    frame[178:434, 256:] = resized_image
    alpha = 0.75
    frame[434:, :256] = alpha * saliency_overlay + (1 - alpha) * resized_image
    frame[434:, 256:] = resized_dec

    actions = _define_available_actions()
    frame = _add_text_to_frame(frame, observations, actions)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def parallel_actor(agent, args, config, eval_routes_queue, final_result_files):
    """
    Run the actor component of the evaluation system.

    Args:
        agent: The agent object
        barrier: Synchronization barrier
        args: Configuration arguments
    """
    islist = lambda x: isinstance(x, list)
    initial = agent.init_policy(args.actor_batch)
    initial = embodied.tree.map(lambda x: x[0], initial, isleaf=islist)
    allstates = defaultdict(lambda: initial)

    # Load checkpoint
    logdir = embodied.Path(args.logdir)
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()

    results_directory = config["env.carla.results_directory"]
    all_video_frames = defaultdict(list)

    @embodied.timer.section("actor_workfn")
    def workfn(obs):
        """Worker function to process observations and generate actions."""
        envids = obs.pop("envid")

        with embodied.timer.section("get_states"):
            states = [allstates[a] for a in envids]
            states = embodied.tree.map(lambda *xs: list(xs), *states)

        acts, outs, states = agent.policy(
            obs,
            states,
            mode="eval",
            compute_saliency=True,
            compute_rew=True,
            compute_cont=True,
            compute_value=True,
            compute_deconv=True,
        )

        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))
        acts["reset"] = obs["is_last"].copy()

        with embodied.timer.section("put_states"):
            for i, a in enumerate(envids):
                allstates[a] = embodied.tree.map(lambda x: x[i], states, isleaf=islist)

        trans = {"envids": envids, **obs, **acts, **outs}
        [x.setflags(write=False) for x in trans.values()]

        return acts, trans

    @embodied.timer.section("actor_donefn")
    def donefn(trans):
        """Handle completed transitions and save results."""
        is_last = trans.pop("is_last")
        # Everytime a route finished with evaluation this tran[log/carla_crashed][i] is set to True
        _ = trans.pop("log/carla_crashed")
        route_file = trans.pop("log/route_file")
        seed = trans.pop("log/seed")
        result_index = trans.pop("log/result_index")
        save_frames = trans.pop("log/save_frames")

        for i, envid in enumerate(trans.pop("envids")):
            observations = {k: v[i][None] for k, v in trans.items()}
            video_frame = generate_video_frame(observations)
            all_video_frames[envid].append(video_frame)

            if is_last[i]:
                _handle_episode_completion(
                    envid,
                    i,
                    all_video_frames,
                    results_directory,
                    route_file,
                    seed,
                    result_index,
                    save_frames,
                    eval_routes_queue,
                    final_result_files,
                )

    # Set up and run the actor server
    server = embodied.distr.ProcServer(args.actor_addr, "Actor", args.ipv6)
    server.bind("act", workfn, donefn, args.actor_threads, args.actor_batch)
    server.run()


def _handle_episode_completion(
    envid,
    i,
    all_video_frames,
    results_directory,
    route_file,
    seed,
    result_index,
    save_frames,
    eval_routes_queue,
    final_result_files,
):
    """Handle completion of an episode including video saving and rescheduling."""
    video_frames = all_video_frames[envid]

    result_file = f"../carla_leaderboard_checkpoints/{results_directory}/route_{result_index[i]}.json"

    if not _is_result_file_complete(result_file):
        eval_routes_queue.put((route_file[i], seed[i], result_index[i]))
        if os.path.exists(result_file):
            open(result_file, "w").close()
            os.remove(result_file)
        print(f"Reschedule route {route_file[i]}", flush=True)
    elif save_frames[i]:
        _save_video_and_frames(video_frames, result_index[i], results_directory)

    del all_video_frames[envid]

    # Check if all evaluations are complete
    if eval_routes_queue.empty() and all(_is_result_file_complete(f) for f in final_result_files):
        raise EvaluationCompletedException("Evaluation completed successfully")


def _save_video_and_frames(video_frames, result_index, results_directory):
    """Save video frames as both MP4 and individual images."""
    parts = result_index.split("_")
    eval_times = int(parts[0])
    route_number = int(parts[1])

    video_dir = pathlib.Path(f"../carla_leaderboard_checkpoints/{results_directory}/videos")
    video_dir.mkdir(exist_ok=True, parents=True)

    images_dir = pathlib.Path(
        f"../carla_leaderboard_checkpoints/{results_directory}/frames/frames_{eval_times}_{route_number}"
    )
    images_dir.mkdir(exist_ok=True, parents=True)

    # Create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20
    video_path = video_dir / f"{eval_times}_{route_number}.mp4"
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (512, 690))

    try:
        for frame_idx, frame in enumerate(video_frames):
            image_path = images_dir / f"frame_{eval_times}_{route_number}_{frame_idx:04d}.png"
            cv2.imwrite(str(image_path), frame)
            out.write(frame)
    finally:
        out.release()


def parallel_env(make_env, envid, args, config, eval_routes_queue):
    """
    Run a single environment instance.
    """
    if isinstance(make_env, bytes):
        make_env = cloudpickle.loads(make_env)

    assert envid >= 0, f"Invalid environment ID: {envid}"
    name = f"Env{envid}"
    _print = lambda x: embodied.print(f"[{name}] {x}", flush=True)

    _print("Initializing environment")

    results_directory = config["env.carla.results_directory"]
    make_env = bind(
        make_env,
        eval=True,
        eval_routes_queue=eval_routes_queue,
        num_envs=args.num_envs,
        results_directory=results_directory,
    )

    env = make_env(envid)
    actor = embodied.distr.Client(args.actor_addr, name, args.ipv6, identity=envid, pings=10, maxage=60, connect=True)

    done = True
    while True:
        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        with embodied.timer.section("env_step"):
            obs = env.step(act)

        obs = {k: np.asarray(v, order="C") for k, v in obs.items()}
        score += obs["reward"]
        length += 1
        done = obs["is_last"]

        if done:
            _print(f"Episode completed: length={length}, score={score:.2f}")

        with embodied.timer.section("env_request"):
            future = actor.act({"envid": envid, **obs})
        try:
            with embodied.timer.section("env_response"):
                act = future.result()
        except embodied.distr.NotAliveError:
            # Wait until we are connected again, so we don't unnecessarily reset the
            # environment hundreds of times while the server is unavailable.
            _print("Lost connection to server, reconnecting...")
            actor.connect()
            done = True
        except embodied.distr.RemoteError as e:
            _print(f"Shutting down env due to agent error: {e}")
            sys.exit(0)
