import argparse, time, pathlib, numpy as np, zmq, jsonpickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]          # .../CARLA
TEAM_CODE = HERE.parents[1]          # .../CARLA/team_code
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))    # so `import rl_config` works

# use team_code/comm_files to match EnvAgent
COMM_FOLDER = TEAM_CODE / "comm_files"
COMM_FOLDER.mkdir(parents=True, exist_ok=True)
import rl_config

# Minimal config fields used by EnvAgent; tune as needed
def build_dummy_config():
    c = rl_config.GlobalConfig()
    # timings
    c.frame_rate = 10
    c.start_delay_frames = 5
    c.action_repeat = 1
    c.eval_time = 20.0   # seconds budget per route (arbitrary for smoke)
    # observations / reward toggles
    c.use_new_bev_obs = False
    c.reward_type = "simple_reward"  # or "roach" if your reward handler requires it
    c.use_value_measurements = True
    c.use_ttc = False
    c.use_comfort_infraction = False
    c.num_value_measurements = 3     # remaining_time, time_till_blocked, perc_route_left
    # traffic tweaks
    c.use_green_wave = False
    c.green_wave_prob = 0.0
    # extra inputs
    c.use_extra_control_inputs = False
    c.use_target_point = False
    # misc
    c.rr_maximum_speed = 13.9  # ~50 km/h
    return c

def bool_from_npbuf(buf):
    # ZMQ gives a memoryview; we know env_agent sends dtype=bool
    return np.frombuffer(buf, dtype=np.bool_).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True, help="RL (ZMQ) port you passed to start_leaderboard")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--throttle", type=float, default=0.35, help="constant forward accel in [0,1]")
    ap.add_argument("--steer_amp", type=float, default=0.0, help="sinusoidal steer amplitude in [-1,1]")
    args = ap.parse_args()

    ctx = zmq.Context()
    comm_folder = pathlib.Path(__file__).resolve().parent.parent / "comm_files"
    comm_folder.mkdir(parents=True, exist_ok=True)
    comm_file = comm_folder / str(args.port)

    # 1) Config channel: EnvAgent connects and expects us to send a config first
    conf_sock = ctx.socket(zmq.PAIR)
    conf_sock.bind(f"ipc://{comm_file}.conf_lock")
    print(f"[smoke] bound config at ipc://{comm_file}.conf_lock")

    # 2) Data channel: observations/actions back and forth
    data_sock = ctx.socket(zmq.PAIR)
    data_sock.bind(f"ipc://{comm_file}.lock")
    print(f"[smoke] bound data at   ipc://{comm_file}.lock")

    # Send config when EnvAgent connects (it will immediately recv)
    cfg = build_dummy_config()
    conf_sock.send_string(jsonpickle.encode(cfg))
    print("[smoke] sent config")
    # Wait for EnvAgent's ack
    try:
        ack = conf_sock.recv_string(flags=0)
        print(f"[smoke] got ack: {ack}")
    except Exception as e:
        print(f"[smoke] (warn) no config ack: {e}")
    conf_sock.close()
    
    steps = 0
    episodes = 0
    print("[smoke] entering main loop (Ctrl-C to quit)")
    while True:
        parts = data_sock.recv_multipart(copy=False)
        # parts order in env_agent.py send_multipart:
        # (bev_sem, measurements, value_meas, reward, termination, truncation, n_steps, suggest)
        if len(parts) < 6:
            print(f"[smoke] warning: got {len(parts)} parts instead of 8, skipping...")
            continue
        term = bool_from_npbuf(parts[4])
        trunc = bool_from_npbuf(parts[5])

        if term or trunc:
            episodes += 1
            steps = 0
            # (no special reply needed; just keep answering actions)

        # Simple driving policy: slight forward accel, optional tiny steer wiggle
        steer = 0.0
        accel = 0.35  # positive -> throttle
        action = np.array([steer, accel], dtype=np.float32)

        data_sock.send(action.tobytes(), copy=False)
        steps += 1
        if steps % 200 == 0:
            print(f"[smoke] episodes={episodes} steps_in_ep={steps}")

    data_sock.close()
    print("[smoke] done.")

if __name__ == "__main__":
    main()
