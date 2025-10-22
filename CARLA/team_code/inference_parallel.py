import argparse, os, subprocess, time, socket, xml.etree.ElementTree as ET, shutil, sys, itertools

def build_env_for_backend(backend, base_env, shard_save_dir,
                          ppo_cpp_build=None,
                          singularity_sif=None,
                          ld_library_path_extra=None):
    """
    backend: "cpp" or "py"
    ppo_cpp_build: path to ppo.cpp/build
    singularity_sif: optional path to a .sif if you must run inside Singularity
    ld_library_path_extra: optional extra colon-separated paths
    """
    env = base_env.copy()
    env['SAVE_PATH'] = shard_save_dir
    env['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(shard_save_dir, ".torch_cache")

    if backend == "cpp":
        env["CPP"] = "1"
        if ppo_cpp_build:
            env["PPO_CPP_INSTALL_PATH"] = ppo_cpp_build
            env["LD_LIBRARY_PATH"] = ppo_cpp_build + ":" + env.get("LD_LIBRARY_PATH", "")
        if ld_library_path_extra:
            env["LD_LIBRARY_PATH"] = ld_library_path_extra + ":" + env.get("LD_LIBRARY_PATH", "")
        if singularity_sif:
            env["PATH_TO_SINGULARITY"] = singularity_sif
    else:
        env.pop("CPP", None)
    return env

def next_free_port(start=2000):
    p = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', p)); return p
            except OSError:
                p += 1

def split_routes_xml(src_xml, out_dir, shards):
    os.makedirs(out_dir, exist_ok=True)
    tree = ET.parse(src_xml)
    root = tree.getroot()
    routes = root.findall('route')
    n = len(routes)
    if n == 0:
        raise RuntimeError("No <route> entries found.")
    per = (n + shards - 1) // shards
    outs = []
    for i in range(shards):
        beg, end = i * per, min((i+1) * per, n)
        if beg >= end: break
        new_root = ET.Element('routes')
        for r in routes[beg:end]:
            new_root.append(r)
        out_path = os.path.join(out_dir, f'chunk_{i+1:02d}_of_{shards:02d}.xml')
        ET.ElementTree(new_root).write(out_path, encoding='utf-8', xml_declaration=True)
        outs.append(out_path)
    return outs

def launch_carla(carla_root, rpc_port, tm_port, sensor_port, primary_port, gpu=0, headless=True, log_file=None):
    null = open(os.devnull, 'w')
    outs = open(log_file, 'w') if log_file else null
    errs = outs
    flags = [
        f"-carla-rpc-port={rpc_port}",
        f"-carla-primary-port={primary_port}",
        f"-carla-streaming-port={sensor_port}",
        "-nosound",
        "-RenderOffScreen",
        f"-graphicsadapter={gpu}",
        "-RPCThreads=2", "-StreamingThreads=2", "-SecondaryThreads=2",
    ]
    if headless:
        flags.append("-nullrhi")
    cmd = ["bash", f"{carla_root}/CarlaUE4.sh"] + flags
    return subprocess.Popen(cmd, stdout=outs, stderr=errs)

def launch_eval(python_bin, evaluator_py, routes_xml, agent_py, agent_cfg, port, tm_port,
                save_dir, checkpoint_json, track="MAP", log_file=None, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env['SAVE_PATH'] = save_dir
    env['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(save_dir, ".torch_cache")
    os.makedirs(save_dir, exist_ok=True)
    null = open(os.devnull, 'w')
    outs = open(log_file, 'w') if log_file else null
    errs = outs
    cmd = [
        python_bin, evaluator_py,
        "--routes", routes_xml,
        "--agent", agent_py,
        "--agent-config", agent_cfg,
        "--track", track,
        "--port", str(port),
        "--traffic-manager-port", str(tm_port),
        "--resume", "1",
        "--checkpoint", checkpoint_json
    ]
    return subprocess.Popen(cmd, stdout=outs, stderr=errs, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True, help="Path to a big routes XML (e.g., inference_route60.xml)")
    ap.add_argument("--shards", type=int, default=4, help="How many parallel workers")
    ap.add_argument("--carla-root", required=True)
    ap.add_argument("--leaderboard-root", required=True)
    ap.add_argument("--agent", required=True, help="Path to eval_agent.py")
    ap.add_argument("--agent-configs", nargs="+", required=True,
                    help="One or more model config dirs; if >1 we round-robin assign per shard")
    ap.add_argument("--work-dir", required=True, help="A working directory for outputs")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--start-port", type=int, default=2000)
    ap.add_argument("--gpu-ids", nargs="+", type=int, default=[0], help="GPU index per shard (cycled if shorter)")
    ap.add_argument("--track", default="MAP")
    ap.add_argument("--cpp", action="store_true", help="Set CPP=1 and LD_LIBRARY_PATH additions per shard")
    args = ap.parse_args()

    # 1) shard the routes
    chunks_dir = os.path.join(args.work_dir, "route_chunks")
    chunks = split_routes_xml(args.routes, chunks_dir, args.shards)

    # 2) per shard: allocate ports & launch server + evaluator
    carla_procs, eval_procs = [], []
    cur = args.start_port
    for i, chunk in enumerate(chunks):
        rpc = next_free_port(cur); cur = rpc + 1
        tm  = next_free_port(cur); cur = tm  + 1
        sensor = next_free_port(cur); cur = sensor + 1
        primary = next_free_port(cur); cur = primary + 1

        gpu = args.gpu_ids[i % len(args.gpu_ids)]
        save_i = os.path.join(args.work_dir, f"infer_worker_{i+1:02d}")
        os.makedirs(save_i, exist_ok=True)

        carla_log = os.path.join(save_i, "carla.log")
        eval_log  = os.path.join(save_i, "evaluator.log")

        print(f"[Shard {i+1}] ports rpc/tm/stream/primary = {rpc}/{tm}/{sensor}/{primary}  | gpu={gpu}")
        srv = launch_carla(args.carla_root, rpc, tm, sensor, primary, gpu=gpu, headless=True, log_file=carla_log)
        carla_procs.append(srv)
        time.sleep(6)  # give server a moment

        cfg = args.agent_configs[i % len(args.agent_configs)]
        chk_json = os.path.join(save_i, "inference_result.json")

        # prepare per-shard env (ex: CPP libs)
        extra_env = {}
        if args.cpp:
            extra_env["CPP"] = "1"
            # If you need custom LD_LIBRARY_PATH / PPO_CPP paths, put them here:
            # extra_env["LD_LIBRARY_PATH"] = "/path/to/ppo.cpp/build:" + os.environ.get("LD_LIBRARY_PATH","")

        evaluator_py = os.path.join(args.leaderboard_root, "leaderboard", "leaderboard_evaluator.py")
        ev = launch_eval(args.python_bin, evaluator_py, chunk, args.agent, cfg, rpc, tm,
                         save_i, chk_json, track=args.track, log_file=eval_log, extra_env=extra_env)
        eval_procs.append(ev)

    # 3) wait & clean up
    code = 0
    try:
        rets = [p.wait() for p in eval_procs]
        code = max(rets)
    finally:
        for p in eval_procs:
            if p.poll() is None: p.terminate()
        time.sleep(2)
        for p in carla_procs:
            if p.poll() is None: p.terminate()
    print("All shards finished with codes:", rets)
    sys.exit(code)

if __name__ == "__main__":
    main()
