import gc
import re
import sys
import threading
import time
from collections import defaultdict, deque
from functools import partial as bind

import cloudpickle
import embodied
import jax
import numpy as np


def prefix(dictionary, prefix):
    """
    Add a prefix to all dictionary keys.
    """
    return {f"{prefix}/{key}": value for key, value in dictionary.items()}


def combined(make_agent, make_replay, make_env, make_logger, args):
    """
    Main function to set up and run the combined training pipeline.

    Args:
        make_agent: Function to create the agent
        make_replay: Function to create the replay buffer
        make_env: Function to create the environment
        make_logger: Function to create the logger
        args: Configuration arguments
    """
    # Validate environment and batch size configuration
    if args.num_envs:
        assert args.actor_batch <= args.num_envs, (
            f"Actor batch size ({args.actor_batch}) cannot exceed " f"number of environments ({args.num_envs})"
        )

    # Automatically assign available ports for inter-process communication
    for address_key in ("actor_addr", "replay_addr", "logger_addr"):
        if "{auto}" in args[address_key]:
            port = embodied.distr.get_free_port()
            args = args.update({address_key: args[address_key].format(auto=port)})

    # Serialize factory functions for cross-process transmission
    make_env = cloudpickle.dumps(make_env)
    make_agent = cloudpickle.dumps(make_agent)
    make_replay = cloudpickle.dumps(make_replay)
    make_logger = cloudpickle.dumps(make_logger)

    # Create worker processes
    workers = [embodied.distr.Process(parallel_env, make_env, i, args, True) for i in range(args.num_envs)]

    # Add agent process (can run as process or thread based on configuration)
    if args.agent_process:
        workers.append(embodied.distr.Process(parallel_agent, make_agent, args))
    else:
        workers.append(embodied.distr.Thread(parallel_agent, make_agent, args))

    # Add replay buffer process if not using remote replay
    if not args.remote_replay:
        workers.append(embodied.distr.Process(parallel_replay, make_replay, args))

    # Add centralized logger process
    workers.append(embodied.distr.Process(parallel_logger, make_logger, args))

    # Execute all workers with specified duration
    embodied.distr.run(workers, args.duration, exit_after=True)


def parallel_agent(make_agent, args):
    """
    Initialize and run the agent in parallel with actor and learner components.

    Args:
        make_agent: Function to create the agent
        args: Configuration arguments
    """
    # Deserialize agent creation function if needed
    if isinstance(make_agent, bytes):
        make_agent = cloudpickle.loads(make_agent)

    agent = make_agent()
    barrier = threading.Barrier(2)  # Synchronize actor and learner startup

    # Initialize monitoring counters for CARLA-specific metrics
    env_crashed = embodied.Counter()
    episode_end_causes = ["red_light", "stop_sign", "route_deviation", "collision", "timeout", "route_end"]
    episode_end_averages = [embodied.RunningAverage() for _ in episode_end_causes]
    total_infraction_averages = embodied.RunningAverage()

    # Create actor and learner worker threads
    workers = [
        embodied.distr.Thread(
            parallel_actor,
            agent,
            barrier,
            args,
            env_crashed,
            episode_end_causes,
            episode_end_averages,
            total_infraction_averages,
        ),
        embodied.distr.Thread(parallel_learner, agent, barrier, args),
    ]

    # Run both workers concurrently
    embodied.distr.run(workers, args.duration)


def parallel_actor(
    agent, barrier, args, env_crashed, episode_end_causes, episode_end_averages, total_infraction_averages
):
    """
    Function to run the actor part of the agent.

    Args:
        agent: The agent object
        barrier: Synchronization barrier
        args: Configuration arguments
    """
    islist = lambda x: isinstance(x, list)
    initial = agent.init_policy(args.actor_batch)
    initial = embodied.tree.map(lambda x: x[0], initial, isleaf=islist)
    allstates = defaultdict(lambda: initial)

    # Wait for learner to restore checkpoint
    barrier.wait()  # Do not collect data before learner restored checkpoint.

    fps = embodied.FPS()
    should_log = embodied.when.Clock(args.log_every)

    # Set up logger and replay buffer clients
    logger = embodied.distr.Client(
        args.logger_addr, "ActorLogger", args.ipv6, maxinflight=8 * args.actor_threads, connect=True
    )
    replay = embodied.distr.Client(
        args.replay_addr, "ActorReplay", args.ipv6, maxinflight=8 * args.actor_threads, connect=True
    )

    all_transitions = defaultdict(list)
    _print = lambda x: embodied.print(f"[Actor] {x}", flush=True)

    @embodied.timer.section("actor_workfn")
    def workfn(obs):
        """Worker function to process observations and generate actions."""
        envids = obs.pop("envid")
        fps.step(obs["is_first"].size)

        with embodied.timer.section("get_states"):
            states = [allstates[a] for a in envids]
            states = embodied.tree.map(lambda *xs: list(xs), *states)

        acts, outs, states = agent.policy(obs, states, compress_image=True)
        compressed_image = outs.pop("compressed_image").astype(np.uint8)
        obs["image"] = compressed_image

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
        """Function to handle completed transitions."""
        # Only add the transitions to the replay buffer once the episode ended
        # Since the episodes are not too long, that does not add much overhead

        crash = trans.pop("log/carla_crashed")

        info_run_red_light = trans.pop("log/carla_run_red_light")
        info_run_stop_sign = trans.pop("log/carla_run_stop_sign")
        info_exceed_route_dev = trans.pop("log/carla_exceed_route_dev")
        info_collision = trans.pop("log/carla_collision")
        info_timeout = trans.pop("log/carla_timeout")
        info_route_end = trans.pop("log/carla_route_end")

        logger.trans(trans)
        for i, envid in enumerate(trans.pop("envids")):
            transitions = all_transitions[envid]

            transitions.append({k: v[i][None] for k, v in trans.items()})

            if transitions[-1]["is_last"]:
                if crash[i] or len(transitions) <= 10:
                    _print(f"Discarding episode of length {len(transitions) + 1}.")
                    transitions.clear()
                    env_crashed.increment()
                else:
                    transition = {
                        key: np.concatenate([d[key] for d in transitions], axis=0) for key in transitions[0].keys()
                    }

                    end_causes = [
                        info_run_red_light,
                        info_run_stop_sign,
                        info_exceed_route_dev,
                        info_collision,
                        info_timeout,
                        info_route_end,
                    ]
                    # run_red_light, run_stop_sign, exceeded_route_deviation, collision_happened, timeout_reached, reached_route_end

                    total_infraction_averages.update(float(np.any(end_causes[:4])))
                    for cause, running_average in zip(end_causes, episode_end_averages):
                        running_average.update(float(cause))

                    transition["envids"] = [envid]

                    replay.add_batch(transition)
                    transitions.clear()

        if should_log():
            stats = {}
            stats["fps/policy"] = fps.result()
            stats["parallel/ep_states"] = len(allstates)
            stats["carla/env_crashed"] = int(env_crashed)
            stats["carla/sum_infractions"] = total_infraction_averages.current()

            for cause, running_average in zip(episode_end_causes, episode_end_averages):
                stats[f"carla/{cause}"] = running_average.current()

            stats.update(prefix(server.stats(), "server/actor"))
            stats.update(prefix(logger.stats(), "client/actor_logger"))
            stats.update(prefix(replay.stats(), "client/actor_replay"))
            logger.add(stats)

    # check if logger only uses results of env 0

    # Set up and run the actor server
    server = embodied.distr.ProcServer(args.actor_addr, "Actor", args.ipv6)
    server.bind("act", workfn, donefn, args.actor_threads, args.actor_batch)
    server.run()


def parallel_learner(agent, barrier, args):
    logdir = embodied.Path(args.logdir)
    agg = embodied.Agg()
    usage = embodied.Usage(**args.usage)
    should_log = embodied.when.Clock(args.log_every)
    should_eval = embodied.when.Clock(args.eval_every)
    should_save = embodied.when.Clock(args.save_every)
    fps = embodied.FPS()
    batch_steps = args.batch_size * (args.batch_length - args.replay_context)
    should_clean = embodied.when.Clock(30 * 60)
    should_reset_planner = embodied.when.Every(args.reset_planner_every, initial=False)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    logger = embodied.distr.Client(args.logger_addr, "LearnerLogger", args.ipv6, maxinflight=1, connect=True)
    updater = embodied.distr.Client(args.replay_addr, "LearnerReplayUpdater", args.ipv6, maxinflight=8, connect=True)

    barrier.wait()

    replays = []

    def parallel_dataset(source, prefetch=2):
        replay = embodied.distr.Client(args.replay_addr, f"LearnerReplay{len(replays)}", args.ipv6, connect=True)
        replays.append(replay)
        call = getattr(replay, source)
        futures = deque([call({}) for _ in range(prefetch)])
        while True:
            futures.append(call({}))
            yield futures.popleft().result()

    dataset_train = agent.dataset(bind(parallel_dataset, "sample_batch_train"))
    dataset_report = agent.dataset(bind(parallel_dataset, "sample_batch_report"))
    carry = agent.init_train(args.batch_size)
    carry_report = agent.init_report(args.batch_size)
    should_save()  # Delay first save.
    should_eval()  # Delay first eval.

    estimated_env_step = 0
    env_steps_per_train_step = args.batch_size * (args.batch_length - args.replay_context) / args.train_ratio

    called = 0
    available_train_steps_world_model = 0
    available_train_steps_planner = 0

    pretraining = agent.config.env.carla.pretraining

    num_trained_only_planner, num_trained_both = 0, 0
    while True:
        with embodied.timer.section("learner_batch_next"):
            batch = next(dataset_train)

        # Their email says: The settings “16, 32, 128, 256” in the initial arXiv version were the ratio configurations
        # we tried. In our final version, the planner training ratio at the end of training was set to 64. During
        # training, as the world model loss gradually decreases, we can focus on the planner training. This is the
        # very spirit of why we take an incremental train ratio. Therefore, the training frequency of the planner
        # increases as training progresses (this can also be observed from the sampling formula). We first change the
        # train ratio approximately 20K CARLA frames after the initial reset, with subsequent changes every 20K–40K
        # frames (total frames is 1-1.5M).

        # We train for 2M frames in total.
        estimated_env_step += env_steps_per_train_step
        called += 1

        # We set train_ratio = 64, which if we train the world model with train ratio 16, we only train it every
        # 4th step
        available_train_steps_world_model += 0.25

        # For the first 800_000 CARLA steps we use a train ratio of 16 for the world model and the planner
        # For the next 1_200_000 CARLA steps we linearly increase the train ratio of the planner to 64 but keep the
        # world model train ratio
        # Afterwards we keep the planner train ratio at 64 and the world model train ratio at 16
        if pretraining:
            available_train_steps_planner += 0.25
        else:
            available_train_steps_planner += min(
                max(0.25, (estimated_env_step - 800_000) / (2_000_000 - 800_000) * 0.75 + 0.25), 1
            )

        if available_train_steps_world_model > 1:
            available_train_steps_world_model -= 1
            available_train_steps_planner -= 1
            train_only_planner = False
        elif available_train_steps_planner > 1:
            available_train_steps_planner -= 1
            train_only_planner = True
        else:
            continue

        with embodied.timer.section("learner_train_step"):
            if train_only_planner:
                num_trained_only_planner += 1
                outs, carry, mets = agent.train_only_planner(batch, carry)
            else:
                num_trained_both += 1
                outs, carry, mets = agent.train(batch, carry)

        if "replay" in outs:
            with embodied.timer.section("learner_replay_update"):
                updater.update(outs["replay"])
        time.sleep(0.0001)
        agg.add(mets)
        fps.step(batch_steps)

        if should_eval():
            with embodied.timer.section("learner_eval"):
                mets, _ = agent.report(next(dataset_report), carry_report)
                logger.add(prefix(mets, "report"))

        if should_log():
            with embodied.timer.section("learner_metrics"):
                stats = {}
                stats.update(prefix(agg.result(), "train"))
                stats.update(prefix(embodied.timer.stats(), "timer/agent"))
                stats.update(prefix(usage.stats(), "usage/agent"))
                stats.update(prefix(logger.stats(), "client/learner_logger"))
                stats.update(prefix(replays[0].stats(), "client/learner_replay0"))
                stats.update({"fps/train": fps.result()})
                stats.update(
                    {"carla/trained_both": num_trained_both, "carla/trained_only_planner": num_trained_only_planner}
                )
            logger.add(stats)

        if should_save():
            checkpoint.save(logdir / f"checkpoint_{estimated_env_step}.ckpt")

        if should_clean():
            gc.collect()

        if should_reset_planner(estimated_env_step):
            agent.reset_planner()


def parallel_replay(make_replay, args):
    if isinstance(make_replay, bytes):
        make_replay = cloudpickle.loads(make_replay)

    replay = make_replay()
    dataset_train = iter(replay.dataset(args.batch_size, args.batch_length))
    dataset_report = iter(replay.dataset(args.batch_size, args.batch_length_eval))

    should_log = embodied.when.Clock(args.log_every)
    logger = embodied.distr.Client(args.logger_addr, "ReplayLogger", args.ipv6, maxinflight=1, connect=True)
    usage = embodied.Usage(**args.usage.update(nvsmi=False))

    should_save = embodied.when.Clock(args.save_every)
    cp = embodied.Checkpoint(embodied.Path(args.logdir) / "replay.ckpt")
    cp.replay = replay
    cp.load_or_save()

    def add_batch(data):
        envid = data.pop("envids")[0]
        for i in range(data["image"].shape[0]):
            replay.add({k: v[i] for k, v in data.items()}, envid)
        return {}

    server = embodied.distr.Server(args.replay_addr, "Replay", args.ipv6)
    server.bind("add_batch", add_batch, workers=1)
    server.bind("sample_batch_train", lambda _: next(dataset_train), workers=1)
    server.bind("sample_batch_report", lambda _: next(dataset_report), workers=1)
    server.bind("update", lambda data: replay.update(data), workers=1)
    with server:
        while True:
            server.check()
            should_save() and cp.save()
            time.sleep(1)
            if should_log():
                stats = prefix(replay.stats(), "replay")
                stats.update(prefix(embodied.timer.stats(), "timer/replay"))
                stats.update(prefix(usage.stats(), "usage/replay"))
                stats.update(prefix(logger.stats(), "client/replay_logger"))
                stats.update(prefix(server.stats(), "server/replay"))
                logger.add(stats)


def parallel_logger(make_logger, args):
    if isinstance(make_logger, bytes):
        make_logger = cloudpickle.loads(make_logger)

    logger = make_logger()
    should_log = embodied.when.Clock(args.log_every)
    usage = embodied.Usage(**args.usage.update(nvsmi=False))

    should_save = embodied.when.Clock(args.save_every)
    cp = embodied.Checkpoint(embodied.Path(args.logdir) / "logger.ckpt")
    cp.step = logger.step
    cp.load_or_save()

    parallel = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    updated = defaultdict(lambda: None)
    dones = defaultdict(lambda: True)

    log_keys_max = re.compile(args.log_keys_max)
    log_keys_sum = re.compile(args.log_keys_sum)
    log_keys_avg = re.compile(args.log_keys_avg)

    @embodied.timer.section("logger_addfn")
    def addfn(metrics):
        logger.add(metrics)

    @embodied.timer.section("logger_transfn")
    def transfn(trans):
        now = time.time()
        envids = trans.pop("envids")
        logger.step.increment(len(trans["is_first"]))
        parallel.add("ep_starts", trans["is_first"].sum(), agg="sum")
        parallel.add("ep_ends", trans["is_last"].sum(), agg="sum")

        for i, addr in enumerate(envids):
            tran = {k: v[i] for k, v in trans.items()}

            updated[addr] = now
            episode = episodes[addr]
            if tran["is_first"]:
                episode.reset()
                parallel.add("ep_abandoned", int(not dones[addr]), agg="sum")
            dones[addr] = tran["is_last"]

            episode.add("score", tran["reward"], agg="sum")
            episode.add("length", 1, agg="sum")
            episode.add("rewards", tran["reward"], agg="stack")

            video_addrs = list(episodes.keys())[: args.log_video_streams]
            if addr in video_addrs:
                for key in args.log_keys_video:
                    if key in tran:
                        episode.add(f"policy_{key}", tran[key], agg="stack")

            for key in trans.keys():
                if log_keys_max.match(key):
                    episode.add(key, tran[key], agg="max")
                if log_keys_sum.match(key):
                    episode.add(key, tran[key], agg="sum")
                if log_keys_avg.match(key):
                    episode.add(key, tran[key], agg="avg")

            if tran["is_last"]:
                result = episode.result()
                logger.add(
                    {
                        "score": result.pop("score"),
                        "length": result.pop("length") - 1,
                    },
                    prefix="episode",
                )
                rew = result.pop("rewards")
                if len(rew) > 1:
                    result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
                epstats.add(result)

        for addr, last in list(updated.items()):
            if now - last >= args.log_episode_timeout:
                print("Dropping episode statistics due to timeout.")
                del episodes[addr]
                del updated[addr]

    server = embodied.distr.Server(args.logger_addr, "Logger", args.ipv6)
    server.bind("add", addfn)
    server.bind("trans", transfn)
    with server:
        while True:
            server.check()
            should_save() and cp.save()
            time.sleep(1)
            if should_log():
                with embodied.timer.section("logger_metrics"):
                    logger.add(parallel.result(), prefix="parallel")
                    logger.add(epstats.result(), prefix="epstats")
                    logger.add(embodied.timer.stats(), prefix="timer/logger")
                    logger.add(usage.stats(), prefix="usage/logger")
                    logger.add(server.stats(), prefix="server/logger")
                logger.write()


def parallel_env(make_env, envid, args, logging=False):
    if isinstance(make_env, bytes):
        make_env = cloudpickle.loads(make_env)
    assert envid >= 0, envid
    name = f"Env{envid}"

    _print = lambda x: embodied.print(f"[{name}] {x}", flush=True)
    should_log = embodied.when.Clock(args.log_every)
    if logging:
        logger = embodied.distr.Client(args.logger_addr, f"{name}Logger", args.ipv6, maxinflight=1, connect=True)
    fps = embodied.FPS()
    if envid == 0:
        usage = embodied.Usage(**args.usage.update(nvsmi=False))

    _print("Make env")
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
        fps.step(1)
        done = obs["is_last"]
        if done:
            _print(f"Episode of length {length} with score {score:.2f}")

        with embodied.timer.section("env_request"):
            future = actor.act({"envid": envid, **obs})
        try:
            with embodied.timer.section("env_response"):
                act = future.result()
        except embodied.distr.NotAliveError:
            # Wait until we are connected again, so we don't unnecessarily reset the
            # environment hundreds of times while the server is unavailable.
            _print("Lost connection to server")
            actor.connect()
            done = True
        except embodied.distr.RemoteError as e:
            _print(f"Shutting down env due to agent error: {e}")
            sys.exit(0)

        if should_log() and logging and envid == 0:
            stats = {f"fps/env{envid}": fps.result()}
            stats.update(prefix(usage.stats(), f"usage/env{envid}"))
            stats.update(prefix(logger.stats(), f"client/env{envid}_logger"))
            stats.update(prefix(actor.stats(), f"client/env{envid}_actor"))
            stats.update(prefix(embodied.timer.stats(), f"timer/env{envid}"))
            logger.add(stats)
