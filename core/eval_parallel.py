"""
core/eval_parallel.py — Multi-process parallel evaluation.

Mirrors core/parallel_envs.py's ParallelEnvs: N persistent worker
processes, each holding exactly ONE Env, reused across MANY different
goals over its lifetime via env.set_goal()+reset() — the same pattern
training already uses to swap goals into a live env slot, which keeps
each worker doing the same amount of work per round as one of
core/train.py's own training workers, regardless of how many goals are
concurrently active (unlike pinning a fixed, small number of goals to
each worker and stepping them serially, which does increasingly more
work per worker as the concurrently-active goal count grows into the
hundreds). A simple work QUEUE of (goal, episode_index) tasks is drained
by whichever slot frees up next — exactly training's "episode ends ->
feed that slot a new goal immediately" loop — so every worker stays
continuously busy regardless of how unevenly episodes finish, matching
training's own throughput.

each worker's persistent Env is only ever constructed ONCE (first
"reset" call) then reused for the rest of the pool's lifetime —
preserves the "construct once, never recreate" invariant (matters most
for fetch_game's MuJoCo fd leak on env.close(), see core/eval.py's module
docstring) even though which GOAL a given worker is running changes
constantly.

Reproducibility: unlike core/eval.py's serial _evaluate_goals_chunk
(all goals march through episodes in lockstep, one round per env-step,
same batch every round), which goal is in which slot at any given
round here depends on episode-completion timing, so this does NOT
reproduce the serial path's exact per-episode RNG draw sequence/action
sampling order. It's a genuinely different (correct, faster) algorithm,
not a parallel port of the same one — validated instead by
tests/test_eval_streaming_parallel_matches_serial.py, which checks (a)
every goal gets exactly n_episodes episodes accounted for (deterministic
by construction) and (b), for minigrid_game/fetch_game, that a
single-process reference of this SAME queue-based algorithm
(evaluate_all_streaming_serial below) gives bit-identical results to
the multi-process version — isolating "is the pool-dispatch/queue logic
correct" from "stochastic action sampling order is round-structure-
dependent" (true of the OLD design too, not something this file
introduces). crafter_game is the one exception: its underlying engine
consumes some randomness during step() that isn't fully pinned down by
the per-episode seed, so a shared process and independent worker
subprocesses genuinely diverge in outcome even with correct dispatch —
see the test file for the empirical check that ruled out the monkey-
patching in _disable_survival_mechanics as the cause.
"""

import multiprocessing as mp
import time

import numpy as np
import torch


def _eval_stream_worker(conn, make_env_fn):
    env = None
    while True:
        cmd, data = conn.recv()

        if cmd == "reset":
            goal, seed = data
            if env is None:
                env = make_env_fn(goal=goal)
            else:
                env.set_goal(goal)
            obs, _ = env.reset(seed=seed)
            conn.send(obs)

        elif cmd == "step":
            obs, _reward, term, trunc, _info = env.step(data)
            conn.send((obs, bool(term), bool(trunc)))

        elif cmd == "close":
            if env is not None:
                env.close()
            conn.close()
            return


class EvalStreamPool:
    """n_workers persistent subprocesses, 1:1 with 'slots'. See module
    docstring — each worker holds exactly one Env, reused across many
    goals over the pool's lifetime."""

    def __init__(self, n_workers: int, make_env_fn):
        assert n_workers >= 1
        self.n_workers = n_workers
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_eval_stream_worker, args=(child_conn, make_env_fn),
                           daemon=True)
            p.start()
            child_conn.close()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def reset_slots(self, slot_goal_seed: dict) -> dict:
        """{slot_i: (goal, seed)} -> {slot_i: obs}."""
        for i, (goal, seed) in slot_goal_seed.items():
            self.conns[i].send(("reset", (goal, seed)))
        return {i: self.conns[i].recv() for i in slot_goal_seed}

    def step_slots(self, slot_action: dict) -> dict:
        """{slot_i: action} -> {slot_i: (obs, term, trunc)}."""
        for i, action in slot_action.items():
            self.conns[i].send(("step", action))
        return {i: self.conns[i].recv() for i in slot_action}

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


def _drain_queue_step(tasks_iter, slot_goal, active, i):
    """Pull the next (goal, episode_idx) task for slot i, or mark it
    inactive if the queue is exhausted. Returns the task or None."""
    try:
        g, ep = next(tasks_iter)
    except StopIteration:
        active[i] = False
        return None
    slot_goal[i] = g
    active[i] = True
    return g, ep


def evaluate_all_streaming(pool, agent, goals: list, n_episodes: int,
                           device: str, seed_offset: int = 1000,
                           progress_every: float = 0.1) -> dict:
    """N=pool.n_workers fixed slots, a work queue of (goal, episode_idx)
    tasks drained as slots free up. See module docstring. Returns
    {goal: success_rate}, same contract as core/eval.py's evaluate_all
    (minus the was_training/eval-mode bookkeeping, which the caller
    still does).

    progress_every : print a progress line (episodes done/total,
    elapsed, ETA) every time this fraction of total episodes completes
    (e.g. 0.1 -> every ~10%). 0/None disables. Without it this call is a
    total black box until it returns -- with a large goal count and a
    still-failing policy (most episodes running to their full max
    length), one checkpoint's eval can take minutes with no way to tell
    "still working" from "stuck"."""
    N = pool.n_workers
    n_total_episodes = len(goals) * n_episodes
    tasks_iter = iter([(g, ep) for g in goals for ep in range(n_episodes)])
    success_counts = {g: 0 for g in goals}

    slot_goal = [None] * N
    active = [False] * N
    obs = [None] * N

    n_done = 0
    next_progress_at = max(1, int(n_total_episodes * progress_every)) if progress_every else None
    t_start = time.time()

    resets = {}
    for i in range(N):
        task = _drain_queue_step(tasks_iter, slot_goal, active, i)
        if task is not None:
            g, ep = task
            resets[i] = (g, seed_offset + ep)
    for i, o in pool.reset_slots(resets).items():
        obs[i] = o

    while any(active):
        idxs = [i for i in range(N) if active[i]]
        obs_batch = torch.from_numpy(np.stack([obs[i] for i in idxs])).to(device)
        goal_batch = [slot_goal[i] for i in idxs]

        with torch.no_grad():
            action, _, _, _ = agent.act(obs_batch, goal_batch)
        actions_np = action.cpu().numpy()

        step_map = {i: int(actions_np[k]) for k, i in enumerate(idxs)}
        results = pool.step_slots(step_map)

        resets = {}
        for i in idxs:
            o, term, trunc = results[i]
            if term or trunc:
                if term:
                    success_counts[slot_goal[i]] += 1
                n_done += 1
                if next_progress_at is not None and n_done >= next_progress_at:
                    dt = time.time() - t_start
                    rate = n_done / dt if dt > 0 else 0.0
                    eta = (n_total_episodes - n_done) / rate if rate > 0 else float("nan")
                    print(f"    [eval progress] {n_done}/{n_total_episodes} episodes "
                         f"({100*n_done/n_total_episodes:.0f}%)  "
                         f"elapsed={dt:.0f}s  eta={eta:.0f}s")
                    next_progress_at += max(1, int(n_total_episodes * progress_every))
                task = _drain_queue_step(tasks_iter, slot_goal, active, i)
                if task is not None:
                    g, ep = task
                    resets[i] = (g, seed_offset + ep)
            else:
                obs[i] = o

        if resets:
            for i, o in pool.reset_slots(resets).items():
                obs[i] = o

    return {g: success_counts[g] / n_episodes for g in goals}


def evaluate_all_streaming_serial(make_env_fn, agent, goals: list, n_episodes: int,
                                  device: str, seed_offset: int = 1000,
                                  n_slots: int = 8) -> dict:
    """Single-process reference implementation of the EXACT SAME
    queue-based algorithm as evaluate_all_streaming (n_slots persistent
    Envs, reused via set_goal()+reset(), work queue drained as slots
    free up) -- used ONLY to validate EvalStreamPool's multi-process
    dispatch against a bit-identical single-process baseline (see
    tests/test_eval_streaming_parallel_matches_serial.py). Not used by
    core/train.py."""
    envs = [None] * n_slots
    tasks_iter = iter([(g, ep) for g in goals for ep in range(n_episodes)])
    success_counts = {g: 0 for g in goals}

    slot_goal = [None] * n_slots
    active = [False] * n_slots
    obs = [None] * n_slots

    def _reset_slot(i, g, seed):
        if envs[i] is None:
            envs[i] = make_env_fn(goal=g)
        else:
            envs[i].set_goal(g)
        o, _ = envs[i].reset(seed=seed)
        return o

    for i in range(n_slots):
        task = _drain_queue_step(tasks_iter, slot_goal, active, i)
        if task is not None:
            g, ep = task
            obs[i] = _reset_slot(i, g, seed_offset + ep)

    while any(active):
        idxs = [i for i in range(n_slots) if active[i]]
        obs_batch = torch.from_numpy(np.stack([obs[i] for i in idxs])).to(device)
        goal_batch = [slot_goal[i] for i in idxs]

        with torch.no_grad():
            action, _, _, _ = agent.act(obs_batch, goal_batch)
        actions_np = action.cpu().numpy()

        for k, i in enumerate(idxs):
            o, _reward, term, trunc, _info = envs[i].step(int(actions_np[k]))
            if term or trunc:
                if term:
                    success_counts[slot_goal[i]] += 1
                task = _drain_queue_step(tasks_iter, slot_goal, active, i)
                if task is not None:
                    g, ep = task
                    obs[i] = _reset_slot(i, g, seed_offset + ep)
            else:
                obs[i] = o

    return {g: success_counts[g] / n_episodes for g in goals}
