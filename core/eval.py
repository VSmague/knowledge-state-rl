"""
core/eval.py — Generic evaluation: run N episodes per goal, return success
rates. Fully game-agnostic — works through Agent.act(), which already
accepts raw goal data in whatever form each game uses (minigrid_game's
(x,y) tuples, crafter_game's achievement strings), via each game's
GoalRepr.

Vectorized: reuses ONE env per goal across all its episodes WITHIN one
evaluate_all() call (no extra env-construction overhead), batching the
network forward pass across every still-active goal within an "episode
round" instead of one goal at a time (~2.75x speedup on CPU over a naive
one-goal-at-a-time loop).

env_cache (optional, both functions below): without it, every call to
evaluate_all() constructs a FRESH Env per goal and closes it at the end —
fine for minigrid_game/crafter_game (cheap pure-Python sims), but
core/train.py calls evaluate_all() once per checkpoint over a whole run
(e.g. 10 times), and for fetch_game specifically, gymnasium_robotics/
MuJoCo's env.close() does NOT release the ~19 file descriptors gym.make()
opens per scene (a real, verified upstream leak) — recreating 100 goals'
worth of envs on EVERY checkpoint compounds across the whole run and
crashed a real Jean Zay job with "Too many open files" after ~9
checkpoints (~900 cumulative envs created, right around the fd limit /
~19 per env).
Passing a persistent dict as env_cache (kept alive across the WHOLE
training run by core/train.py, not just one evaluate_all() call) makes
each goal's Env get constructed ONCE, ever, for the entire run, then only
reset() between rounds — matching how these environments are normally
used everywhere else (create once, reset repeatedly, never reconstruct)
and eliminating the leak's per-checkpoint compounding entirely rather
than just buying more headroom against it.
"""

from collections import defaultdict

import numpy as np
import torch


def select_eval_goals(goals: list, group_fn, eval_goals_per_group: int = None,
                      eval_max_goals: int = None) -> list:
    """
    Pick which goals a checkpoint eval should actually run, out of the
    full `goals` list — independent of the mechanics of running episodes
    (that's evaluate_all's job). Sample ONCE per training run (not
    per-checkpoint) so every checkpoint of that run evaluates the SAME
    subsample — a stable, comparable learning curve, rather than a
    different random slice each time. Relies on the caller's global numpy
    RNG already being seeded (e.g. np.random.seed(args.seed)) for
    reproducibility.

    eval_goals_per_group : if given, sample this many goals WITH
        replacement from each group (group_fn(goal)), then deduplicate —
        evaluate_all()'s {goal: rate} return can't represent the same
        goal twice anyway (a later duplicate would just silently
        overwrite an earlier one's result, wasting the eval episodes
        spent on it), so a group with collisions ends up with somewhat
        FEWER than eval_goals_per_group unique goals, not an error. If
        group_fn is None, subsampling is skipped entirely (every goal is
        its own group, so there is nothing meaningful to subsample).
    eval_max_goals : flat cap, independent of grouping — the fix for
        games with no natural grouping (group_fn=None or identity, so
        eval_goals_per_group can't reduce anything): sampled WITHOUT
        replacement, applied on top of whatever eval_goals_per_group
        already produced.
    """
    eval_goals = goals
    if eval_goals_per_group is not None:
        if group_fn is None:
            print("  [warn] eval_goals_per_group set but no group_fn -- "
                 "evaluating every goal, subsampling skipped.")
        else:
            by_group = defaultdict(list)
            for g in goals:
                by_group[group_fn(g)].append(g)
            eval_goals = []
            for grp_goals in by_group.values():
                idx = np.random.choice(len(grp_goals), size=eval_goals_per_group,
                                       replace=True)
                eval_goals.extend(dict.fromkeys(grp_goals[i] for i in idx))
            print(f"  [eval subsample] {len(eval_goals)} unique eval goals "
                 f"(sampled with replacement, up to {eval_goals_per_group}/group "
                 f"x {len(by_group)} groups) vs {len(goals)} total goals")

    if eval_max_goals is not None and len(eval_goals) > eval_max_goals:
        before = len(eval_goals)
        idx = np.random.choice(len(eval_goals), size=eval_max_goals, replace=False)
        eval_goals = [eval_goals[i] for i in idx]
        print(f"  [eval max-goals cap] {len(eval_goals)} eval goals "
             f"(randomly capped from {before}, eval_max_goals={eval_max_goals})")

    return eval_goals


def group_success_from_goal_results(results: dict, group_fn) -> dict:
    """Aggregate {goal: success_rate} -> {group: {"mean":..., "std":...}}
    (std = spread across the goals that fall into that group). group_fn
    may be None (each goal is its own group)."""
    buckets = defaultdict(list)
    for g, sr in results.items():
        grp = group_fn(g) if group_fn is not None else g
        buckets[grp].append(sr)
    return {grp: dict(mean=float(np.mean(v)), std=float(np.std(v)))
            for grp, v in buckets.items()}


def _evaluate_goals_chunk(agent, make_env_fn, goals: list, n_episodes: int,
                          device: str, seed_offset: int,
                          env_cache: dict = None) -> np.ndarray:
    """
    goals : a chunk of goals to evaluate together.
    Runs n_episodes ROUNDS, reusing ONE env per goal across all rounds
    (reset() with a new seed each round). Within each round, the agent's
    forward pass is batched across every still-active goal.

    env_cache : optional dict, goal -> Env, persisted by the CALLER across
    multiple evaluate_all() calls (see module docstring). If given, envs
    are built at most once per goal (ever) and never closed here — the
    caller owns their lifetime. If None (default), envs are built fresh
    and closed at the end of this call, as before.

    Returns an int array: number of successes per goal (out of n_episodes).
    """
    n = len(goals)
    if env_cache is None:
        envs = [make_env_fn(goal=g) for g in goals]
    else:
        envs = []
        for g in goals:
            if g not in env_cache:
                env_cache[g] = make_env_fn(goal=g)
            envs.append(env_cache[g])
    success_counts = np.zeros(n, dtype=np.int32)

    for ep in range(n_episodes):
        obs = []
        for env in envs:
            o, _ = env.reset(seed=seed_offset + ep)
            obs.append(o)

        done       = np.zeros(n, dtype=bool)
        term_flags = np.zeros(n, dtype=bool)

        while not done.all():
            active = np.where(~done)[0]
            obs_batch = torch.from_numpy(np.stack([obs[i] for i in active])).to(device)
            goal_batch = [goals[i] for i in active]

            with torch.no_grad():
                action, _, _, _ = agent.act(obs_batch, goal_batch)
            actions_np = action.cpu().numpy()

            for k, i in enumerate(active):
                o, _, terminated, truncated, _ = envs[i].step(int(actions_np[k]))
                obs[i] = o
                if terminated or truncated:
                    done[i]       = True
                    term_flags[i] = terminated

        success_counts += term_flags.astype(np.int32)

    if env_cache is None:
        for env in envs:
            env.close()
    return success_counts


def evaluate_all(agent, make_env_fn, goals: list, n_episodes: int,
                 device: str, seed_offset: int = 1000,
                 max_parallel: int = 512, env_cache: dict = None,
                 eval_pool=None) -> dict:
    """
    Return {goal: success_rate} for every goal.

    max_parallel caps how many goals are evaluated together in one chunk
    (bounds memory for very large goal counts) — still relevant even with
    env_cache: it bounds the peak concurrent envs open during the FIRST
    call that populates the cache (later calls just reset() the same
    envs, so max_parallel no longer matters for those).

    env_cache : see module docstring — pass a dict that outlives this one
    call (e.g. built once in core/train.py's train() and reused across
    every checkpoint) to build each goal's Env at most once for the whole
    run instead of once per call. Ignored when eval_pool is given.

    eval_pool : an core/eval_parallel.EvalStreamPool, or None (default:
    serial, single-process env-stepping via env_cache, as before). See
    that module's docstring — a large goal count under an
    early/undertrained policy makes serial env-stepping the real
    checkpoint-eval bottleneck (episodes rarely terminate early, so
    every goal runs close to the max episode length, times
    n_episodes). eval_pool streams (goal, episode) work through
    pool.n_workers persistent workers, immediately handing a worker the
    next task the instant it frees up (exactly core/train.py's own PPO
    rollout loop -- fixed worker count, immediate goal reassignment on
    episode end), instead of the serial path's "every goal marches
    through its episodes in lockstep, one shared round at a time"
    structure -- so results are NOT expected to bit-match the serial
    path (different, faster algorithm, not a parallel port of the same
    one); see tests/test_eval_streaming_parallel_matches_serial.py for
    how it's actually validated. max_parallel/chunking is IGNORED when
    eval_pool is given -- the whole goal list streams through the fixed
    worker pool in one pass, no chunk boundary needed.
    """
    was_training = getattr(agent, "encoder", None) is not None and agent.encoder.training
    agent.encoder.eval(); agent.goal_repr.eval(); agent.ac.eval()

    if eval_pool is not None:
        from core.eval_parallel import evaluate_all_streaming
        result = evaluate_all_streaming(eval_pool, agent, goals, n_episodes, device,
                                        seed_offset=seed_offset)
        if was_training:
            agent.encoder.train(); agent.goal_repr.train(); agent.ac.train()
        return result

    n_goals = len(goals)
    success_counts = np.zeros(n_goals, dtype=np.int32)

    for start in range(0, n_goals, max_parallel):
        chunk = goals[start:start + max_parallel]
        success_counts[start:start + len(chunk)] = _evaluate_goals_chunk(
            agent, make_env_fn, chunk, n_episodes, device, seed_offset,
            env_cache=env_cache)

    if was_training:
        agent.encoder.train(); agent.goal_repr.train(); agent.ac.train()

    return {g: float(success_counts[i]) / n_episodes for i, g in enumerate(goals)}
