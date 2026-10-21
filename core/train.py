"""
core/train.py — Generic goal-conditioned PPO with curriculum learning.
Works identically for minigrid_game, crafter_game, and fetch_game (or any
future game following the same Env/Agent contract) via GAME_REGISTRY
dispatch.

Usage:
    python -m core.train --game minigrid_game --curriculum magellan_ks --steps 5_000_000
    python -m core.train --game crafter_game  --curriculum uniform     --steps 5_000_000
    python -m core.train --game fetch_game    --curriculum magellan    --steps 5_000_000

Reproducibility of training environments: each worker gets a LOCAL RNG
seeded deterministically from args.seed (+ worker index), which draws a
fresh episode seed on every single reset() — so two full re-runs with the
same --seed replay byte-for-byte identical sequences of training layouts
(same door colors, same key positions, in the same order), while episodes
WITHIN one run still vary genuinely from each other (for generalization
across colors/positions). Evaluation (core/eval.py) uses its own fixed
seed_offset per round, for a fair comparison across checkpoints.
"""

import argparse
import functools
import json
import os
import time
import multiprocessing as mp
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

from core.parallel_envs import ParallelEnvs
from core.curriculum import build_curriculum, CURRICULUM_REGISTRY
from core.eval import evaluate_all, select_eval_goals, group_success_from_goal_results

import crafter_game.env as crafter_env
import crafter_game.architecture as crafter_arch
import fetch_game.env as fetch_game_env
import fetch_game.architecture as fetch_game_arch
import minigrid_game.env as minigrid_game_env
import minigrid_game.architecture as minigrid_game_arch


GAME_REGISTRY = {
    "crafter_game":  {"env": crafter_env,       "architecture": crafter_arch},
    "fetch_game":    {"env": fetch_game_env,    "architecture": fetch_game_arch},
    "minigrid_game": {"env": minigrid_game_env, "architecture": minigrid_game_arch},
}

# fetch_game uses size, but differently from the other two: size[0] = N,
# the number of random (start,target) goals (see fetch_game/env.py) —
# DEFAULT_SIZE below is only the fallback when --size is omitted
# entirely; goals() has its own DEFAULT_N for the same fallback value,
# kept in sync manually.
# minigrid_game's size is (n_cols, n_rows, w, h) — see its env.py's
# module docstring; default (5,5,5,5) is a 5x5 grid of 5x5 rooms.
DEFAULT_SIZE = {"crafter_game": (64, 64), "fetch_game": (30,), "minigrid_game": (5, 5, 5, 5)}


def build_goal_vectors(game_name: str, env_module, goals: list, size: tuple,
                       embeddings_path: str = None,
                       embeddings_matrix=None):
    """
    Fixed numeric goal vectors for the curriculum competence MLP / DeepSet
    — thin dispatcher to the game module's own build_goal_vectors (part of
    the shared Env contract every game module implements, see readme).
    game_name is unused (env_module already determines the dispatch) but
    kept for call-site compatibility.
    """
    return env_module.build_goal_vectors(
        goals, size, embeddings_path=embeddings_path,
        embeddings_matrix=embeddings_matrix)


def build_group_fn(game_name: str, env_module, size: tuple):
    """group_fn for the curriculum's per-group ALP/entropy logging — only
    minigrid_game customizes this (per-room grouping), so it's an OPTIONAL
    hook: None (identity grouping) if the game module doesn't define one.
    game_name is unused, kept for call-site compatibility (see
    build_goal_vectors above)."""
    if hasattr(env_module, "build_group_fn"):
        return env_module.build_group_fn(size)
    return None


def _group_of(goal, group_fn):
    return group_fn(goal) if group_fn is not None else goal


def _fmt_group(g) -> str:
    """Short label for a group id — works for any hashable group label
    (an int room index, a (col, row) room tuple, a crafter_game
    achievement string, ...) without any game-specific branch."""
    if isinstance(g, (int, np.integer)):
        return f"g{int(g)}"
    s = str(g)
    return s if len(s) <= 18 else s[:16] + ".."


def _fmt_group_rates(rate_by_group: dict, count_by_group: dict = None,
                     alp_by_group: dict = None, max_items: int = 32) -> str:
    """Compact 'g0=99%(120,a=.012) g1=40%(80,a=.003) ...' line, env-agnostic."""
    if not rate_by_group:
        return "(no data)"
    keys = sorted(rate_by_group.keys(), key=lambda x: (str(type(x)), x))
    parts = []
    for g in keys[:max_items]:
        r = rate_by_group[g]
        extras = []
        if count_by_group is not None and g in count_by_group:
            extras.append(str(count_by_group[g]))
        if alp_by_group is not None and g in alp_by_group:
            extras.append(f"a={alp_by_group[g]:.3f}")
        if extras:
            parts.append(f"{_fmt_group(g)}={r:.0%}({','.join(extras)})")
        else:
            parts.append(f"{_fmt_group(g)}={r:.0%}")
    if len(keys) > max_items:
        parts.append(f"...+{len(keys) - max_items}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def gae(rewards, values, dones, last_values, gamma, lam):
    """rewards, values, dones: (T, N). last_values: (N,)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nextvalues = last_values if t == T - 1 else values[t + 1]
        nextnonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


# ---------------------------------------------------------------------------
# PPO update — 100% game-agnostic: only ever calls agent.evaluate_actions()
# ---------------------------------------------------------------------------

def ppo_update(agent, opt, buf, advantages, returns, args, device):
    T, N = buf["actions"].shape
    b_obs   = buf["obs"].reshape(T * N, *buf["obs"].shape[2:])
    b_goals = buf["goals"].reshape(T * N).tolist()
    b_act   = buf["actions"].reshape(T * N)
    b_logp  = buf["log_probs"].reshape(T * N)
    b_adv   = advantages.reshape(T * N)
    b_ret   = returns.reshape(T * N)
    b_adv   = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    obs_t  = torch.from_numpy(b_obs).to(device)
    act_t  = torch.from_numpy(b_act).long().to(device)
    logp_t = torch.from_numpy(b_logp).to(device)
    adv_t  = torch.from_numpy(b_adv).to(device)
    ret_t  = torch.from_numpy(b_ret).to(device)

    mb_size = max(1, (T * N) // args.num_minibatches)
    stats = dict(pg_loss=0.0, v_loss=0.0, entropy=0.0)
    n_upd = 0

    for _ in range(args.ppo_epochs):
        perm = np.random.permutation(T * N)
        for start in range(0, T * N, mb_size):
            mb_idx   = perm[start:start + mb_size]
            mb_idx_t = torch.from_numpy(mb_idx).long()

            mb_obs      = obs_t[mb_idx_t]
            mb_goals    = [b_goals[i] for i in mb_idx]
            mb_act      = act_t[mb_idx_t]
            mb_old_logp = logp_t[mb_idx_t]
            mb_adv      = adv_t[mb_idx_t]
            mb_ret      = ret_t[mb_idx_t]

            new_logp, entropy, value = agent.evaluate_actions(mb_obs, mb_goals, mb_act)

            ratio    = (new_logp - mb_old_logp).exp()
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()
            v_loss   = F.mse_loss(value, mb_ret)
            ent      = entropy.mean()

            loss = pg_loss + 0.5 * v_loss - args.entropy_coef * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()

            stats["pg_loss"] += pg_loss.item()
            stats["v_loss"]  += v_loss.item()
            stats["entropy"] += ent.item()
            n_upd += 1

    for k in stats:
        stats[k] /= max(n_upd, 1)
    return stats


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entropy per group -- computed from the rollout buffer just collected
# ---------------------------------------------------------------------------

def compute_group_entropy(agent, buf_obs, buf_goals, group_fn, device,
                          max_samples_per_group: int = 256):
    """Mean policy entropy per group, from this rollout's transitions.
    group_fn=None -> each goal is its own group (matches curriculum's
    default identity grouping).

    Subsamples each group to at most max_samples_per_group transitions
    instead of running the whole buffer through the network — this used
    to be a full extra forward pass over EVERY collected transition, once
    per rollout, on top of the PPO update itself (measured to roughly
    halve throughput on crafter_game with its heavier CNN). A few hundred
    samples per group already gives a stable mean-entropy estimate, so
    the accuracy cost of subsampling is negligible relative to the
    speedup."""
    T, N = buf_obs.shape[:2]
    flat_obs   = buf_obs.reshape(T * N, *buf_obs.shape[2:])
    flat_goals = buf_goals.reshape(T * N).tolist()
    gfn = group_fn or (lambda g: g)

    group_indices = defaultdict(list)
    for idx, g in enumerate(flat_goals):
        group_indices[gfn(g)].append(idx)

    result = {}
    agent.encoder.eval(); agent.goal_repr.eval(); agent.ac.eval()
    with torch.no_grad():
        for group, idxs in group_indices.items():
            if len(idxs) > max_samples_per_group:
                idxs = list(np.random.choice(idxs, max_samples_per_group, replace=False))
            chunk_size = 512
            entropies = []
            for start in range(0, len(idxs), chunk_size):
                chunk = idxs[start:start + chunk_size]
                obs_t = torch.from_numpy(flat_obs[chunk]).to(device)
                goals_c = [flat_goals[i] for i in chunk]
                _, _, entropy, _ = agent.act(obs_t, goals_c)
                entropies.append(entropy.mean().item())
            result[group] = float(np.mean(entropies))
    agent.encoder.train(); agent.goal_repr.train(); agent.ac.train()
    return result


def save_checkpoint(agent, estimator, global_step, ep_count, total_success,
                    args, out_dir, pct_label, size, goal_dim):
    ckpts_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)
    path = os.path.join(ckpts_dir, f"ckpt_pct{pct_label:03d}.pt")
    payload = dict(
        encoder=agent.encoder.state_dict(),
        goal_repr=agent.goal_repr.state_dict(),
        ac=agent.ac.state_dict(),
        step=global_step, episode=ep_count, total_success=total_success,
        pct=pct_label, game=args.game, curriculum=args.curriculum,
        size=size, enc_dim=args.enc_dim, hidden_dim=args.hidden_dim,
        goal_dim=goal_dim)
    if hasattr(estimator, "current_mlp"):
        payload["curriculum_mlp"] = {
            k: v.detach().cpu().clone()
            for k, v in estimator.current_mlp.state_dict().items()}
    if args.curriculum == "magellan_ks":
        payload["ks_dim"] = args.ks_dim
        payload["ks_item_hidden"] = args.ks_item_hidden
    torch.save(payload, path)
    print(f"  [ckpt] saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    import random as _random

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, args.torch_threads))

    if args.seed is not None:
        # Reproducible mode: every source of randomness this process
        # controls directly gets seeded from args.seed, including Python's
        # native random.seed() — Uniform's sample_uniform() uses
        # random.choice(), which np.random.seed() alone doesn't cover.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)
    # else: seed=None means "fully random, don't care about reproducing
    # this run" — torch/numpy/random are left un-seeded (drawing from OS
    # entropy by default), and ParallelEnvs below also gets seed=None,
    # which each worker turns into an independently OS-entropy-seeded
    # local RNG (see core/parallel_envs.py) rather than a deterministic
    # per-worker sequence.

    game        = GAME_REGISTRY[args.game]
    env_module  = game["env"]
    arch_module = game["architecture"]

    size   = tuple(args.size) if args.size else DEFAULT_SIZE[args.game]
    length = args.length

    env_kwargs = dict(size=size, length=length, shaping=args.shaping)
    if hasattr(env_module, "extra_env_kwargs"):
        env_kwargs.update(env_module.extra_env_kwargs(args))
    goals = env_module.goals(size, subset=args.crafter_goals)
    make_env_fn = functools.partial(env_module.Env, **env_kwargs)

    # Game-specific goal representation (policy vs curriculum), if the
    # game module provides one — see crafter_game.env.build_train_goal_repr
    # for why policy and curriculum need to be built separately there.
    # Games with no such notion (minigrid_game, fetch_game) just skip
    # this entirely.
    frozen_embeddings_path = None
    policy_embeddings_matrix = None
    curriculum_embeddings_matrix = None
    if hasattr(env_module, "build_train_goal_repr"):
        (policy_embeddings_matrix, frozen_embeddings_path,
         curriculum_embeddings_matrix) = env_module.build_train_goal_repr(args, goals)

    goal_vectors, goal_dim = build_goal_vectors(
        args.game, env_module, goals, size,
        embeddings_path=None,
        embeddings_matrix=curriculum_embeddings_matrix)
    group_fn = build_group_fn(args.game, env_module, size)

    # See --eval_goals_per_group/--eval_max_goals's help text.
    eval_goals = select_eval_goals(goals, group_fn,
                                   eval_goals_per_group=args.eval_goals_per_group,
                                   eval_max_goals=args.eval_max_goals)

    run_suffix = f"_{args.run_name}" if args.run_name else ""
    out_dir = os.path.join(
        args.exp_root, f"{args.game}_{args.curriculum}_{args.steps}steps{run_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    if policy_embeddings_matrix is not None:
        tt_path = os.path.join(out_dir, "tech_tree_goal_embeddings.npy")
        np.save(tt_path, policy_embeddings_matrix)
        frozen_embeddings_path = tt_path
        print(f"  [goal repr] tech_tree embeddings -> {tt_path}  "
              f"shape={policy_embeddings_matrix.shape}")

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 62)
    print(f"  {args.game.upper()} / {args.curriculum.upper()}  "
         f"-  size={size}  -  {len(goals)} goals")
    print(f"  Steps: {args.steps:,}  device={device}  seed={args.seed}")
    if args.shaping:
        reward_desc = (env_module.shaping_description()
                       if hasattr(env_module, "shaping_description") else "shaped")
    else:
        reward_desc = "binary"
    print(f"  reward={reward_desc}")
    if hasattr(env_module, "describe_goal_embeds"):
        print(f"  {env_module.describe_goal_embeds(frozen_embeddings_path, args.crafter_goal_repr, goal_dim)}")
    if hasattr(arch_module, "describe_agent_hparams"):
        print(f"  {arch_module.describe_agent_hparams(args)}")
    print(f"  Out dir: {out_dir}")
    print("=" * 62 + "\n")

    # -- Agent --------------------------------------------------------------
    agent_kwargs = dict(enc_dim=args.enc_dim, hidden_dim=args.hidden_dim)
    if hasattr(arch_module, "extra_agent_kwargs"):
        agent_kwargs.update(arch_module.extra_agent_kwargs(args, frozen_embeddings_path))
    agent = arch_module.build_agent(size, device=device, **agent_kwargs)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    total_updates = max(1, args.steps // (args.rollout_len * args.n_envs))
    scheduler = None
    if not args.no_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_updates)

    # -- Curriculum -----------------------------------------------------------
    cur_kwargs = dict(eps_min=args.eps_min, device=device,
                      candidate_subsample=args.curriculum_candidate_subsample)
    if args.curriculum == "online":
        cur_kwargs.update(M=args.window)
    if args.curriculum == "magellan":
        cur_kwargs.update(goal_vectors=goal_vectors, goal_dim=goal_dim,
                          M=args.magellan_buffer, N=args.magellan_snapshots,
                          train_every=args.magellan_train_every,
                          alp_deadzone=args.alp_deadzone,
                          balanced_training=args.balanced_training)
    if args.curriculum == "magellan_ks":
        cur_kwargs.update(goal_vectors=goal_vectors, goal_dim=goal_dim,
                          N=args.magellan_snapshots, train_every=args.magellan_train_every,
                          K=args.ks_window, item_hidden=args.ks_item_hidden,
                          knowledge_dim=args.ks_dim, M=args.ks_train_buffer,
                          pool_mode=args.ks_pool_mode, n_heads=args.ks_pool_heads,
                          transformer_layers=args.ks_transformer_layers,
                          train_batch=args.ks_train_batch,
                          aux_loss_weight=args.ks_aux_loss_weight,
                          aux_mode=args.ks_aux_mode,
                          cl_weight=args.ks_cl_weight,
                          cl_temperature=args.ks_cl_temperature,
                          cl_subsample_frac=args.ks_cl_subsample_frac,
                          alp_mode=args.ks_alp_mode,
                          alp_ema=args.ks_alp_ema,
                          alp_p_theta=args.ks_alp_p_theta,
                          prospective_subsample=args.ks_prospective_subsample,
                          alp_deadzone=args.alp_deadzone,
                          balanced_training=args.balanced_training,
                          balance_alpha=args.ks_balance_alpha,
                          main_leave_self_out=args.ks_main_leave_self_out,
                          main_leave_self_out_mode=args.ks_main_leave_self_out_mode)
    estimator = build_curriculum(args.curriculum, goals, group_fn=group_fn, **cur_kwargs)

    # -- Workers -- ALWAYS seed=None (see module docstring) ------------------
    # obs_dtype: uint8 by default (minigrid_game/crafter_game's pixel
    # observations) unless the game module opts into something else —
    # fetch_game/env.py declares OBS_DTYPE = np.float32 (continuous robot
    # state, not pixels). See core/parallel_envs.py's _env_worker docstring.
    obs_dtype = getattr(env_module, "OBS_DTYPE", np.uint8)
    cur_goals = [estimator.sample_goal() for _ in range(args.n_envs)]
    vec = ParallelEnvs(args.n_envs, make_env_fn, cur_goals, seed=args.seed,
                       obs_dtype=obs_dtype)

    T, N = args.rollout_len, args.n_envs
    obs_shape = vec.obs.shape[1:]

    buf_obs      = np.zeros((T, N, *obs_shape), dtype=obs_dtype)
    buf_actions  = np.zeros((T, N), dtype=np.int64)
    buf_logprobs = np.zeros((T, N), dtype=np.float32)
    buf_values   = np.zeros((T, N), dtype=np.float32)
    buf_rewards  = np.zeros((T, N), dtype=np.float32)
    buf_dones    = np.zeros((T, N), dtype=bool)
    buf_goals    = np.empty((T, N), dtype=object)

    global_step, ep_count, total_success = 0, 0, 0
    eval_history = []
    recent_success = defaultdict(list)
    recent_success_window = defaultdict(lambda: deque(maxlen=100))
    # Env-agnostic train diagnostic: success since last console log, keyed
    # by group_fn(goal) (rooms on minigrid_game, achievement name on
    # crafter_game).
    log_group_hits = defaultdict(int)
    log_group_n    = defaultdict(int)
    # How many times each individual goal was actually drawn by
    # sample_goal() since the last console log — same per-goal granularity
    # as sampling_snapshots (get_sampling_probs()'s theoretical P(goal)),
    # so the two are directly comparable. Deliberately NOT grouped
    # (log_group_n above is per-group, per-room for minigrid_game — not
    # what we want here).
    recent_goal_draws = defaultdict(int)

    # -- Diagnostic histories: appended at log_every (not every rollout) ----
    group_alp_history     = []   # [(step, {group: alp})]
    entropy_log           = []   # [(step, {group: mean_entropy})]
    sampling_snapshots     = []   # [(step, {goal: prob})]
    effective_sampling_snapshots = []   # [(step, {goal: actual_draw_frac})]
    competence_snapshots   = []   # [(step, {goal: competence})] -- magellan/magellan_ks only
    success_snapshots      = []   # [(step, {goal: recent_success_rate})]
    loss_history            = []   # [(step, loss, aux_loss|None, cl_loss|None)] -- magellan/magellan_ks only
    ppo_loss_history        = []   # [(step, pg_loss, v_loss, entropy)] -- every curriculum

    # -- ks-specific logs: appended at checkpoint cadence (expensive) ------
    ks_history             = [] if args.curriculum == "magellan_ks" else None
    ks_mlp_history          = [] if args.curriculum == "magellan_ks" else None
    ks_raw_window_history   = [] if args.curriculum == "magellan_ks" else None
    ks_buffer_competence_history = [] if args.curriculum == "magellan_ks" else None

    ckpt_step_interval = max(1, args.steps // args.n_checkpoints)
    next_ckpt_idx  = global_step // ckpt_step_interval + 1
    next_ckpt_step = next_ckpt_idx * ckpt_step_interval
    next_ckpt_pct  = round(100 * next_ckpt_idx / args.n_checkpoints)
    next_log_ep    = args.log_every
    t_start = time.time()

    # eval's max_parallel defaults to 512 (bounds how many goals'
    # envs/scenes are held open at once during evaluation) — a game module
    # can override this down via an EVAL_MAX_PARALLEL attribute if its
    # goal count or per-env resource cost needs a tighter cap (an earlier
    # MuJoCo-based Fetch variant did, to avoid a "too many open files"
    # crash from holding hundreds of scenes open at once). None of the
    # currently registered games declare it, so this stays at the default.
    eval_max_parallel = getattr(env_module, "EVAL_MAX_PARALLEL", 512)

    # Persisted across the WHOLE run (not just one evaluate_all() call) so
    # every goal's Env is built at most ONCE, ever, then only reset()
    # between checkpoints — the standard way these environments are meant
    # to be used (see core/eval.py's module docstring). Matters most for
    # fetch_game (gymnasium_robotics/MuJoCo leaks file descriptors on
    # close(), so recreating envs every checkpoint would compound into a
    # "Too many open files" crash) but harmless and even mildly
    # beneficial (skips re-construction overhead) for minigrid_game/
    # crafter_game too, so this isn't game-gated.
    eval_env_cache = {}

    # Optional multi-process eval (see core/eval_parallel.py's module
    # docstring) -- addresses a real bottleneck where the serial path
    # above steps every active goal's Env one at a time on a single
    # core, which dominates checkpoint wall-time once the goal count is
    # large and the policy is still mostly failing (episodes run to
    # their full max length instead of terminating early). Streams
    # (goal, episode) work through a fixed pool of persistent workers,
    # exactly like this function's own training rollout loop below
    # reassigns a finished env slot's goal immediately (see
    # core/eval_parallel.py -- an earlier "pin each goal to one of a few
    # workers" design only divided the work by n_workers instead of by
    # the number of concurrently-active goals, and was still too slow).
    # Off by default (args.eval_workers=0) so every existing script
    # keeps its exact current behavior; opt in with --eval_workers N for
    # a large goal count (e.g. minigrid_game's larger room configs).
    eval_pool = None
    if args.eval_workers > 0:
        from core.eval_parallel import EvalStreamPool
        eval_pool = EvalStreamPool(args.eval_workers, make_env_fn)

    def do_eval_and_checkpoint(pct_label):
        agent.encoder.eval(); agent.goal_repr.eval(); agent.ac.eval()
        results = evaluate_all(agent, make_env_fn, eval_goals, args.eval_episodes, device,
                               max_parallel=eval_max_parallel, env_cache=eval_env_cache,
                               eval_pool=eval_pool)
        agent.encoder.train(); agent.goal_repr.train(); agent.ac.train()
        mean_sr = float(np.mean(list(results.values())))
        by_group = group_success_from_goal_results(results, group_fn)
        print(f"  [eval @ {pct_label}%] mean success = {mean_sr:.1%}")
        print(f"    by group: {_fmt_group_rates({g: d['mean'] for g, d in by_group.items()})}")
        eval_history.append(dict(pct=pct_label, step=global_step,
                                 episode=ep_count, results=results,
                                 results_by_group=by_group))
        save_checkpoint(agent, estimator, global_step, ep_count, total_success,
                        args, out_dir, pct_label, size, goal_dim)

        if args.curriculum == "magellan_ks":
            ks_history.append((pct_label, global_step, estimator.get_knowledge_state()))
            sd = {k: v.detach().cpu().clone()
                 for k, v in estimator.current_mlp.state_dict().items()}
            ks_mlp_history.append((pct_label, global_step, sd))
            ks_raw_window_history.append(
                (pct_label, global_step, estimator.get_raw_window_snapshot()))
            ks_buffer_competence_history.append(
                (pct_label, global_step, estimator.get_competence_by_ks_index()))

    # -- Main loop ------------------------------------------------------------
    while global_step < args.steps:
        for t in range(T):
            # Snapshot obs BEFORE step — must match the state that produced
            # action/logp/value. Storing vec.obs after step() was a bug
            # (off-by-one: buffer held s_{t+1} while logp was logπ(a|s_t)),
            # which broke PPO ratios and stalled learning near room-0 only.
            obs_N = vec.obs.copy()
            obs_t = torch.from_numpy(obs_N).to(device)
            with torch.no_grad():
                action, logp, entropy, value = agent.act(obs_t, cur_goals)
            actions_np = action.cpu().numpy()

            obs_next, rewards_N, terms_N, truncs_N = vec.step(actions_np)
            dones_N = terms_N | truncs_N

            buf_obs[t]      = obs_N
            buf_actions[t]  = actions_np
            buf_logprobs[t] = logp.cpu().numpy()
            buf_values[t]   = value.cpu().numpy()
            buf_rewards[t]  = rewards_N
            buf_dones[t]    = dones_N
            for i in range(N):
                buf_goals[t, i] = cur_goals[i]

            global_step += N

            for i in range(N):
                if dones_N[i]:
                    success = bool(terms_N[i])
                    total_success += int(success)
                    ep_count += 1
                    goal = cur_goals[i]
                    recent_success[goal].append(success)
                    recent_success_window[goal].append(success)
                    grp = _group_of(goal, group_fn)
                    log_group_hits[grp] += int(success)
                    log_group_n[grp]    += 1

                    estimator.update(goal, success)
                    estimator.anneal_epsilon(ep_count, args.epsilon_period, args.eps_min)

                    new_goal = estimator.sample_goal()
                    recent_goal_draws[new_goal] += 1
                    cur_goals[i] = new_goal
                    vec.reset_env(i, new_goal)

        # -- Bootstrap value --------------------------------------------------
        with torch.no_grad():
            obs_t = torch.from_numpy(vec.obs).to(device)
            last_values = agent.forward_value(obs_t, cur_goals).cpu().numpy()

        advantages, returns = gae(buf_rewards, buf_values, buf_dones,
                                  last_values, args.gamma, args.lam)

        buf = dict(obs=buf_obs, actions=buf_actions, log_probs=buf_logprobs, goals=buf_goals)
        stats = ppo_update(agent, opt, buf, advantages, returns, args, device)
        if scheduler is not None:
            scheduler.step()

        # -- Loss histories: EVERY rollout, not gated on log_every -----------
        # Unlike entropy_log below (a real extra forward pass,
        # deliberately throttled), these are just reading values already
        # computed this rollout (stats from ppo_update() above, and the
        # competence estimator's cached _last_loss/_last_aux_loss/
        # _last_cl_loss from its most recent _train_mlp() call) -- so
        # recording them every rollout instead of only every log_every
        # episodes is free (no extra compute), and gives much denser,
        # smoother loss curves for the report's plots.
        ppo_loss_history.append(
            (global_step, stats["pg_loss"], stats["v_loss"], stats["entropy"]))
        if hasattr(estimator, "get_last_loss"):
            loss = estimator.get_last_loss()
            if loss is not None:
                aux_loss = (estimator.get_last_aux_loss()
                           if hasattr(estimator, "get_last_aux_loss") else None)
                cl_loss = (estimator.get_last_cl_loss()
                          if hasattr(estimator, "get_last_cl_loss") else None)
                loss_history.append((global_step, loss, aux_loss, cl_loss))

        # -- Periodic eval + checkpoint (every ~1/n_checkpoints, e.g. 10%) ----
        if global_step >= next_ckpt_step:
            do_eval_and_checkpoint(next_ckpt_pct)
            next_ckpt_step += ckpt_step_interval
            next_ckpt_pct  += round(100 / args.n_checkpoints)

        # -- Console log + diagnostic snapshots (log_every episodes) ----------
        if ep_count >= next_log_ep:
            sps = global_step / max(time.time() - t_start, 1)
            pct = 100.0 * global_step / args.steps
            alp_str = ""
            galp = None
            if hasattr(estimator, "get_alp_all_groups"):
                galp = estimator.get_alp_all_groups()
                if galp:
                    alp_str = f"  alp_max={max(galp.values()):.4f}"
            print(f"  [{pct:5.1f}%] ep={ep_count}  eps={estimator.epsilon:.3f}  "
                 f"entropy={stats['entropy']:.3f}  "
                 f"pg_loss={stats['pg_loss']:.3f}  v_loss={stats['v_loss']:.3f}  "
                 f"{sps:.0f} sps{alp_str}")
            rates = {g: log_group_hits[g] / max(log_group_n[g], 1)
                     for g in log_group_n}
            print(f"    train by group: {_fmt_group_rates(rates, log_group_n, galp)}")
            log_group_hits.clear()
            log_group_n.clear()

            # Same cadence as console — enough for plots, avoids a full
            # extra forward pass (compute_group_entropy) every PPO update.
            entropy_log.append((global_step, compute_group_entropy(
                agent, buf_obs, buf_goals, group_fn, device)))
            sampling_snapshots.append((global_step, estimator.get_sampling_probs()))
            n_draws = sum(recent_goal_draws.values())
            effective_sampling_snapshots.append((global_step, {
                g: c / n_draws for g, c in recent_goal_draws.items()} if n_draws else {}))
            recent_goal_draws.clear()
            success_snapshots.append((global_step, {
                g: float(np.mean(v)) for g, v in recent_success_window.items() if v}))
            if galp is not None:
                group_alp_history.append((global_step, galp))
            if hasattr(estimator, "get_competence_all"):
                competence_snapshots.append(
                    (global_step, estimator.get_competence_all()))

            next_log_ep += args.log_every

    vec.close()

    if not eval_history or eval_history[-1]["pct"] < 100:
        do_eval_and_checkpoint(100)

    for env in eval_env_cache.values():
        env.close()
    if eval_pool is not None:
        eval_pool.close()

    stats_path = os.path.join(out_dir, "stats.pt")
    torch.save(dict(
        eval_history=eval_history, goals=goals, game=args.game,
        curriculum=args.curriculum, size=size, goal_dim=goal_dim,
        group_alp_history=group_alp_history, entropy_log=entropy_log,
        sampling_snapshots=sampling_snapshots,
        effective_sampling_snapshots=effective_sampling_snapshots,
        competence_snapshots=competence_snapshots,
        success_snapshots=success_snapshots, loss_history=loss_history,
        ppo_loss_history=ppo_loss_history,
    ), stats_path)

    if args.curriculum == "magellan_ks":
        for name, data in [("ks_history", ks_history),
                           ("ks_mlp_history", ks_mlp_history),
                           ("ks_raw_window_history", ks_raw_window_history)]:
            p = os.path.join(out_dir, f"{name}.pt")
            torch.save(dict(history=data, ks_dim=args.ks_dim,
                            ks_item_hidden=args.ks_item_hidden,
                            goal_dim=goal_dim), p)
            print(f"  -> {p}")

        # Separate file: needs goal_names/buffer_size on top of the usual
        # ks_dim/goal_dim, since each history entry is a (S, G) grid rather
        # than a single ks vector or state_dict.
        p = os.path.join(out_dir, "ks_buffer_competence_history.pt")
        torch.save(dict(history=ks_buffer_competence_history,
                        goal_names=list(estimator.valid_goals),
                        buffer_size=estimator.N, ks_dim=args.ks_dim,
                        goal_dim=goal_dim), p)
        print(f"  -> {p}")

    print(f"\n  Done - {ep_count} ep, {total_success} successes "
         f"({100 * total_success / max(ep_count, 1):.1f}%)")
    print(f"  Stats -> {stats_path}")
    print(f"  All outputs -> {out_dir}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _seed_type(value: str):
    """Allows --seed None on the CLI (argparse's plain type=int can't
    parse the string "None" — this makes it a real, valid option meaning
    "fully random, no reproducibility")."""
    if value.lower() == "none":
        return None
    return int(value)


def _bool_type(value: str) -> bool:
    """Parse CLI booleans reliably. argparse's type=bool is broken: bool("False")
    is True because any non-empty string is truthy."""
    v = value.lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected true/false, got {value!r}")


def _print_overrides(args, parser) -> None:
    """Print every CLI arg that differs from its argparse default — the
    actually-in-effect config, visible at a glance instead of having to
    read a long --game ... --foo ... line character by character. Uses
    parser._actions (private argparse API, but stable in practice and
    there's no public equivalent for this)."""
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    overrides = {k: v for k, v in vars(args).items()
                if k in defaults and v != defaults[k]}
    print(f"  Overrides (vs argparse defaults): {overrides}")


def parse_args():
    p = argparse.ArgumentParser()

    game_grp = p.add_argument_group("Game / environment")
    game_grp.add_argument("--game",   choices=list(GAME_REGISTRY.keys()), required=True)
    game_grp.add_argument("--size",   type=int, nargs="+", default=None,
                   help="minigrid_game: n_cols n_rows room_w room_h "
                        "(default 5 5 5 5). "
                        "crafter_game: world_w world_h (default 64 64).")
    game_grp.add_argument("--length", type=int, default=None,
                   help="Max episode steps. None -> each game's own default.")
    game_grp.add_argument("--shaping", type=_bool_type, nargs="?", const=True,
                   default=False,
                   help="Shaped reward instead of binary 0/1. minigrid_game: "
                        "speed-shaped native reward. crafter_game: tech-tree "
                        "unlock bonuses. Pass --shaping, or --shaping False "
                        "to force off.")

    curr_grp = p.add_argument_group("Curriculum")
    curr_grp.add_argument("--curriculum", choices=list(CURRICULUM_REGISTRY.keys()),
                   default="magellan_ks")
    curr_grp.add_argument("--window",         type=int,   default=100)
    curr_grp.add_argument("--epsilon_period", type=int,   default=320)
    curr_grp.add_argument("--eps_min",        type=float, default=0.2)
    curr_grp.add_argument("--alp_deadzone",   type=float, default=0.0,
                   help="magellan/magellan_ks: below this max-ALP threshold, "
                        "sample uniformly instead of proportionally. 0.0 "
                        "(default) disables this.")
    curr_grp.add_argument("--balanced_training", action="store_true",
                   help="magellan/magellan_ks: re-weight the competence-"
                        "MLP's own training data so every goal contributes "
                        "equally to the gradient, regardless of how often "
                        "the curriculum sampled it. Off by default.")
    curr_grp.add_argument("--ks_balance_alpha", type=float, default=0.0,
                   help="magellan_ks, with --balanced_training: per-goal "
                        "sampling weight exponent in [0,1]. 0 (default) = "
                        "full equalization. 1 = no correction.")

    curr_grp.add_argument("--curriculum_candidate_subsample", type=int, default=0,
                   help="If >0, sample_goal() scores only a random subset "
                        "of this many goals per episode instead of all "
                        "|G| (cheaper at large goal counts; epsilon-greedy "
                        "exploration still draws from the full set). 0 "
                        "(default) = score every goal.")
    curr_grp.add_argument("--magellan_buffer",      type=int, default=5000)
    curr_grp.add_argument("--magellan_snapshots",   type=int, default=10)
    curr_grp.add_argument("--magellan_train_every", type=int, default=32)

    curr_grp.add_argument("--ks_window",       type=int, default=30)
    curr_grp.add_argument("--ks_item_hidden",  type=int, default=32)
    curr_grp.add_argument("--ks_dim",          type=int, default=16)
    curr_grp.add_argument("--ks_pool_mode",
                   choices=["mean", "attention", "lstm", "transformer"],
                   default="mean",
                   help="magellan_ks: how the recent-outcomes window is "
                        "pooled into ks. 'mean' (default): unweighted "
                        "average. 'attention': learned weighting via one "
                        "shared query. 'lstm': sequential, the only mode "
                        "sensitive to recency/order. 'transformer': "
                        "self-attention between items, then a masked mean. "
                        "See core/curriculum.py's DeepSetKnowledgeEncoder "
                        "for details and caveats.")
    curr_grp.add_argument("--ks_pool_heads", type=int, default=1,
                   help="Independent attention heads for --ks_pool_mode="
                        "attention/transformer. Default 1.")
    curr_grp.add_argument("--ks_transformer_layers", type=int, default=2,
                   help="Stacked self-attention layers for "
                        "--ks_pool_mode=transformer. Default 2.")
    curr_grp.add_argument("--ks_main_leave_self_out", action="store_true",
                   help="magellan_ks: strip a training example's own goal "
                        "from its window before encoding ks for the main "
                        "loss, forcing genuine cross-goal inference. Off "
                        "by default.")
    curr_grp.add_argument("--ks_main_leave_self_out_mode", type=str,
                   default="strip", choices=["strip", "mask", "mask_embed"],
                   help="With --ks_main_leave_self_out: 'strip' (default) "
                        "deletes self-occurrences from the window. 'mask' "
                        "keeps the window's length/order and replaces the "
                        "outcome with a sentinel. 'mask_embed' does the "
                        "same via a dedicated learned embedding instead of "
                        "a raw sentinel.")
    curr_grp.add_argument("--ks_train_buffer", type=int, default=50_000)
    curr_grp.add_argument("--ks_train_batch",  type=int, default=256)
    curr_grp.add_argument("--ks_aux_loss_weight", type=float, default=0.0,
                   help="magellan_ks: weight of the auxiliary ks-recovery "
                        "loss. 0 (default) = off.")
    curr_grp.add_argument("--ks_aux_mode", choices=["window", "leave_self_out"],
                   default="window",
                   help="With --ks_aux_loss_weight>0: 'window' (default) "
                        "decodes against the main window. 'leave_self_out' "
                        "strips the example's own goal first, forcing "
                        "cross-goal inference.")
    curr_grp.add_argument("--ks_cl_weight", type=float, default=0.0,
                   help="magellan_ks: weight of the contrastive (NT-Xent) "
                        "loss on ks. 0 (default) = off.")
    curr_grp.add_argument("--ks_cl_temperature", type=float, default=0.2,
                   help="NT-Xent temperature (only used if --ks_cl_weight > 0)")
    curr_grp.add_argument("--ks_cl_subsample_frac", type=float, default=0.7,
                   help="Fraction of a window's items kept in each "
                        "augmented view for the contrastive loss "
                        "(only used if --ks_cl_weight > 0)")
    curr_grp.add_argument("--ks_alp_mode",
                   choices=["endpoints", "path", "to_now", "omni_ks",
                            "prospective_self", "prospective_self_signed",
                            "prospective_global"],
                   default="endpoints",
                   help="magellan_ks: how to compute ALP. 'endpoints' "
                        "(default) = |C(ks_now)-C(ks_old)|. 'path' = mean "
                        "consecutive |ΔC| along the ks buffer. 'to_now' = "
                        "mean distance from every buffered snapshot to "
                        "now. 'omni_ks' = OMNI-style double-EMA "
                        "reweighting (--ks_alp_p_theta). 'prospective_self'"
                        "/'prospective_global' simulate one step ahead "
                        "instead of comparing retrospective snapshots — "
                        "see core/curriculum.py's alp_mode docstring for "
                        "the full mechanism and caveats of each.")
    curr_grp.add_argument("--ks_alp_ema", type=float, default=0.0,
                   help="EMA smoothing along the ks trajectory before ALP "
                        "(0 = off).")
    curr_grp.add_argument("--ks_alp_p_theta", type=float, default=0.1)
    curr_grp.add_argument("--ks_prospective_subsample", type=int, default=50,
                    help="With --ks_alp_mode=prospective_global: random-"
                            "subsample size for the O(n_goals^2) global-shift "
                            "estimate (the full grid is too slow at large "
                            "goal counts).")

    ppo_grp = p.add_argument_group("PPO / optimization")
    ppo_grp.add_argument("--steps",           type=int,   default=50_000_000)
    ppo_grp.add_argument("--rollout_len",     type=int,   default=1024)
    ppo_grp.add_argument("--n_envs",          type=int,   default=32)
    ppo_grp.add_argument("--torch_threads",   type=int,   default=2)
    ppo_grp.add_argument("--num_minibatches", type=int,   default=4)
    ppo_grp.add_argument("--ppo_epochs",      type=int,   default=4)
    ppo_grp.add_argument("--lr",              type=float, default=3e-4)
    ppo_grp.add_argument("--no_lr_schedule",  action="store_true",
                   help="Keep lr constant at --lr instead of the default "
                        "linear decay to 10%% of --lr by the end of training.")
    ppo_grp.add_argument("--clip_eps",        type=float, default=0.2)
    ppo_grp.add_argument("--gamma",           type=float, default=0.99)
    ppo_grp.add_argument("--lam",             type=float, default=0.95)
    ppo_grp.add_argument("--entropy_coef",    type=float, default=0.01)
    ppo_grp.add_argument("--rnn_bptt_len",    type=int,   default=16,
                   help="core.train_rnn only: truncated-BPTT chunk length "
                        "for the recurrent PPO update.")

    agent_grp = p.add_argument_group("Agent / architecture (shared)")
    agent_grp.add_argument("--enc_dim",         type=int,   default=128)
    agent_grp.add_argument("--hidden_dim",      type=int,   default=512)

    crafter_grp = p.add_argument_group("Crafter only")
    crafter_grp.add_argument("--crafter_goal_repr",
                   choices=["onehot", "tech_tree", "llm"],
                   default="onehot",
                   help="Goal representation shared by policy and "
                        "curriculum. 'onehot' (default): trainable "
                        "embedding. 'tech_tree': fixed structured features. "
                        f"'llm': frozen Qwen embeddings at "
                        f"{crafter_env.DEFAULT_LLM_GOAL_EMBEDS}.")
    crafter_grp.add_argument("--crafter_goals", choices=["all", "crafting"], default="all",
                   help="Goal set for curriculum/eval. 'all' (default): "
                        "all 22 achievements. 'crafting': the 14 wood/"
                        "stone/iron/diamond tool achievements only. "
                        "Forced to 'crafting' automatically when "
                        "--crafter_disable_survival is set.")
    crafter_grp.add_argument("--crafter_separate_ac", action="store_true",
                   help="Independent actor/critic trunks instead of the "
                        "default shared trunk.")
    crafter_grp.add_argument("--crafter_disable_survival", action="store_true",
                   help="Disable hunger/thirst/zombie threats (world gen "
                        "unchanged, hazards self-remove). Forces "
                        "--crafter_goals=crafting, since survival-tied "
                        "achievements become unreachable.")

    log_grp = p.add_argument_group("Logging / eval")
    log_grp.add_argument("--n_checkpoints", type=int, default=10,
                   help="Eval+checkpoint points spread evenly (default 10 -> every 10%%)")
    log_grp.add_argument("--eval_episodes", type=int, default=20)
    log_grp.add_argument("--log_every",     type=int, default=20000)
    log_grp.add_argument("--eval_workers", type=int, default=0,
                   help="0 (default): serial checkpoint eval. >0: stream "
                        "eval through N persistent worker processes "
                        "(core/eval_parallel.py) — faster at large goal "
                        "counts, not bit-identical to the serial path.")
    log_grp.add_argument("--eval_goals_per_group", type=int, default=None,
                   help="Evaluate only this many goals per group (sampled "
                        "once, reused every checkpoint) instead of every "
                        "goal. Training still sees the full goal list.")
    log_grp.add_argument("--eval_max_goals", type=int, default=None,
                   help="Cap the total number of goals evaluated per "
                        "checkpoint (sampled once, reused every "
                        "checkpoint), regardless of grouping. Applied "
                        "after --eval_goals_per_group.")

    misc_grp = p.add_argument_group("Reproducibility / misc")
    misc_grp.add_argument("--seed",     type=_seed_type, default=42,
                   help="Int for reproducible runs, or 'None' for fully "
                        "random (no reproducibility guarantee).")
    misc_grp.add_argument("--device",   default=None)
    misc_grp.add_argument("--run_name", default="")
    misc_grp.add_argument("--exp_root", default="exp",
                   help="Base directory each run's output folder is "
                        "created under (default: 'exp').")

    args = p.parse_args()
    if args.crafter_disable_survival and args.crafter_goals != "crafting":
        print("  [note] --crafter_disable_survival forces "
              "--crafter_goals=crafting (survival-tied achievements are "
              "unreachable once hunger/thirst/zombies are disabled)")
        args.crafter_goals = "crafting"
    _print_overrides(args, p)
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train(parse_args())
