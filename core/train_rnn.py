"""
core/train_rnn.py — Recurrent (LSTM) variant of core/train.py's PPO loop,
crafter_game-only (minigrid_game/fetch_game have no RecurrentAgent). A
SEPARATE file rather than a flag on core/train.py, because recurrence
isn't just an architecture choice: hidden state has to be threaded
through every rollout step and reset per-env on episode boundaries,
which needs a different rollout loop and a different (sequence-
respecting) PPO minibatching scheme — not something that fits cleanly
as an if/else inside the existing, tested core/train.py without risking
its behavior for the non-recurrent path.

Reuses core/train.py's game-agnostic helpers (build_goal_vectors,
build_group_fn, formatting, gae, parse_args, DEFAULT_SIZE)
via import instead of duplicating them — only the genuinely different
parts (the rollout loop, PPO update, checkpointing, eval) are
reimplemented here, against crafter_game.architecture.RecurrentAgent's
extended contract (act/forward_value/evaluate_actions all take AND
return a `hidden` = (h, c) pair, unlike Agent's plain 3-method contract).

Usage:
    python -m core.train_rnn --game crafter_game --curriculum magellan_ks --steps 5_000_000

Verified with a real short run (real torch, real gradients end-to-end
through ppo_update_recurrent's backward()/optimizer.step()) — still only
a smoke test at small scale, not evidence about sample efficiency or
learning quality at a real training budget.
"""

import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch

from core.parallel_envs import ParallelEnvs
from core.curriculum import build_curriculum
from core.eval import group_success_from_goal_results
from core.train import (
    DEFAULT_SIZE, build_goal_vectors, build_group_fn,
    _group_of, _fmt_group_rates, gae,
    parse_args,
)

import crafter_game.env as crafter_env
import crafter_game.architecture as crafter_arch


# ---------------------------------------------------------------------------
# Checkpointing — own version, not core.train.save_checkpoint: RecurrentAgent
# has an extra `lstm` submodule that function doesn't know about.
# ---------------------------------------------------------------------------

def save_checkpoint_recurrent(agent, estimator, global_step, ep_count,
                              total_success, args, out_dir, pct_label,
                              size, goal_dim):
    ckpts_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpts_dir, exist_ok=True)
    path = os.path.join(ckpts_dir, f"ckpt_pct{pct_label:03d}.pt")
    payload = dict(
        encoder=agent.encoder.state_dict(),
        goal_repr=agent.goal_repr.state_dict(),
        lstm=agent.lstm.state_dict(),
        ac=agent.ac.state_dict(),
        hidden_size=agent.hidden_size,
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
# Eval — own version, not core.eval.evaluate_all: hidden state resets fresh
# at the start of each episode-round (all envs reset together there, so
# init_hidden(n) once per round is enough — no per-env mid-round reset
# logic needed, unlike training).
# ---------------------------------------------------------------------------

def _evaluate_goals_chunk_recurrent(agent, make_env_fn, goals: list,
                                    n_episodes: int, device: str,
                                    seed_offset: int) -> np.ndarray:
    n = len(goals)
    envs = [make_env_fn(goal=g) for g in goals]
    success_counts = np.zeros(n, dtype=np.int32)

    for ep in range(n_episodes):
        obs = []
        for env in envs:
            o, _ = env.reset(seed=seed_offset + ep)
            obs.append(o)

        done       = np.zeros(n, dtype=bool)
        term_flags = np.zeros(n, dtype=bool)
        h_full, c_full = agent.init_hidden(n)

        while not done.all():
            active = np.where(~done)[0]
            obs_batch = torch.from_numpy(np.stack([obs[i] for i in active])).to(device)
            goal_batch = [goals[i] for i in active]
            active_t = torch.as_tensor(active, dtype=torch.long, device=device)
            hidden = (h_full[active_t], c_full[active_t])

            with torch.no_grad():
                action, _, _, _, (h_new, c_new) = agent.act(obs_batch, goal_batch, hidden)
            actions_np = action.cpu().numpy()
            h_full = h_full.clone(); c_full = c_full.clone()
            h_full[active_t] = h_new
            c_full[active_t] = c_new

            for k, i in enumerate(active):
                o, _, terminated, truncated, _ = envs[i].step(int(actions_np[k]))
                obs[i] = o
                if terminated or truncated:
                    done[i]       = True
                    term_flags[i] = terminated

        success_counts += term_flags.astype(np.int32)

    for env in envs:
        env.close()
    return success_counts


def evaluate_all_recurrent(agent, make_env_fn, goals: list, n_episodes: int,
                           device: str, seed_offset: int = 1000,
                           max_parallel: int = 512) -> dict:
    was_training = agent.encoder.training
    agent.encoder.eval(); agent.goal_repr.eval(); agent.lstm.eval(); agent.ac.eval()

    n_goals = len(goals)
    success_counts = np.zeros(n_goals, dtype=np.int32)
    for start in range(0, n_goals, max_parallel):
        chunk = goals[start:start + max_parallel]
        success_counts[start:start + len(chunk)] = _evaluate_goals_chunk_recurrent(
            agent, make_env_fn, chunk, n_episodes, device, seed_offset)

    if was_training:
        agent.encoder.train(); agent.goal_repr.train(); agent.lstm.train(); agent.ac.train()
    return {g: float(success_counts[i]) / n_episodes for i, g in enumerate(goals)}


# ---------------------------------------------------------------------------
# PPO update — sequence-respecting (truncated BPTT), minibatches split by
# ENV not by flat timestep (unlike core.train.ppo_update's
# np.random.permutation(T*N) — that shuffle is exactly what recurrence
# can't tolerate: an LSTM needs its inputs in temporal order with a
# correctly-propagated hidden state, not a random bag of individual steps).
# ---------------------------------------------------------------------------

def ppo_update_recurrent(agent, opt, buf, advantages, returns, args, device):
    T, N = buf["actions"].shape
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t   = torch.from_numpy(buf["obs"]).to(device)
    act_t   = torch.from_numpy(buf["actions"]).long().to(device)
    logp_t  = torch.from_numpy(buf["log_probs"]).to(device)
    adv_t   = torch.from_numpy(adv.astype(np.float32)).to(device)
    ret_t   = torch.from_numpy(returns.astype(np.float32)).to(device)
    dones_t = torch.from_numpy(buf["dones"]).to(device)
    goals   = buf["goals"]  # (T, N) object array — indexed directly, not a tensor

    bptt = max(1, args.rnn_bptt_len)
    n_minibatches = max(1, min(args.num_minibatches, N))
    mb_size = max(1, N // n_minibatches)

    stats = dict(pg_loss=0.0, v_loss=0.0, entropy=0.0)
    n_upd = 0

    for _ in range(args.ppo_epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, mb_size):
            mb_envs = perm[start:start + mb_size]
            mb_envs_t = torch.as_tensor(mb_envs, dtype=torch.long, device=device)
            B = len(mb_envs)
            h, c = agent.init_hidden(B)

            opt.zero_grad()
            chunk_pg = chunk_v = chunk_ent = 0.0
            n_chunks = 0

            # encode_sequence is called PER BPTT CHUNK, not once for the
            # whole T-length minibatch — calling it once for all of T and
            # then backward()-ing per chunk crashed on real hardware
            # ("Trying to backward through the graph a second time"):
            # every x_seq[t] slice shares the SAME upstream encoder/
            # goal_repr graph nodes (one batched forward over all of T),
            # so the first chunk's backward() frees graph nodes the next
            # chunk's backward() also needs. Encoding per-chunk instead
            # gives each chunk its own independent graph, freed cleanly
            # after its own backward() — still batches bptt frames'
            # worth of CNN calls into one instead of doing them one
            # timestep at a time, just not all of T at once.
            for chunk_start in range(0, T, bptt):
                chunk_end = min(chunk_start + bptt, T)
                obs_chunk  = obs_t[chunk_start:chunk_end, mb_envs_t]
                goal_chunk = goals[chunk_start:chunk_end, mb_envs]
                x_chunk = agent.encode_sequence(obs_chunk, goal_chunk)  # (chunk_len, B, feat_dim)

                pg_buf, v_buf, ent_buf = [], [], []
                for tt in range(chunk_end - chunk_start):
                    t = chunk_start + tt
                    act_mb = act_t[t, mb_envs_t]

                    new_logp, entropy, value, (h, c) = agent.evaluate_actions_from_features(
                        x_chunk[tt], act_mb, (h, c))

                    ratio  = (new_logp - logp_t[t, mb_envs_t]).exp()
                    mb_adv = adv_t[t, mb_envs_t]
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                    pg_buf.append(torch.max(pg1, pg2))
                    v_buf.append((value - ret_t[t, mb_envs_t]) ** 2)
                    ent_buf.append(entropy)

                    # Reset hidden state for envs whose episode ended
                    # exactly at this step — the NEXT step (if any, in
                    # this chunk or the next) starts a fresh episode for
                    # that env slot.
                    done_mask = dones_t[t, mb_envs_t]
                    if done_mask.any():
                        h = h.clone(); c = c.clone()
                        h[done_mask] = 0.0
                        c[done_mask] = 0.0

                pg_loss_chunk = torch.cat([x.reshape(-1) for x in pg_buf]).mean()
                v_loss_chunk  = torch.cat([x.reshape(-1) for x in v_buf]).mean()
                ent_chunk     = torch.cat([x.reshape(-1) for x in ent_buf]).mean()
                loss_chunk = pg_loss_chunk + 0.5 * v_loss_chunk - args.entropy_coef * ent_chunk
                loss_chunk.backward()
                # Truncated BPTT: carry hidden VALUES forward into the
                # next chunk, but cut the gradient graph here so each
                # chunk's backward() only covers its own ~bptt steps.
                h = h.detach(); c = c.detach()

                chunk_pg += pg_loss_chunk.item()
                chunk_v  += v_loss_chunk.item()
                chunk_ent += ent_chunk.item()
                n_chunks += 1

            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            opt.step()

            stats["pg_loss"] += chunk_pg / max(n_chunks, 1)
            stats["v_loss"]  += chunk_v / max(n_chunks, 1)
            stats["entropy"] += chunk_ent / max(n_chunks, 1)
            n_upd += 1

    for k in stats:
        stats[k] /= max(n_upd, 1)
    return stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_recurrent(args):
    import random as _random
    import functools

    assert args.game == "crafter_game", (
        "core.train_rnn only supports --game crafter_game (RecurrentAgent "
        "has no minigrid_game/fetch_game equivalent) — use core.train for "
        "those.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, args.torch_threads))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)

    env_module  = crafter_env
    arch_module = crafter_arch

    size   = tuple(args.size) if args.size else DEFAULT_SIZE[args.game]
    length = args.length

    env_kwargs = dict(size=size, length=length, shaping=args.shaping)
    if hasattr(env_module, "extra_env_kwargs"):
        env_kwargs.update(env_module.extra_env_kwargs(args))
    goals = env_module.goals(size, subset=args.crafter_goals)
    make_env_fn = functools.partial(env_module.Env, **env_kwargs)

    frozen_embeddings_path = None
    policy_embeddings_matrix = None
    curriculum_embeddings_matrix = None
    if hasattr(env_module, "build_train_goal_repr"):
        (policy_embeddings_matrix, frozen_embeddings_path,
         curriculum_embeddings_matrix) = env_module.build_train_goal_repr(args, goals)

    goal_vectors, goal_dim = build_goal_vectors(
        args.game, env_module, goals, size,
        embeddings_path=None, embeddings_matrix=curriculum_embeddings_matrix)
    group_fn = build_group_fn(args.game, env_module, size)

    run_suffix = f"_{args.run_name}" if args.run_name else ""
    out_dir = os.path.join(
        args.exp_root, f"{args.game}_{args.curriculum}_rnn_{args.steps}steps{run_suffix}")
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
    print(f"  {args.game.upper()} / {args.curriculum.upper()} / RNN  "
         f"-  size={size}  -  {len(goals)} goals")
    print(f"  Steps: {args.steps:,}  device={device}  seed={args.seed}  "
         f"bptt={args.rnn_bptt_len}")
    reward_desc = (env_module.shaping_description()
                   if args.shaping and hasattr(env_module, "shaping_description")
                   else ("shaped" if args.shaping else "binary"))
    print(f"  reward={reward_desc}")
    print(f"  Out dir: {out_dir}")
    print("=" * 62 + "\n")

    # -- Agent ----------------------------------------------------------------
    # separate_ac reuses --crafter_separate_ac (same conceptual choice as
    # the feedforward Agent) — default False (shared trunk, matching
    # GoalConditionedAC's own default); pass --crafter_separate_ac to
    # reproduce OMNI's actual choice instead. See
    # RecurrentGoalConditionedAC's docstring for why this is a flag, not
    # a fixed choice, for the recurrent case.
    agent = arch_module.build_recurrent_agent(
        size, device=device, enc_dim=args.enc_dim, hidden_size=args.hidden_dim,
        frozen_embeddings_path=frozen_embeddings_path,
        separate_ac=args.crafter_separate_ac)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    total_updates = max(1, args.steps // (args.rollout_len * args.n_envs))
    scheduler = None
    if not args.no_lr_schedule:
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_updates)

    # -- Curriculum -------------------------------------------------------------
    cur_kwargs = dict(eps_min=args.eps_min, device=device)
    if args.curriculum == "online":
        cur_kwargs.update(M=args.window)
    if args.curriculum == "magellan":
        cur_kwargs.update(goal_vectors=goal_vectors, goal_dim=goal_dim,
                          M=args.magellan_buffer, N=args.magellan_snapshots,
                          train_every=args.magellan_train_every)
    if args.curriculum == "magellan_ks":
        cur_kwargs.update(goal_vectors=goal_vectors, goal_dim=goal_dim,
                          N=args.magellan_snapshots, train_every=args.magellan_train_every,
                          K=args.ks_window, item_hidden=args.ks_item_hidden,
                          knowledge_dim=args.ks_dim, M=args.ks_train_buffer,
                          train_batch=args.ks_train_batch,
                          aux_loss_weight=args.ks_aux_loss_weight,
                          cl_weight=args.ks_cl_weight,
                          cl_temperature=args.ks_cl_temperature,
                          cl_subsample_frac=args.ks_cl_subsample_frac,
                          alp_mode=args.ks_alp_mode,
                          alp_ema=args.ks_alp_ema,
                          alp_p_theta=args.ks_alp_p_theta)
    estimator = build_curriculum(args.curriculum, goals, group_fn=group_fn, **cur_kwargs)

    # -- Workers ----------------------------------------------------------------
    cur_goals = [estimator.sample_goal() for _ in range(args.n_envs)]
    vec = ParallelEnvs(args.n_envs, make_env_fn, cur_goals, seed=args.seed)

    T, N = args.rollout_len, args.n_envs
    obs_shape = vec.obs.shape[1:]

    buf_obs      = np.zeros((T, N, *obs_shape), dtype=np.uint8)
    buf_actions  = np.zeros((T, N), dtype=np.int64)
    buf_logprobs = np.zeros((T, N), dtype=np.float32)
    buf_values   = np.zeros((T, N), dtype=np.float32)
    buf_rewards  = np.zeros((T, N), dtype=np.float32)
    buf_dones    = np.zeros((T, N), dtype=bool)
    buf_goals    = np.empty((T, N), dtype=object)

    # Persistent across rollout batches (unlike the feedforward loop, which
    # has no state to carry) — only reset per-env when THAT env's episode
    # ends, not at rollout_len boundaries. An episode can span multiple
    # rollout batches.
    hidden = agent.init_hidden(N)

    global_step, ep_count, total_success = 0, 0, 0
    eval_history = []
    recent_success = defaultdict(list)
    recent_success_window = defaultdict(lambda: deque(maxlen=100))
    log_group_hits = defaultdict(int)
    log_group_n    = defaultdict(int)
    recent_goal_draws = defaultdict(int)

    group_alp_history     = []
    entropy_log           = []
    sampling_snapshots     = []
    effective_sampling_snapshots = []
    competence_snapshots   = []
    success_snapshots      = []
    loss_history            = []

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

    def do_eval_and_checkpoint(pct_label):
        agent.encoder.eval(); agent.goal_repr.eval(); agent.lstm.eval(); agent.ac.eval()
        results = evaluate_all_recurrent(agent, make_env_fn, goals, args.eval_episodes, device)
        agent.encoder.train(); agent.goal_repr.train(); agent.lstm.train(); agent.ac.train()
        mean_sr = float(np.mean(list(results.values())))
        by_group = group_success_from_goal_results(results, group_fn)
        print(f"  [eval @ {pct_label}%] mean success = {mean_sr:.1%}")
        print(f"    by group: {_fmt_group_rates({g: d['mean'] for g, d in by_group.items()})}")
        eval_history.append(dict(pct=pct_label, step=global_step,
                                 episode=ep_count, results=results,
                                 results_by_group=by_group))
        save_checkpoint_recurrent(agent, estimator, global_step, ep_count, total_success,
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
            obs_N = vec.obs.copy()
            obs_t = torch.from_numpy(obs_N).to(device)
            with torch.no_grad():
                action, logp, entropy, value, hidden = agent.act(obs_t, cur_goals, hidden)
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

            reset_idx = []
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
                    reset_idx.append(i)

            if reset_idx:
                h, c = hidden
                h = h.clone(); c = c.clone()
                idx_t = torch.as_tensor(reset_idx, dtype=torch.long, device=device)
                h[idx_t] = 0.0
                c[idx_t] = 0.0
                hidden = (h.detach(), c.detach())

        # -- Bootstrap value (also advances `hidden` onto s_T, ready for the
        # next rollout batch's first act() call on that same observation) --
        with torch.no_grad():
            obs_t = torch.from_numpy(vec.obs).to(device)
            last_values, hidden = agent.forward_value(obs_t, cur_goals, hidden)
            last_values = last_values.cpu().numpy()
            hidden = (hidden[0].detach(), hidden[1].detach())

        advantages, returns = gae(buf_rewards, buf_values, buf_dones,
                                  last_values, args.gamma, args.lam)

        buf = dict(obs=buf_obs, actions=buf_actions, log_probs=buf_logprobs,
                  goals=buf_goals, dones=buf_dones)
        stats = ppo_update_recurrent(agent, opt, buf, advantages, returns, args, device)
        if scheduler is not None:
            scheduler.step()

        if global_step >= next_ckpt_step:
            do_eval_and_checkpoint(next_ckpt_pct)
            next_ckpt_step += ckpt_step_interval
            next_ckpt_pct  += round(100 / args.n_checkpoints)

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
            if hasattr(estimator, "get_last_loss"):
                loss = estimator.get_last_loss()
                if loss is not None:
                    loss_history.append((global_step, loss))

            next_log_ep += args.log_every

    vec.close()

    if not eval_history or eval_history[-1]["pct"] < 100:
        do_eval_and_checkpoint(100)

    stats_path = os.path.join(out_dir, "stats.pt")
    torch.save(dict(
        eval_history=eval_history, goals=goals, game=args.game,
        curriculum=args.curriculum, size=size, goal_dim=goal_dim,
        group_alp_history=group_alp_history, entropy_log=entropy_log,
        sampling_snapshots=sampling_snapshots,
        effective_sampling_snapshots=effective_sampling_snapshots,
        competence_snapshots=competence_snapshots,
        success_snapshots=success_snapshots, loss_history=loss_history,
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


def main():
    args = parse_args()
    train_recurrent(args)


if __name__ == "__main__":
    main()
