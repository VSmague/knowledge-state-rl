"""
core/parallel_envs.py — Generic multiprocessing vectorized envs.

Fully game-agnostic: works through a `make_env_fn(goal=..., seed=...)`
factory rather than game-specific constructor args (n_cols/n_rows/room_w/
... for minigrid_game, or anything crafter_game-specific), so it lives
here in core/ rather than inside any one game's own module.

`make_env_fn` just needs to be a picklable callable (multiprocessing uses
"spawn", so it gets sent to each worker subprocess) returning something
matching the shared Env contract (minigrid_game/env.py and
crafter_game/env.py both already do):
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    env.set_goal(goal)
    env.close()

Typical construction, from core/train.py, for either game:
    import functools
    make_env_fn = functools.partial(game.env.Env, size=size, length=length)
    vec = ParallelEnvs(n_envs, make_env_fn, initial_goals, seed)

`functools.partial` of a module-level class (Env) with simple picklable
kwargs (size/length — tuples/ints) pickles fine for spawn, so this needs
no special-casing per game.
"""

import multiprocessing as mp

import numpy as np


def _env_worker(conn, make_env_fn, initial_goal, seed, obs_dtype=np.uint8):
    """
    Runs in a subprocess. Accepts commands from the main process via conn:
      ("step",   action)  -> obs, reward, term, trunc
      ("reset",  goal)    -> obs        (change goal, then reset)
      ("close",  None)    -> exits

    obs_dtype defaults to uint8 (minigrid_game/crafter_game's pixel
    observations) — fetch_game/env.py declares OBS_DTYPE = np.float32 at
    module level (its observations are continuous robot/object state, not
    pixels; casting to uint8 would destroy them) and core/train.py reads
    that per-game, via getattr(env_module, "OBS_DTYPE", np.uint8), to pass
    here. minigrid_game/crafter_game don't declare OBS_DTYPE, so they're
    completely unaffected — this parameter exists so ONE game can opt
    into a different dtype without changing behavior for the others.

    `seed` seeds a LOCAL, per-worker RNG (np.random.default_rng(seed)) —
    used to draw a FRESH episode seed on every single reset() call (the
    initial one and every subsequent "reset" command). This gives full
    reproducibility: the same base seed always produces the exact same
    SEQUENCE of episode seeds for a given worker, so two full re-runs with
    the same --seed replay byte-for-byte identical training environments
    (same door colors, same key positions, in the same order) — while
    still varying genuinely from episode to episode WITHIN a single run
    (each draw is a new value, not the same one reused), which is what
    lets the agent generalize across key positions/colors rather than
    memorizing one fixed layout.

    Relying on env.py's own internal seed=None auto-random fallback
    (np.random.randint on the possibly-unseeded global numpy state)
    wouldn't work for this: that fallback isn't reproducible across
    process boundaries (a fresh subprocess doesn't inherit the main
    process's seeded numpy state), so it can't give the full-
    reproducibility guarantee described above on its own — hence the
    explicit local RNG here.
    """
    local_rng = np.random.default_rng(seed)

    def _next_seed() -> int:
        return int(local_rng.integers(0, 2**31 - 1))

    env = make_env_fn(goal=initial_goal)
    obs, _ = env.reset(seed=_next_seed())
    conn.send(np.array(obs, dtype=obs_dtype))   # send initial obs

    while True:
        cmd, data = conn.recv()

        if cmd == "step":
            obs, reward, term, trunc, _ = env.step(data)
            conn.send((np.array(obs, dtype=obs_dtype),
                      float(reward), bool(term), bool(trunc)))

        elif cmd == "reset":
            new_goal = data
            env.set_goal(new_goal)
            obs, _ = env.reset(seed=_next_seed())
            conn.send(np.array(obs, dtype=obs_dtype))

        elif cmd == "close":
            env.close()
            conn.close()
            return


class ParallelEnvs:
    """Manages N env workers. Vectorised step() + per-env reset(goal)."""

    def __init__(self, n_envs: int, make_env_fn, initial_goals: list, seed: int = None,
                obs_dtype=np.uint8):
        self.n_envs = n_envs
        self.conns  = []
        self.procs  = []

        for i in range(n_envs):
            worker_seed = None if seed is None else seed + i
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_env_worker,
                args=(child_conn, make_env_fn, initial_goals[i], worker_seed, obs_dtype),
                daemon=True)
            p.start()
            child_conn.close()   # close child end in parent
            self.conns.append(parent_conn)
            self.procs.append(p)

        self._last_obs = np.stack([c.recv() for c in self.conns])

    @property
    def obs(self) -> np.ndarray:
        """Current observations (N, ...obs_shape)."""
        return self._last_obs

    def step(self, actions: np.ndarray):
        """
        Send actions to all envs simultaneously, collect results.
        Returns obs (N,...), rewards (N,), terms (N,), truncs (N,).
        """
        for i, conn in enumerate(self.conns):
            conn.send(("step", int(actions[i])))

        obs_list, rewards, terms, truncs = [], [], [], []
        for conn in self.conns:
            img, r, term, trunc = conn.recv()
            obs_list.append(img)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)

        self._last_obs = np.stack(obs_list)
        return (self._last_obs,
                np.array(rewards, dtype=np.float32),
                np.array(terms,   dtype=bool),
                np.array(truncs,  dtype=bool))

    def reset_env(self, i: int, goal) -> None:
        """Reset env i with a specific goal. Blocks until new obs received."""
        self.conns[i].send(("reset", goal))
        self._last_obs[i] = self.conns[i].recv()

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)