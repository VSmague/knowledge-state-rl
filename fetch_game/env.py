"""
fetch_game/env.py — goal-conditioned wrapper around FetchPickAndPlace-v4.
A goal is a single continuous 6-vector (x0, y0, z0, xg, yg, zg): x0,y0,z0
is the object's fixed starting position, xg,yg,zg is the target it must
end up at. The agent has to do the whole chain (reach, grasp, lift if
needed, carry, place) in one shot — there is no task/level structure or
intermediate sub-goal telling it which stage it's on; goal difficulty is
left entirely for the curriculum to discover on its own.

=== The goal pool ===

N is a runtime argument (--size N, DEFAULT_N if omitted). The N (start,
target) pairs are drawn ONCE from a FIXED, hardcoded seed
(_GOAL_POOL_SEED), independent of the training run's own --seed, so
every curriculum/seed in a given experiment trains on the EXACT SAME N
goals. The pool is generated incrementally and cached at module level
(_goal_pool, extended lazily up to whatever N has been requested so
far), so goal i is the SAME point regardless of which N was asked for
first — asking for N=50 after N=30 extends the same list rather than
redrawing everything, keeping N-ablation runs nested/comparable.

Both (x0,y0) and (xg,yg) are drawn uniformly across the full table
(_sample_table_xy, inset by TABLE_MARGIN). z0 is not a free choice — the
object always starts resting on the table, so z0 is just the table
height offset (included in the goal vector for a complete 6-dim
encoding, even though it carries no information on its own). zg is drawn
uniformly from [table_z, table_z+LIFT_H_MAX] independently per goal,
with no forced on-table fraction — a goal can still land close enough to
the surface to be push-solvable, but as the tail of a continuous
distribution rather than a fixed half of the pool.

=== Success check ===

A single fixed threshold (SUCCESS_THRESH, default 0.05m): success is the
object's center landing within SUCCESS_THRESH of the target, full stop —
no "was it lifted" check, no displacement/lift guard, no hold-duration
requirement. A pure push that never lifts the object is just as
legitimate a solution as a genuine pick-and-place, and if a (start,
target) pair happens to already satisfy the goal on reset, that's real
signal that the goal is trivially easy, not a bug to guard against.

=== goal_vector ===

Purely continuous: [x0,y0,z0,xg,yg,zg], each linearly normalized to
roughly [-1,1] (x/y via TABLE_X/TABLE_Y, z via [table_z,
table_z+LIFT_H_MAX]) — GOAL_DIM=6. No one-hot, no task/level structure.

=== Feasibility ===

Verified with a scripted full-chain controller (reach, close on the
object, lift to a cruise height, carry to the target xy, descend)
against a real sample of the goal pool on real MuJoCo, before any
training run.

=== Shaping ===

No shaping mechanism exists for this game — Env.__init__ accepts
`shaping` only for GAME_REGISTRY signature parity with minigrid_game/
crafter_game, and it has no effect regardless of its value.
"""

import os
import resource
import time

import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _hard > _soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ValueError, OSError):
    pass


def _make_env_with_retries(env_id: str, attempts: int = 10, backoff: float = 1.5,
                            **make_kwargs):
    """Retries gym.make with exponential backoff + jitter — MuJoCo scene
    construction can fail transiently under concurrent workers each
    opening file descriptors."""
    import random
    last_err = None
    for attempt in range(attempts):
        try:
            return gym.make(env_id, **make_kwargs)
        except (ValueError, OSError) as e:
            last_err = e
            try:
                n_fds = len(os.listdir("/proc/self/fd"))
            except OSError:
                n_fds = "?"
            print(f"[fetch_game/env.py] gym.make({env_id!r}) failed on attempt "
                  f"{attempt + 1}/{attempts} (open fds: {n_fds}): {e}", flush=True)
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1) + random.uniform(0, 1.0))
    raise last_err


_ENV_ID = "FetchPickAndPlace-v4"
_MAX_EPISODE_STEPS = 200

BINS = 3
_LEVELS = np.linspace(-1.0, 1.0, BINS) * 0.5   # = [-0.5, 0, 0.5], real step 0.025m
_GRIPPER_LEVELS = np.array([-1.0, 1.0], dtype=np.float32)  # closed, open

_ACTION_TABLE = np.array(
    [[dx, dy, dz, g]
     for dx in _LEVELS
     for dy in _LEVELS
     for dz in _LEVELS
     for g in _GRIPPER_LEVELS],
    dtype=np.float32,
)
N_ACTIONS = len(_ACTION_TABLE)  # 54

OBS_DTYPE = np.float32
OBS_DIM = 25 + 3  # native "observation" (25) + desired_goal (3)

# Table geometry, verified against the real MuJoCo model for reachability.
TABLE_X = (1.05, 1.55)
TABLE_Y = (0.40, 1.10)
TABLE_MARGIN = 0.03
_TABLE_Z = 0.4247  # height_offset, used only for goal_vector's z0/zg
                    # normalization; ACTUAL placement at reset time reads
                    # the live u.height_offset, not this hardcoded copy.
_LIFT_H_MAX = 0.30  # verified-safe ceiling for how high zg can sit above the table

SUCCESS_THRESH = float(os.environ.get("FETCH_SUCCESS_THRESH", "0.05"))

DEFAULT_N = 30
_GOAL_POOL_SEED = 20260914
_goal_pool_rng = np.random.RandomState(_GOAL_POOL_SEED)
_goal_pool: list = []  # lazily extended, see module docstring


def _sample_table_xy(np_random) -> np.ndarray:
    """Uniform random (x,y) across the full table, inset by TABLE_MARGIN.
    Works with both a gymnasium np_random Generator (real env resets)
    and a plain np.random.RandomState (the goal pool's own fixed-seed
    RNG below) — both expose a compatible .uniform(low, high) call."""
    x = np_random.uniform(TABLE_X[0] + TABLE_MARGIN, TABLE_X[1] - TABLE_MARGIN)
    y = np_random.uniform(TABLE_Y[0] + TABLE_MARGIN, TABLE_Y[1] - TABLE_MARGIN)
    return np.array([x, y], dtype=np.float32)


def _ensure_goal_pool(n: int) -> None:
    """Extend the module-level _goal_pool (a list of (x0,y0,z0,xg,yg,zg)
    tuples) up to length n, drawing new points from the fixed-seed RNG
    only for the newly-needed indices — see module docstring's "nested
    pools" reasoning."""
    while len(_goal_pool) < n:
        x0, y0 = _sample_table_xy(_goal_pool_rng)
        xg, yg = _sample_table_xy(_goal_pool_rng)
        zg = _TABLE_Z + _goal_pool_rng.uniform(0.0, _LIFT_H_MAX)
        _goal_pool.append((float(x0), float(y0), _TABLE_Z,
                           float(xg), float(yg), float(zg)))


def goals(size: tuple = None, subset: str = "all") -> list:
    """N goal ids (plain ints 0..N-1) — N = size[0] if given, else
    DEFAULT_N. subset accepted-but-unused, interface parity with
    core/train.py's call convention."""
    n = int(size[0]) if size else DEFAULT_N
    _ensure_goal_pool(n)
    return list(range(n))


def goal_point(goal_idx: int) -> tuple:
    """The real (x0,y0,z0,xg,yg,zg) tuple for a goal id — ensures the
    pool covers it first (safe even if called with an index beyond what
    goals() last returned)."""
    _ensure_goal_pool(goal_idx + 1)
    return _goal_pool[goal_idx]


def build_goal_vectors(goals: list, size: tuple = None, embeddings_path: str = None,
                       embeddings_matrix=None):
    vectors = {g: goal_vector(g) for g in goals}
    return vectors, GOAL_DIM


def goal_vector(goal_idx: int) -> np.ndarray:
    """[x0,y0,z0,xg,yg,zg], each linearly mapped to roughly [-1,1] —
    x/y via TABLE_X/TABLE_Y, z via [table_z, table_z+_LIFT_H_MAX].
    Purely continuous, no one-hot / task structure at all (see module
    docstring)."""
    x0, y0, z0, xg, yg, zg = goal_point(goal_idx)

    def norm_x(v): return -1.0 + 2.0 * (v - TABLE_X[0]) / (TABLE_X[1] - TABLE_X[0])
    def norm_y(v): return -1.0 + 2.0 * (v - TABLE_Y[0]) / (TABLE_Y[1] - TABLE_Y[0])
    def norm_z(v): return -1.0 + 2.0 * (v - _TABLE_Z) / _LIFT_H_MAX if _LIFT_H_MAX > 0 else 0.0

    return np.array([norm_x(x0), norm_y(y0), norm_z(z0),
                     norm_x(xg), norm_y(yg), norm_z(zg)], dtype=np.float32)


GOAL_DIM = 6


class Env:
    def __init__(self, goal, size=None, seed=None, shaping=False, length=None, **_):
        # size/shaping accepted for GAME_REGISTRY signature parity with
        # the other games, unused here (size only matters at goals()-
        # build time, already baked into which goal id this Env was
        # constructed with).
        if length is not None:
            raise ValueError(
                f"fetch_game's episode length is fixed at "
                f"{_MAX_EPISODE_STEPS} steps and isn't configurable via "
                f"--length, got length={length!r}")
        self._seed = seed
        self.obs_dim = OBS_DIM
        self._env = _make_env_with_retries(_ENV_ID, max_episode_steps=_MAX_EPISODE_STEPS)
        self.goal = None
        self._start_xy = None
        self._target = None
        self._set_goal_fields(goal)

    def _set_goal_fields(self, goal):
        self.goal = goal
        x0, y0, z0, xg, yg, zg = goal_point(goal)
        self._start_xy = np.array([x0, y0], dtype=np.float32)
        self._target = np.array([xg, yg, zg], dtype=np.float32)

    def set_goal(self, goal):
        self._set_goal_fields(goal)

    def _obj_xpos(self, u):
        return u._utils.get_site_xpos(u.model, u.data, "object0").copy()

    def _pack_obs(self, obs: dict) -> np.ndarray:
        packed = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        pad = OBS_DIM - packed.shape[0]
        if pad > 0:
            packed = np.concatenate([packed, np.zeros(pad, dtype=np.float32)])
        return packed

    def _place_object_at(self, u, xy: np.ndarray):
        """Places the object at a fixed xy (this goal's own x0,y0) via
        set_joint_qpos + mj_forward, instead of a fresh random draw each
        episode."""
        object_qpos = u._utils.get_joint_qpos(u.model, u.data, "object0:joint")
        object_qpos[:2] = xy
        u._utils.set_joint_qpos(u.model, u.data, "object0:joint", object_qpos)
        u._mujoco.mj_forward(u.model, u.data)

    def reset(self, seed: int = None):
        use_seed = seed if seed is not None else self._seed
        obs, info = self._env.reset(seed=use_seed)
        u = self._env.unwrapped

        self._place_object_at(u, self._start_xy)

        u.goal = self._target.copy()
        obs = u._get_obs()
        return self._pack_obs(obs), {}

    def _check_success(self, u) -> bool:
        # Plain distance check, no displacement/lift guard: the object
        # already being at the target on reset is a legitimate outcome
        # for that goal (see module docstring), not a bug to guard against.
        obj = self._obj_xpos(u)
        return float(np.linalg.norm(obj - self._target)) < SUCCESS_THRESH

    def step(self, action: int):
        a = _ACTION_TABLE[int(action)]
        obs, _native_reward, terminated, truncated, info = self._env.step(a)
        u = self._env.unwrapped
        success = self._check_success(u)
        reward = 1.0 if success else 0.0
        terminated = bool(terminated) or success
        return self._pack_obs(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()
