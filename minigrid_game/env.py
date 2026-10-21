"""
minigrid_game/env.py — Env for the 2D room-GRID task: locked doors, one
randomly-colored key per door, agent must find the key then unlock the
door, generalized from a simple 1D chain of rooms to a 2D rectangle of
rooms, connected both rightward (x) AND upward (y) — not just rightward
like a linear chain.

    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

size : (n_cols, n_rows, w, h). n_cols, n_rows must be strictly positive
       ints; w, h must be integers > 2.

Room (c, r) -- c in [0, n_cols), column index left->right; r in
[0, n_rows), row index BOTTOM->top (r=0 is the bottom row, matching the
agent's spawn corner, r=n_rows-1 is the top row). Room (0, 0) is always
the spawn room (agent starts at its bottom-left interior cell). Room
(n_cols-1, n_rows-1) -- top-right -- is the hardest to reach: getting
there needs unlocking (n_cols-1) rightward doors AND (n_rows-1) upward
doors, in some order, i.e. its Manhattan "door distance" from spawn is
(n_cols-1)+(n_rows-1).

Each room has:
  - a RIGHT door + randomly-colored key (opens the room at (c+1, r)),
    present iff c < n_cols - 1
  - an UP door + randomly-colored key (opens the room at (c, r+1)),
    present iff r < n_rows - 1
Both keys, when present, are placed at a random free cell within the
SAME room they unlock access FROM (i.e. finding both of a room's keys
never requires having already left that room) -- the same "key lives one
room before its door" pattern as a 1D chain of rooms, just doubled per
room (up to 2 keys instead of at most 1). The single top-right room has
NEITHER key (nothing beyond it to reach) -- this isn't special-cased anywhere
below, it just falls out of the c < n_cols-1 / r < n_rows-1 conditions.

length : max episode steps. None -> 2 * (h+w) * (n_cols+n_rows) -- 200
       at the default 5x5x5x5 size, roughly a 1.5-2x margin over the
       realistic optimal path to the hardest (top-right) goal (a
       BFS-verified ~56-60-cell shortest path, ~90-130 actions once
       turning/pickup/toggle are counted). Deliberately tight rather
       than generous: eval wall-time is dominated by episodes that run
       to the max length, and if the single hardest goal ends up
       practically unsolvable at this budget for some seeds, that's
       fine -- it's one goal out of hundreds, not something to design
       around.

goal : (x, y) cell. None -> top-right interior cell of the top-right
       room. Must be in goals(size) if given explicitly.

seed : >= 0 if given, reused every reset() unless reset() is itself
       given an explicit seed; None -> a fresh random seed drawn on
       construction AND on every reset() that doesn't pass its own.

Door position is FIXED (local mid-row for right doors, local mid-column
for up doors). Door COLOR and key POSITION are
randomized fresh every episode.

Agent (x, y) position + facing direction are ALWAYS packed into the
observation.
"""

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Key, Goal
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions


N_ACTIONS  = 6
KEY_COLORS = ["yellow", "red", "blue", "purple", "green", "grey"]

# Size of the extra (x, y, dx+1, dy+1) block Env._pack_obs appends after
# the flattened image -- always present here (see module docstring), the
# single source of truth minigrid_game/architecture.py imports instead
# of hardcoding "4".
POS_DIR_DIM = 4

# Direction the agent starts each episode facing (0 = dir_vec (1,0),
# "facing right") -- arbitrary but fixed.
_AGENT_SPAWN_DIR = 0


# ---------------------------------------------------------------------------
# Layout geometry — pure functions of `size`, no env instance / RNG needed
# ---------------------------------------------------------------------------

def grid_dims(size: tuple) -> tuple:
    n_cols, n_rows, w, h = size
    grid_w = n_cols * (w + 1) + 1
    grid_h = n_rows * (h + 1) + 1
    return grid_w, grid_h


def _room_x0(c: int, w: int) -> int:
    """Left interior x-edge of column c (0-indexed, left->right)."""
    return c * (w + 1) + 1


def _room_y0(r: int, h: int, n_rows: int) -> int:
    """Top interior y-edge of row r (0-indexed, BOTTOM->top -- r=0 is
    the bottom row, so it gets the LARGEST y0 since y increases
    downward in array/image space; r=n_rows-1, the top row, gets the
    smallest y0)."""
    return (n_rows - 1 - r) * (h + 1) + 1


def _agent_spawn(size: tuple) -> tuple:
    """Bottom-left interior cell of room (0, 0) -- equivalently, the
    bottom-left interior cell of the whole grid (1, grid_h-2)."""
    _, grid_h = grid_dims(size)
    return (1, grid_h - 2)


def goals(size: tuple, subset: str = "all") -> list:
    """All valid (x, y) goal cells — every interior floor cell of every
    room, EXCLUDING the agent spawn cell. subset accepted-but-unused,
    interface parity with crafter_game.env.goals(size, subset=...).
    Unlike the other games, size has no module-level default here: every
    element (n_cols, n_rows, room_w, room_h) genuinely determines which
    goals exist, so there's no size-independent fallback to fall back
    to — core/train.py's DEFAULT_SIZE always supplies one when --size is
    omitted on the command line."""
    n_cols, n_rows, w, h = size
    excluded = {_agent_spawn(size)}

    out = []
    for r in range(n_rows):
        y0 = _room_y0(r, h, n_rows)
        for c in range(n_cols):
            x0 = _room_x0(c, w)
            for dx in range(w):
                for dy in range(h):
                    cell = (x0 + dx, y0 + dy)
                    if cell not in excluded:
                        out.append(cell)
    return out


def _default_goal(size: tuple) -> tuple:
    """Top-right interior cell of the top-right room (n_cols-1,
    n_rows-1) -- the hardest room to reach."""
    n_cols, n_rows, w, h = size
    x0 = _room_x0(n_cols - 1, w)
    y0 = _room_y0(n_rows - 1, h, n_rows)
    return (x0 + w - 1, y0)


def room_id(goal: tuple, size: tuple) -> tuple:
    """(col, row) room index for a goal (x, y) -- for
    core/curriculum.py's optional group_fn (per-room ALP/entropy
    logging aggregation), returning a 2D (col, row) tuple instead of a
    single int since rooms form a grid here, not a chain."""
    n_cols, n_rows, w, h = size
    x, y = goal
    c = (x - 1) // (w + 1)
    r = n_rows - 1 - (y - 1) // (h + 1)
    return (c, r)


def build_group_fn(size: tuple):
    """core/train.py hook (called via hasattr dispatch): group_fn for
    the curriculum's per-group ALP/entropy logging."""
    return lambda g: room_id(g, size)


def build_goal_vectors(goals: list, size: tuple, embeddings_path: str = None,
                       embeddings_matrix=None):
    """core/train.py hook: fixed numeric goal vectors for the
    curriculum competence MLP / DeepSet. Normalized (x, y) in [0,1]^2 —
    embeddings_path/embeddings_matrix accepted-but-unused, interface
    parity."""
    grid_w, grid_h = grid_dims(size)
    vectors = {g: np.array([g[0] / (grid_w - 1), g[1] / (grid_h - 1)], dtype=np.float32)
              for g in goals}
    return vectors, 2


# ---------------------------------------------------------------------------
# Grid construction (the only RNG-dependent part: door colors, key
# positions)
# ---------------------------------------------------------------------------

def gen_grid(size: tuple, goal: tuple, rng: np.random.Generator):
    """
    Builds a Grid of n_cols x n_rows rooms (w x h each), each room
    connected to its right neighbor (if any) and its upper neighbor (if
    any) by a locked door at a FIXED local position, each door's key
    randomly colored and randomly placed somewhere in the room the door
    opens FROM.

    Returns (grid, agent_start_pos, agent_start_dir).
    """
    n_cols, n_rows, w, h = size
    grid_w, grid_h = grid_dims(size)

    grid = Grid(grid_w, grid_h)
    grid.wall_rect(0, 0, grid_w, grid_h)

    agent_pos = _agent_spawn(size)
    grid.set(goal[0], goal[1], Goal())

    for r in range(n_rows):
        y0 = _room_y0(r, h, n_rows)
        for c in range(n_cols):
            x0 = _room_x0(c, w)
            has_right = c < n_cols - 1
            has_up    = r < n_rows - 1
            if not has_right and not has_up:
                continue

            if has_right:
                color = KEY_COLORS[int(rng.integers(len(KEY_COLORS)))]
                door_y = y0 + h // 2
                for y in range(y0, y0 + h):
                    if y != door_y:
                        grid.set(x0 + w, y, Wall())
                grid.set(x0 + w, door_y, Door(color, is_locked=True))
                _place_key(grid, rng, x0, y0, w, h, color, agent_pos, goal)

            if has_up:
                color = KEY_COLORS[int(rng.integers(len(KEY_COLORS)))]
                door_x = x0 + w // 2
                for x in range(x0, x0 + w):
                    if x != door_x:
                        grid.set(x, y0 - 1, Wall())
                grid.set(door_x, y0 - 1, Door(color, is_locked=True))
                _place_key(grid, rng, x0, y0, w, h, color, agent_pos, goal)

    return grid, agent_pos, _AGENT_SPAWN_DIR


def _place_key(grid, rng, x0, y0, w, h, color, agent_pos, goal):
    forbidden = {agent_pos, goal}
    candidates = [
        (x, y)
        for x in range(x0, x0 + w)
        for y in range(y0, y0 + h)
        if (x, y) not in forbidden and grid.get(x, y) is None
    ]
    kx, ky = candidates[int(rng.integers(len(candidates)))]
    grid.set(kx, ky, Key(color))


# ---------------------------------------------------------------------------
# ActionReducer — reduced 6-action space (drops Actions.done)
# ---------------------------------------------------------------------------

class ActionReducer:
    ACTIONS = [Actions.left, Actions.right, Actions.forward,
              Actions.pickup, Actions.drop, Actions.toggle]

    @classmethod
    def to_minigrid(cls, action: int) -> int:
        return int(cls.ACTIONS[action])

    @classmethod
    def n(cls) -> int:
        return len(cls.ACTIONS)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class _MiniGridRoomGrid(MiniGridEnv):
    """Internal MiniGridEnv subclass — thin, just wires gen_grid() into
    MiniGrid's required _gen_grid() hook. Use Env below instead of this
    directly (matches crafter_game/env.py's wrapper-class pattern)."""

    def __init__(self, size, goal, length, render_mode=None):
        self._room_size = size
        self._goal = goal
        grid_w, grid_h = grid_dims(size)
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(mission_space=mission_space, width=grid_w, height=grid_h,
                         max_steps=length, see_through_walls=False,
                         render_mode=render_mode)

    def _gen_grid(self, width, height):
        rng = np.random.default_rng(self.np_random.integers(0, 2**31 - 1))
        grid, pos, d = gen_grid(self._room_size, self._goal, rng)
        self.grid = grid
        self.agent_pos = pos
        self.agent_dir = d
        self.mission = "reach the goal"

    def step(self, mg_action: int):
        # No-op if toggling a door that's already open / already
        # unlocked (avoids wasting the action).
        if mg_action == int(Actions.toggle):
            cell = self.grid.get(*self.front_pos)
            if isinstance(cell, Door) and (cell.is_open or not cell.is_locked):
                mg_action = int(Actions.done)
        return super().step(mg_action)


class Env:
    def __init__(self, size: tuple = (5, 5, 5, 5), length: int = None,
                goal: tuple = None, seed: int = None, shaping: bool = False,
                **_):
        n_cols, n_rows, w, h = size
        assert isinstance(n_cols, int) and n_cols > 0, \
            f"size[0] (n_cols) must be a strictly positive int, got {n_cols!r}"
        assert isinstance(n_rows, int) and n_rows > 0, \
            f"size[1] (n_rows) must be a strictly positive int, got {n_rows!r}"
        assert isinstance(w, int) and w > 2, \
            f"size[2] (w) must be an int > 2, got {w!r}"
        assert isinstance(h, int) and h > 2, \
            f"size[3] (h) must be an int > 2, got {h!r}"

        if length is None:
            length = 2 * (h + w) * (n_cols + n_rows)

        if goal is None:
            goal = _default_goal(size)
        else:
            assert goal in goals(size), \
                f"goal {goal!r} is not a valid cell for size={size}"

        self._seed_is_fixed = seed is not None
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        else:
            assert isinstance(seed, int) and seed >= 0, \
                f"seed must be a non-negative int, got {seed!r}"

        grid_w, grid_h = grid_dims(size)
        assert grid_w <= 256 and grid_h <= 256, (
            f"agent (x, y) is packed into uint8 (range 0-255) alongside "
            f"the image — grid is {grid_w}x{grid_h}, too large for that.")

        self.size = size
        self.length = length
        self.goal = goal
        self._seed = seed
        self.shaping = shaping
        self._env = _MiniGridRoomGrid(size, goal, length)

    def _pack_obs(self, obs: dict) -> np.ndarray:
        """Flattened image + agent (x, y, dx+1, dy+1) -- always packed
        (see module docstring)."""
        image = obs["image"]
        ax, ay = self._env.agent_pos
        dx, dy = self._env.dir_vec
        extra = np.array([ax, ay, dx + 1, dy + 1], dtype=np.uint8)
        return np.concatenate([image.reshape(-1), extra])

    def reset(self, seed: int = None):
        if seed is not None:
            use_seed = seed
        elif self._seed_is_fixed:
            use_seed = self._seed
        else:
            use_seed = int(np.random.randint(0, 2**31 - 1))
            self._seed = use_seed
        obs, info = self._env.reset(seed=use_seed)
        return self._pack_obs(obs), info

    def step(self, action: int):
        mg_action = ActionReducer.to_minigrid(action)
        obs, _native_reward, terminated, truncated, info = self._env.step(mg_action)
        # Strictly binary reward by default -- MiniGrid's own
        # speed-shaped native reward is discarded unless shaping.
        if self.shaping:
            reward = _native_reward
        else:
            reward = 1.0 if terminated else 0.0
        return self._pack_obs(obs), reward, terminated, truncated, info

    def set_goal(self, goal: tuple):
        """Change the goal used from the NEXT reset() onward."""
        assert goal in goals(self.size), \
            f"goal {goal!r} is not a valid cell for size={self.size}"
        self.goal = goal
        self._env._goal = goal

    def close(self):
        self._env.close()
