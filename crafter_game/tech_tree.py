"""
crafter_game/tech_tree.py — Hard / soft achievement dependencies for Crafter.

Used by crafter_game/env.py (optional intermediate tech-tree bonus
rewards, --shaping) and the "tech_tree" --crafter_goal_repr option
(structured goal feature vectors, via build_tech_tree_embeddings).

Hard edges = true recipe/tool prerequisites (data.yaml / objects.py).
Soft edges = practical help only (e.g. sword raises melee damage).
"""

from functools import lru_cache

import numpy as np


# Hard prerequisites (must unlock A before B is possible)
DEPENDENCIES = [
    ("collect_wood", "place_table"),
    ("collect_sapling", "place_plant"),
    ("place_plant", "eat_plant"),
    ("place_table", "make_wood_pickaxe"),
    ("collect_wood", "make_wood_pickaxe"),
    ("place_table", "make_wood_sword"),
    ("collect_wood", "make_wood_sword"),
    ("make_wood_pickaxe", "collect_stone"),
    ("make_wood_pickaxe", "collect_coal"),
    ("collect_stone", "place_stone"),
    ("collect_stone", "place_furnace"),
    ("collect_stone", "make_stone_pickaxe"),
    ("place_table", "make_stone_pickaxe"),
    ("collect_stone", "make_stone_sword"),
    ("place_table", "make_stone_sword"),
    ("make_stone_pickaxe", "collect_iron"),
    ("collect_iron", "make_iron_pickaxe"),
    ("collect_coal", "make_iron_pickaxe"),
    ("place_furnace", "make_iron_pickaxe"),
    ("place_table", "make_iron_pickaxe"),
    ("collect_iron", "make_iron_sword"),
    ("collect_coal", "make_iron_sword"),
    ("place_furnace", "make_iron_sword"),
    ("place_table", "make_iron_sword"),
    ("make_iron_pickaxe", "collect_diamond"),
]

# Any sword raises melee damage for these goals (soft help).
SWORDS = (
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)
MELEE_SOFT_GOALS = (
    "eat_cow",
    "defeat_zombie",
    "defeat_skeleton",
)

# Soft / practical help (dashed in plots + shaping bonuses when enabled)
SOFT_DEPENDENCIES = [
    (sword, goal) for sword in SWORDS for goal in MELEE_SOFT_GOALS
]

ROOTS = [
    "collect_wood", "collect_sapling", "collect_drink", "eat_cow",
    "defeat_zombie", "defeat_skeleton", "wake_up",
]

# Structured features for goal embeddings (curriculum + optional frozen policy).
ACTION_TYPES = ("collect", "place", "make", "eat", "defeat", "wake")
OBJECT_TYPES = (
    "wood", "stone", "iron", "sword", "pickaxe", "other",
)


def _action_type(goal: str) -> str:
    for a in ACTION_TYPES:
        if goal.startswith(a):
            return a
    raise ValueError(f"unknown action type for goal {goal!r}")


def _object_types(goal: str) -> list:
    """Coarse object / material tokens from the achievement name."""
    rest = goal
    for a in ACTION_TYPES:
        if goal.startswith(a + "_"):
            rest = goal[len(a) + 1:]
            break
        if goal.startswith(a):
            rest = goal[len(a):].lstrip("_")
            break
    parts = rest.split("_") if rest else []
    known = ("wood", "stone", "iron", "sword", "pickaxe")
    found = [p for p in parts if p in known]
    # Deduplicate, preserve order
    out = []
    for p in found:
        if p not in out:
            out.append(p)
    if not out:
        out = ["other"]
    return out


def goal_tech_tree_vector(goal: str, all_goals: list) -> np.ndarray:
    """Hand-crafted feature vector for one Crafter achievement.

    Layout (fixed order):
      [0 : n)         per-goal codes: 3 = identity, 2 = hard prereq,
                      1 = soft help (e.g. sword → melee), 0 = none
      [n : n+A)       one-hot action type
      [n+A : n+A+O)   multi-hot object tokens
                      (wood/stone/iron/sword/pickaxe/other)

    For n=22, A=6, O=6 → dim = 22+6+6 = 34.
    """
    n = len(all_goals)
    A, O = len(ACTION_TYPES), len(OBJECT_TYPES)
    dim = n + A + O
    v = np.zeros(dim, dtype=np.float32)
    g2i = {g: i for i, g in enumerate(all_goals)}

    # Soft first, then hard, then identity — higher codes overwrite.
    if goal in MELEE_SOFT_GOALS:
        for sword in SWORDS:
            if sword in g2i:
                v[g2i[sword]] = 1.0
    for p in hard_prerequisites(goal):
        if p in g2i:
            v[g2i[p]] = 2.0
    if goal in g2i:
        v[g2i[goal]] = 3.0

    act = _action_type(goal)
    v[n + ACTION_TYPES.index(act)] = 1.0

    base = n + A
    for obj in _object_types(goal):
        v[base + OBJECT_TYPES.index(obj)] = 1.0

    return v


def build_tech_tree_embeddings(goals: list) -> np.ndarray:
    """Stack tech-tree feature vectors. Shape (n_goals, D).

    `goals` must be in the same order as crafter_game.env.ACHIEVEMENTS when
    used for the policy GoalRepr.
    """
    goals = list(goals)
    mat = np.stack([goal_tech_tree_vector(g, goals) for g in goals], axis=0)
    return mat.astype(np.float32)


def _direct_parents() -> dict:
    parents = {}
    for pre, goal in DEPENDENCIES:
        parents.setdefault(goal, []).append(pre)
    return parents


@lru_cache(maxsize=None)
def hard_prerequisites(goal: str) -> frozenset:
    """Transitive hard ancestors of `goal` (empty for roots / unknown)."""
    parents = _direct_parents()
    out = set()
    stack = list(parents.get(goal, []))
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(parents.get(p, []))
    return frozenset(out)


@lru_cache(maxsize=None)
def shaping_bonus_achievements(goal: str) -> frozenset:
    """Achievements that yield a one-time shaping bonus toward `goal`.

    = hard transitive prerequisites
      ∪ any sword unlock, if `goal` is a melee soft-help goal
        (eat_cow / defeat_zombie / defeat_skeleton).
    """
    out = set(hard_prerequisites(goal))
    if goal in MELEE_SOFT_GOALS:
        out.update(SWORDS)
    return frozenset(out)
