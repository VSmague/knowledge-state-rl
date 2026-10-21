"""
crafter_game/env.py — Env for Crafter, matching the shared Env contract
exactly (see minigrid_game/env.py for the same contract on a different
game):
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

size : world size IN TILES (Crafter's own `area` param) — default (64,
       64). NOT the same as the (64,64,3) pixel observation shape, which
       stays fixed internally (Crafter's own render view/size, not
       exposed here).
length : max episode steps. None -> Crafter's own default (10000).
goal : one of the 22 achievement names (str). None -> no targeted goal;
       episode just runs until Crafter's own `done`, reward always 0.0
       (no early success termination possible without a goal).
seed : >= 0 if given (0 allowed). None -> a fresh random seed is drawn,
       and drawn AGAIN on every reset() call that doesn't pass its own
       seed explicitly — "always fresh unless fixed" (see note on reset()
       below).

Reward (default, shaping=False): STRICTLY BINARY — 1.0 the step the
targeted goal is unlocked, 0.0 every other step, episode terminates
immediately on that step. Crafter's own native reward is discarded.

Reward (shaping=True): same final-goal 1.0 + terminate, plus a small
one-time bonus (DEFAULT_TECH_TREE_BONUS) for each shaping helper unlock
toward the current goal (hard tech-tree ancestors, plus any sword when
the goal is eat_cow / defeat_zombie / defeat_skeleton — see
crafter_game/tech_tree.py). Curriculum success remains `terminated`
(final goal only). Same CLI flag as minigrid_game's `--shaping`,
different mechanism per game.

disable_survival : if True, monkey-patches the underlying crafter package
       (process-wide, once) so hunger/thirst/sleep never deplete health
       and zombies/skeletons never threaten the player (see
       _disable_survival_mechanics() below for exactly what's patched).
       World generation itself is untouched — zombies/skeletons still
       spawn, they just vanish harmlessly on their first tick.

crafter.Env regenerates the world on .reset() via hash(seed, episode), so
there's no need to reconstruct the whole Env (Textures, World, …) every
episode — only _seed is updated when a new base seed is requested, then
.reset() is called.
"""

import os

import numpy as np
import crafter

from crafter_game.tech_tree import shaping_bonus_achievements


ACHIEVEMENTS = list(crafter.constants.achievements)   # 22 canonical goal names, fixed order
N_ACTIONS    = 17
OBS_SHAPE    = (64, 64, 3)
DEFAULT_TECH_TREE_BONUS = 0.1

# Achievements tied to survival mechanics (hunger/thirst/sleep) or combat,
# excluded from the "crafting" goals() subset — see goals(subset=...).
# By itself this only restricts which achievements are offered as
# curriculum/eval goals; the underlying crafter.Env mechanics are still
# live unless disable_survival=True is also passed to Env (see below).
_SURVIVAL_ACHIEVEMENTS = frozenset({
    "collect_sapling", "collect_drink", "eat_cow", "eat_plant",
    "place_plant", "defeat_zombie", "defeat_skeleton", "wake_up",
})
CRAFTING_ACHIEVEMENTS = [a for a in ACHIEVEMENTS if a not in _SURVIVAL_ACHIEVEMENTS]


# ---------------------------------------------------------------------------
# "Deep" survival/combat removal — monkey-patches the underlying crafter
# package itself (not just which achievements we target as goals). Checked
# against danijar/crafter's objects.py / engine.py source directly:
#   - Player._update_life_stats(self) is the method that increments
#     _hunger/_thirst/_fatigue each step and drives food/water/energy
#     depletion and the _recover counter that raises/lowers health from
#     survival neglect. No-opping it means health only ever changes from
#     direct combat damage.
#   - Zombie.update(self) / Skeleton.update(self) are what make them chase
#     the player, shoot arrows, and deal damage. Replacing them with
#     "remove self from the world" means they still spawn (per crafter's
#     own worldgen — untouched, so terrain/world generation stays
#     identical) but vanish on their very first tick, before ever acting.
#     Safer than patching worldgen's spawn conditions directly, which are
#     interleaved with unrelated tile/resource spawn logic.
#   - World.remove(obj) (engine.py) is the real removal API: marks
#     obj.removed=True and clears it from the world's object/chunk index.
# Cows are left alone — eat_cow just isn't offered as a goal when
# --crafter_goals crafting is used; cows don't threaten the player, so
# there's nothing to neutralize there.
# ---------------------------------------------------------------------------

_survival_patched = False


def _disable_survival_mechanics() -> None:
    global _survival_patched
    if _survival_patched:
        return
    # Assert the patch TARGETS exist before overwriting them — silently
    # setting a never-before-existing attribute would "succeed" and patch
    # nothing, since Python lets you add arbitrary attributes to a class.
    for cls, name in [(crafter.objects.Player, "_update_life_stats"),
                      (crafter.objects.Zombie, "update"),
                      (crafter.objects.Skeleton, "update")]:
        assert hasattr(cls, name), (
            f"crafter.objects.{cls.__name__}.{name} doesn't exist — the "
            f"installed crafter version's internals don't match what "
            f"_disable_survival_mechanics() was written against. Don't "
            f"trust this patch; inspect crafter.objects source directly.")
    crafter.objects.Player._update_life_stats = lambda self: None
    crafter.objects.Zombie.update = lambda self: self.world.remove(self)
    crafter.objects.Skeleton.update = lambda self: self.world.remove(self)
    _survival_patched = True


def goals(size: tuple = None, subset: str = "all") -> list:
    """All valid goals. size is optional/unused for interface parity with
    minigrid_game/env.py's goals(size) — Crafter's achievement list
    doesn't depend on world size.

    subset: "all" (default, 22 achievements) or "crafting" (14 — the pure
    wood/stone/iron/diamond tool tech tree, excluding anything survival
    or combat related). See CRAFTING_ACHIEVEMENTS / _SURVIVAL_ACHIEVEMENTS.
    """
    if subset == "crafting":
        return list(CRAFTING_ACHIEVEMENTS)
    if subset != "all":
        raise ValueError(f"goals(subset=...) must be 'all' or 'crafting', got {subset!r}")
    return list(ACHIEVEMENTS)


class Env:
    def __init__(self, size: tuple = (64, 64), length: int = None,
                goal: str = None, seed: int = None, shaping: bool = False,
                disable_survival: bool = False):
        assert goal is None or goal in ACHIEVEMENTS, \
            f"goal must be one of {ACHIEVEMENTS} or None, got {goal!r}"

        if disable_survival:
            _disable_survival_mechanics()

        if length is None:
            length = 10000   # Crafter's own default

        # seed=None means "give me a fresh random world every episode" —
        # tracked via _seed_is_fixed so reset() knows whether to keep
        # reusing the same seed or draw a brand new one each time.
        self._seed_is_fixed = seed is not None
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        else:
            assert isinstance(seed, int) and seed >= 0, \
                f"seed must be a non-negative int, got {seed!r}"

        self.size = size
        self.length = length
        self.goal = goal
        self.shaping = bool(shaping)
        self.tech_tree_bonus = DEFAULT_TECH_TREE_BONUS if self.shaping else 0.0
        self._seed = seed
        self._env = crafter.Env(area=size, length=length, seed=seed)
        self._prev_achievements = {k: 0 for k in ACHIEVEMENTS}
        self._bonus_given = set()  # prereq names already rewarded this ep

    def reset(self, seed: int = None):
        if seed is not None:
            use_seed = seed
        elif self._seed_is_fixed:
            use_seed = self._seed
        else:
            use_seed = int(np.random.randint(0, 2**31 - 1))
            self._seed = use_seed

        # Re-seed without rebuilding — crafter.Env.reset() regen world via
        # hash(_seed, _episode); reset _episode so the next reset() uses ep 1.
        if self._env._seed != use_seed:
            self._env._seed = use_seed
            self._env._episode = 0

        obs = self._env.reset()
        self._prev_achievements = {k: 0 for k in ACHIEVEMENTS}
        self._bonus_given = set()
        return obs, {}

    def step(self, action: int):
        obs, _native_reward, done, info = self._env.step(action)

        achievements = info.get("achievements", {})
        newly = [
            name for name in ACHIEVEMENTS
            if achievements.get(name, 0) > self._prev_achievements.get(name, 0)
        ]
        self._prev_achievements = dict(achievements)

        success = False
        reward = 0.0
        if self.goal is not None:
            if self.goal in newly:
                success = True
                reward = 1.0
            elif self.tech_tree_bonus > 0.0:
                helpers = shaping_bonus_achievements(self.goal)
                for name in newly:
                    if name in helpers and name not in self._bonus_given:
                        self._bonus_given.add(name)
                        reward += self.tech_tree_bonus

        terminated = bool(success)           # final goal only
        truncated  = bool(done) and not terminated

        info = dict(info)
        info["goal_success"] = success
        info["tech_tree_bonus_given"] = sorted(self._bonus_given)
        return obs, reward, terminated, truncated, info

    def set_goal(self, goal: str):
        """Change the goal used from the NEXT reset() onward. self.goal
        is read directly by step() each call, so this could just be an
        attribute assignment — kept as an explicit method (not just
        `env.goal = x`) for interface parity with other games, and to
        keep the validation assert in one place."""
        assert goal is None or goal in ACHIEVEMENTS, \
            f"goal must be one of {ACHIEVEMENTS} or None, got {goal!r}"
        self.goal = goal

    def close(self):
        pass


# ---------------------------------------------------------------------------
# core/train.py hooks — everything crafter_game-specific that core/train.py
# needs, gathered here so core/ can call it via hasattr(env_module, ...)
# instead of branching on `args.game == "crafter_game"`.
# ---------------------------------------------------------------------------

# Default Qwen / LLM frozen goal embeddings.
DEFAULT_LLM_GOAL_EMBEDS = "crafter_game/crafter_goal_embeddings.npy"


def resolve_goal_repr(repr_name: str, goals_list: list):
    """Map --crafter_goal_repr to curriculum/policy embedding sources.

    Returns (embeddings_matrix_or_None, frozen_npy_path_or_None).

      onehot    → (None, None)  curriculum one-hot; trainable policy embed
      tech_tree → (matrix, None) caller should save matrix then freeze policy
      llm       → (None, DEFAULT_LLM_GOAL_EMBEDS) shared frozen .npy
    """
    repr_name = (repr_name or "onehot").lower()
    if repr_name == "onehot":
        return None, None
    if repr_name == "tech_tree":
        from crafter_game.tech_tree import build_tech_tree_embeddings
        return build_tech_tree_embeddings(goals_list), None
    if repr_name == "llm":
        return None, DEFAULT_LLM_GOAL_EMBEDS
    raise ValueError(
        f"crafter_goal_repr must be onehot|tech_tree|llm, got {repr_name!r}")


def extra_env_kwargs(args) -> dict:
    """Extra Env() constructor kwargs derived from crafter_game-only CLI
    args — folded into core/train.py's env_kwargs via hasattr dispatch."""
    return dict(disable_survival=args.crafter_disable_survival)


def shaping_description() -> str:
    """One-line description of --shaping's effect for crafter_game, printed
    by core/train.py at startup when --shaping is on."""
    return f"shaped(tech_tree_bonus={DEFAULT_TECH_TREE_BONUS})"


def describe_goal_embeds(frozen_embeddings_path, crafter_goal_repr, goal_dim) -> str:
    """One-line description of the resolved goal representation, printed
    by core/train.py at startup."""
    if frozen_embeddings_path:
        return (f"goal embeds: {crafter_goal_repr} → {frozen_embeddings_path}  "
               f"(policy + curriculum, dim={goal_dim})")
    return (f"goal embeds: onehot (curriculum) + learned nn.Embedding  "
           f"(policy), dim={goal_dim}")


def build_train_goal_repr(args, curriculum_goals: list):
    """Everything core/train.py needs to set up crafter_game's goal
    representation for both the policy and the curriculum, built
    DELIBERATELY separately: the policy's GoalRepr always indexes against
    the full 22 achievements (crafter_game.architecture.GOAL_TO_IDX is a
    fixed global mapping, independent of --crafter_goals), while the
    curriculum only needs vectors for `curriculum_goals` (possibly the
    14-goal "crafting" subset) — reusing one matrix for both breaks the
    moment they're different lengths (policy expects 22 rows, would get
    14).

    Returns (policy_embeddings_matrix, frozen_embeddings_path,
    curriculum_embeddings_matrix).
    """
    policy_embeddings_matrix, frozen_embeddings_path = resolve_goal_repr(
        args.crafter_goal_repr, ACHIEVEMENTS)
    if args.crafter_goal_repr == "llm" and not os.path.isfile(frozen_embeddings_path):
        raise FileNotFoundError(
            f"--crafter_goal_repr llm requires {frozen_embeddings_path} "
            f"(generate via python -m crafter_game.goal_llm_embed)")

    curriculum_embeddings_matrix = None
    if args.crafter_goal_repr == "tech_tree":
        if args.crafter_goals == "all":
            curriculum_embeddings_matrix = policy_embeddings_matrix
        else:
            from crafter_game.tech_tree import build_tech_tree_embeddings
            curriculum_embeddings_matrix = build_tech_tree_embeddings(curriculum_goals)
    elif args.crafter_goal_repr == "llm":
        full = np.load(frozen_embeddings_path).astype(np.float32)
        if args.crafter_goals == "all":
            curriculum_embeddings_matrix = full
        else:
            idx = [ACHIEVEMENTS.index(g) for g in curriculum_goals]
            curriculum_embeddings_matrix = full[idx]
    # onehot: stays None -> build_goal_vectors falls through to its own
    # np.eye(len(curriculum_goals)) branch, already independent of the
    # policy's separately-sized trainable embedding table.

    return policy_embeddings_matrix, frozen_embeddings_path, curriculum_embeddings_matrix


def build_goal_vectors(goals: list, size: tuple, embeddings_path: str = None,
                       embeddings_matrix=None):
    """core/train.py hook: fixed numeric goal vectors for the curriculum
    competence MLP / DeepSet. Priority: embeddings_matrix, then
    embeddings_path (.npy), else one-hot. size accepted-but-unused, for
    interface parity with minigrid_game.env.build_goal_vectors."""
    n = len(goals)
    if embeddings_matrix is not None:
        emb = np.asarray(embeddings_matrix, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ValueError(
                f"curriculum embeddings_matrix: expected ({n}, D), got {emb.shape}")
        vectors = {g: emb[i] for i, g in enumerate(goals)}
        return vectors, int(emb.shape[1])
    if embeddings_path:
        emb = np.load(embeddings_path).astype(np.float32)
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ValueError(
                f"curriculum embeddings at {embeddings_path}: expected "
                f"({n}, D), got {emb.shape}")
        vectors = {g: emb[i] for i, g in enumerate(goals)}
        return vectors, int(emb.shape[1])
    eye = np.eye(n, dtype=np.float32)
    vectors = {g: eye[i] for i, g in enumerate(goals)}
    return vectors, n
