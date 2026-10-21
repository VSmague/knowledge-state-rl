"""
core/curriculum.py — Goal curriculum strategies for MAGELLAN-style
training. Fully game-agnostic: never assumes a goal is an (x,y) tuple —
works identically for minigrid_game's coordinate goals and crafter_game's
achievement-name goals.

Available curricula
-------------------
  uniform      : sample goals uniformly at random (baseline)
  online       : OnlineALP — per-goal ALP = |mean_recent - mean_old|
  magellan     : MagellanALP — learned MLP, ALP = |C_now - C_old|
  magellan_ks  : MagellanKnowledgeState — MLP conditioned on a DeepSet
                 summary of recent outcomes (fixed-size "knowledge
                 state"), environment-agnostic by construction (no
                 dependence on n_rooms / grid size / etc.)

Game-agnostic by construction, in two independent ways:

  1. Goal grouping (for per-group ALP/entropy logging) is an OPTIONAL
     `group_fn: goal -> hashable` passed in by the caller — minigrid_game
     passes `lambda g: minigrid_game.env.room_id(g, size)` for a real
     per-room breakdown; if not given (the default, and what crafter_game
     uses), each goal is its own group (identity function), so
     `get_alp_all_groups()` still gives a meaningful breakdown either way:
     per-room for minigrid_game, one entry per achievement for crafter_game.
     This file never computes a goal's group itself, only looks one up
     via `group_fn` if given.

  2. Goal representation is decoupled from the competence networks:
     CompetenceMLP / CompetenceMLPWithKnowledgeState take an explicit
     `goal_dim` and expect the CALLER to supply `goal_vectors: dict[goal,
     array-like]` (a fixed-size numeric vector per goal, computed ONCE
     outside this file — normalized (x,y) for minigrid_game, e.g. a
     one-hot or embedding vector for crafter_game). This file never computes a
     goal's numeric representation itself, only looks it up.

DeepSetKnowledgeEncoder is fully generic on top of this: it operates on
whatever-sized goal vectors are inside each window item (see the note in
MagellanKnowledgeState below on how goal_dim interacts with each item's
width).
"""

import numpy as np
import random
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Goal-space utilities
# ---------------------------------------------------------------------------

def sample_uniform(valid_goals: list):
    return random.choice(valid_goals)


def _identity_group(goal):
    """Default group_fn: every goal is its own group — gives a per-goal
    breakdown (meaningful for crafter_game's 22 discrete achievements;
    minigrid_game passes its own room_id-based group_fn instead for a
    coarser per-room breakdown)."""
    return goal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_by_weight(valid_goals: list, scores: np.ndarray,
                      epsilon: float, deadzone: float = 0.0):
    """ALP-weighted sampling with epsilon-greedy uniform fallback."""
    if random.random() < epsilon:
        return sample_uniform(valid_goals)
    probs = _alp_scores_to_probs(scores, deadzone=deadzone)
    return valid_goals[int(np.random.choice(len(valid_goals), p=probs))]


def _draw_candidates(valid_goals: list, subsample: int) -> list:
    """Fresh random subset of `valid_goals`, size `subsample`, redrawn on
    every call so no goal is permanently excluded across episodes — only
    the goals actually SCORED this one call are restricted; epsilon-
    greedy exploration (handled by the caller, before this is reached)
    still draws uniformly from the FULL `valid_goals`, never just the
    subset, so this never shrinks any goal's floor probability. Returns
    `valid_goals` itself (not a copy) when `subsample <= 0` or is not
    smaller than the full set, so callers can cheaply check
    `candidates is valid_goals` to skip the restrict/restore dance
    around get_alp_all() entirely. This turns the
    O(|G|) per-episode scoring cost (the dominant cost at large goal
    counts, e.g. magellan_ks's own encoder forward pass across |G|
    candidates every episode) into O(subsample), at the cost of noisier
    goal selection -- every goal still gets an equal chance of being IN
    the scored subset each episode, so pressure toward genuinely high-
    ALP goals is preserved on average, just per-draw noisier. Off by
    default (subsample=0): every goal is scored on every call."""
    if subsample <= 0 or subsample >= len(valid_goals):
        return valid_goals
    return random.sample(valid_goals, subsample)


def _omni_reweight(p: np.ndarray, p_theta: float = 0.1) -> np.ndarray:
    """OMNI (2024) reweighting f(p) applied element-wise."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    denom = p + p_theta * (1.0 - 2.0 * p)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    return (1.0 - p_theta) * p / denom


def _alp_scores_to_probs(scores: np.ndarray, deadzone: float = 0.0) -> np.ndarray:
    """
    Convert raw ALP scores to a probability vector: p(g) ∝ ALP(g).

    deadzone : if the single HIGHEST score is still below this absolute
        value, treat the whole vector as "no real signal" and fall back
        to uniform instead of proportional sampling. Without this, once
        every goal's TRUE ALP has converged near zero (everything either
        saturated-success or stuck-at-zero), proportional sampling still
        divides by whatever tiny total exists, and a goal that happens
        to have slightly noisier (not more real) competence -- typically
        whichever goal is currently sampled the most, since it has the
        most recent training data feeding the competence net's local fit
        -- ends up with almost all the probability mass by sheer ratio,
        even though the absolute difference is numerical noise.
        Confirmed on real Fetch training logs: alp_max~0.0001-0.01 while
        one already-mastered goal (100% success for a long time) still
        received ~80-90% of the sampling budget, starving goals that had
        never once succeeded. 0.0 (default) disables this check (only a
        LITERAL zero total falls back to uniform) -- opt-in via each
        curriculum's alp_deadzone kwarg.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores
    if deadzone > 0.0 and scores.max() < deadzone:
        return np.ones(n, dtype=np.float64) / n
    total = scores.sum()
    if total <= 0:
        return np.ones(n, dtype=np.float64) / n
    return scores / total


# ---------------------------------------------------------------------------
# Uniform (baseline)
# ---------------------------------------------------------------------------

class Uniform:
    """Uniform random goal sampling — no estimator."""

    def __init__(self, valid_goals: list, **_):
        self.valid_goals = valid_goals
        self.epsilon     = 0.0

    def update(self, goal, success: bool): pass

    def sample_goal(self):
        return sample_uniform(self.valid_goals)

    def anneal_epsilon(self, episode: int, epsilon_period: int, eps_min: float = 0.1):
        pass

    def get_alp(self, goal) -> float: return 0.0
    def get_alp_all(self) -> dict: return {g: 0.0 for g in self.valid_goals}
    def get_alp_all_groups(self) -> dict: return {}

    def get_sampling_probs(self) -> dict:
        n = len(self.valid_goals)
        return {g: 1.0 / n for g in self.valid_goals}

    def stats(self) -> dict: return {"epsilon": 0.0}


# ---------------------------------------------------------------------------
# OnlineALP  (per-goal sliding window, ALP = |mean_recent - mean_old|)
# ---------------------------------------------------------------------------

class OnlineALP:
    """
    Per-goal ALP estimator. Buffer size = 2*M.
    ALP(g) = |mean(last M) - mean(first M)|.

    Parameters
    ----------
    valid_goals : list of goals (any hashable — (x,y) tuples, strings, ...)
    group_fn    : goal -> hashable group id, for get_alp_all_groups()
                  logging. Default: identity (each goal is its own group).
    M           : half-window size
    eps_min     : minimum epsilon after annealing
    candidate_subsample : if >0 and < |valid_goals|, sample_goal() only
                  scores a fresh random subset of this size each episode
                  instead of every goal — see _draw_candidates's
                  docstring. 0 (default) = score every goal, unchanged
                  behavior.
    """

    def __init__(self, valid_goals: list, group_fn=None,
                 M: int = 10, eps_min: float = 0.1,
                 candidate_subsample: int = 0, **_):
        self.valid_goals  = valid_goals
        self.group_fn     = group_fn or _identity_group
        self.M            = M
        self.eps_min      = eps_min
        self.epsilon      = 1.0
        self.candidate_subsample = candidate_subsample
        self.buffers      = {g: deque(maxlen=2 * M) for g in valid_goals}
        self.goal_to_group = {g: self.group_fn(g) for g in valid_goals}

    def update(self, goal, success: bool):
        self.buffers[goal].append(float(success))

    def get_alp(self, goal) -> float:
        buf = list(self.buffers[goal])
        if len(buf) < 2:
            return 0.0
        mid = len(buf) // 2
        return abs(np.mean(buf[mid:]) - np.mean(buf[:mid]))

    def get_alp_all(self) -> dict:
        return {g: self.get_alp(g) for g in self.valid_goals}

    def get_alp_all_groups(self) -> dict:
        group_alp = {}
        for g in self.valid_goals:
            group_alp.setdefault(self.goal_to_group[g], []).append(self.get_alp(g))
        return {r: float(np.mean(v)) for r, v in group_alp.items()}

    def sample_goal(self):
        if random.random() < self.epsilon:
            return sample_uniform(self.valid_goals)
        candidates = _draw_candidates(self.valid_goals, self.candidate_subsample)
        scores = np.array([self.get_alp(g) for g in candidates], dtype=np.float64)
        return _sample_by_weight(candidates, scores, 0.0)

    def anneal_epsilon(self, episode: int, epsilon_period: int, eps_min: float = None):
        eps_min = eps_min if eps_min is not None else self.eps_min
        self.epsilon = eps_min + (1.0 - eps_min) * np.exp(-episode / max(epsilon_period, 1))

    def get_sampling_probs(self) -> dict:
        scores = np.array([self.get_alp(g) for g in self.valid_goals],
                          dtype=np.float64)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(self.valid_goals)) / len(self.valid_goals)
        n = len(self.valid_goals)
        mixed = (1 - self.epsilon) * probs + self.epsilon / n
        return {g: float(p) for g, p in zip(self.valid_goals, mixed)}

    def stats(self) -> dict:
        alp_vals = [self.get_alp(g) for g in self.valid_goals]
        visited  = sum(1 for g in self.valid_goals if len(self.buffers[g]) > 0)
        return {"alp_mean": float(np.mean(alp_vals)),
                "alp_max":  float(np.max(alp_vals)),
                "goals_visited": visited,
                "goals_total":   len(self.valid_goals),
                "epsilon":       self.epsilon}


# ---------------------------------------------------------------------------
# CompetenceMLP  (used by MagellanALP)
# ---------------------------------------------------------------------------

class CompetenceMLP(nn.Module):
    def __init__(self, goal_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DeepSet knowledge-state encoder
# ---------------------------------------------------------------------------

class DeepSetKnowledgeEncoder(nn.Module):
    """
    Summarises a variable-size set of recent (goal_vec, success) outcomes
    into a fixed-size "knowledge state" vector — independent of how many
    goals the environment has (and, now, independent of the goal
    representation's own dimensionality — see item_dim below), so it
    generalises across environments.

    pool_mode="mean" (default): knowledge_state = rho( mean_i( phi(item_i) ) )
    pool_mode="attention": knowledge_state = rho( Σ_i α_i · phi(item_i) ),
    with α_i a learned, content-dependent weighting instead of the fixed
    1/K every item gets under "mean" (mean is the degenerate case of
    attention where the query is forced uniform).
    Concretely (Set Transformer's PMA, Lee et al. 2019): a learned query
    `self.query` (n_heads of them) attends over per-item keys/values
    projected from phi(item_i); n_heads>1 lets the pooling keep several
    independent weightings instead of collapsing everything into one
    compromise (concatenated before rho).
    pool_mode="lstm": knowledge_state = rho( LSTM(phi(item_1..K))_final ).
    Items are fed through a single-layer LSTM IN ORDER (oldest first —
    matches how recent_window/train_buffer already store them, no
    reordering needed) and the final hidden state is used as the pooled
    summary. Unlike "mean"/"attention", this mode is NOT permutation-
    invariant on purpose: it's the one pool_mode that captures recency/
    trend within a window (was an agent stuck for the whole window, or
    just started succeeding?) — info mean/attention structurally discard
    since they treat the window as an unordered set. Despite the class
    name, this mode is not a "deep set" reduction at all; kept in this
    class anyway since it's a drop-in third option behind the same
    interface.
    pool_mode="transformer": knowledge_state = rho( maskedmean_i(
    TransformerEncoder(phi(item_1..K))_i ) ). The one mode where items
    attend to EACH OTHER (multi-head self-attention, `transformer_layers`
    stacked `nn.TransformerEncoderLayer`s) before pooling — "attention"
    above only lets a single learned query look at items, items never
    interact with one another there; "transformer" can in principle pick
    up patterns like "goal A tends to succeed right after goal B does"
    within the window. Permutation-EQUIVARIANT self-attention followed by
    a masked mean, so still permutation-invariant overall like "mean"/
    "attention" (no positional encoding is added — add one if recency
    within the window should matter here too, deliberately left out for
    now to isolate "cross-item interaction" from "order sensitivity",
    which "lstm" already covers separately).

    All non-lstm modes share the exact same external interface
    (forward/batch_forward -> (knowledge_dim,) or (B, knowledge_dim)),
    so aux_loss/cl_loss/get_alp/ks_buffer in MagellanKnowledgeState are
    unaffected by which mode is chosen — only this class's internals
    change. One real interaction to know about: MagellanKnowledgeState's
    contrastive loss (_augmented_views) builds its two views by
    subsampling a window and SORTING the kept indices (so the relative
    chronological order of surviving items is preserved in both views)
    — this matters only for "lstm" (order-sensitive); harmless no-op for
    "mean"/"attention" (order-invariant either way).

    Caveats (untested — read before relying on either non-default mode
    for a real run):
      - attention: adds real degrees of freedom with no direct
        supervision on what α should be (only an indirect gradient
        through the competence BCE), so nothing guarantees it converges
        to the hoped-for "downweight redundant items" behavior over
        "mean" — worth logging α in practice before trusting it.
      - Both attention and lstm: get_alp compares C(ks) across ks_buffer
        snapshots over time — if the learned reduction (α weights, or
        the LSTM's gates) keeps drifting during training independent of
        the agent's real outcomes, ks itself becomes noisier across
        snapshots than under "mean" (which has no trainable weights in
        its reduction step, so ks can only change if the underlying data
        does), which could inject spurious ALP.
      - lstm specifically: a single stochastic outcome landing in the
        LAST position of the window has outsized influence on the final
        hidden state (that's the recency-bias we want), but it also
        means ks_now can be noisier step-to-step than mean/attention
        purely from which single episode happened to land last — the
        flip side of being recency-sensitive at all.
      - --ks_aux_loss_weight > 0 forces ks to stay decodable back into
        per-goal rates for every goal in the window, which works against
        full attention collapse onto one item (attention mode) — worth
        trying together rather than either non-default mode alone.

    Item width = goal_dim + 1 (the goal vector, plus the success flag),
    so this works for any game's goal vector dimensionality.
    """

    def __init__(self, goal_dim: int, item_hidden: int = 32, knowledge_dim: int = 16,
                pool_mode: str = "mean", n_heads: int = 1, transformer_layers: int = 2):
        super().__init__()
        if pool_mode not in ("mean", "attention", "lstm", "transformer"):
            raise ValueError(
                f"pool_mode must be 'mean', 'attention', 'lstm', or "
                f"'transformer', got {pool_mode!r}")
        if pool_mode not in ("attention", "transformer") and n_heads != 1:
            raise ValueError("n_heads only applies to pool_mode='attention' "
                             "or 'transformer'")
        self.goal_dim = goal_dim
        self.item_dim = goal_dim + 1
        self.knowledge_dim = knowledge_dim
        self.pool_mode = pool_mode
        self.n_heads = n_heads
        self.phi = nn.Sequential(
            nn.Linear(self.item_dim, item_hidden), nn.ReLU(),
            nn.Linear(item_hidden, item_hidden), nn.ReLU(),
        )
        # Learned placeholder used by MagellanKnowledgeState's
        # main_leave_self_out_mode="mask_embed" to overwrite a self-
        # occurrence's phi(item) output (see batch_forward's self_mask
        # param) instead of corrupting its raw (goal, success) pair
        # before phi() the way mode="mask" does (a hand-picked sentinel
        # like -1.0 still has to be filtered through the same phi()
        # weights as every real value, and — unlike mean pooling's
        # gentle dilution — an order/interaction-sensitive pool_mode
        # like lstm/transformer must actively process or attend to it
        # as if it were a real event). Always
        # constructed (regardless of pool_mode or whether main_leave_
        # self_out is ever used) so it's a fixed, uniform part of every
        # checkpoint's state_dict — load_competence_mlp_with_ks handles
        # its absence in checkpoints saved before this was added.
        self.self_mask_embedding = nn.Parameter(torch.zeros(item_hidden))
        if pool_mode == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=item_hidden, nhead=n_heads,
                dim_feedforward=item_hidden * 2, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            pooled_dim = item_hidden
        elif pool_mode == "attention":
            # One learned query per head; keys/values projected from
            # phi(item) per head (concatenated via the Linear's output
            # width), attention score = scaled dot product (standard
            # Transformer scaling, 1/sqrt(item_hidden)) — masked to -1e9
            # (not -inf, to avoid NaN softmax rows for all-padding
            # windows; those rows get zeroed via is_empty below anyway,
            # same as pool_mode="mean").
            self.query = nn.Parameter(torch.randn(n_heads, item_hidden) * (item_hidden ** -0.5))
            self.w_k = nn.Linear(item_hidden, item_hidden * n_heads, bias=False)
            self.w_v = nn.Linear(item_hidden, item_hidden * n_heads, bias=False)
            self._attn_scale = item_hidden ** 0.5
            pooled_dim = item_hidden * n_heads
        elif pool_mode == "lstm":
            self.lstm = nn.LSTM(input_size=item_hidden, hidden_size=item_hidden,
                                batch_first=True)
            pooled_dim = item_hidden
        else:
            pooled_dim = item_hidden
        self.rho = nn.Sequential(
            nn.Linear(pooled_dim, item_hidden), nn.ReLU(),
            nn.Linear(item_hidden, knowledge_dim),
        )

    def _attention_pool(self, h: torch.Tensor, mask: torch.Tensor,
                        return_weights: bool = False):
        """
        h    : (B, K, item_hidden) — phi(items), already computed.
        mask : (B, K) — 1.0 real item, 0.0 padding.
        Returns (B, item_hidden * n_heads) — heads concatenated. If
        return_weights, returns (pooled, alpha) instead, alpha shape
        (B, K, H) — the attention weight each item received, for
        interpretability probing only (see attention_weights() below);
        every existing call site passes return_weights=False (default)
        and is completely unaffected.
        """
        B, K, D = h.shape
        k = self.w_k(h).view(B, K, self.n_heads, D)         # (B,K,H,D)
        v = self.w_v(h).view(B, K, self.n_heads, D)         # (B,K,H,D)
        q = self.query.view(1, 1, self.n_heads, D)          # (1,1,H,D)
        scores = (q * k).sum(dim=-1) / self._attn_scale     # (B,K,H)
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        alpha = torch.softmax(scores, dim=1)                # (B,K,H)
        pooled = (alpha.unsqueeze(-1) * v).sum(dim=1)        # (B,H,D)
        pooled = pooled.reshape(B, self.n_heads * D)
        return (pooled, alpha) if return_weights else pooled

    @torch.no_grad()
    def attention_weights(self, items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Diagnostic-only: returns the (B, K, n_heads) attention weight
        each window item received, for pool_mode="attention". Not called
        anywhere in the training/sampling hot path — purely for
        interpretability probing against a loaded checkpoint's encoder.
        Raises for pool_mode="transformer"/"lstm"/"mean" (no per-item
        weights to report: masked-mean and the LSTM's final hidden state
        don't expose a comparable scalar per item)."""
        if self.pool_mode != "attention":
            raise ValueError(
                f"attention_weights() is only defined for pool_mode="
                f"'attention', got {self.pool_mode!r} (mean/transformer/"
                f"lstm have no comparable per-item weight to report)")
        h = self.phi(items)
        _, alpha = self._attention_pool(h, mask, return_weights=True)
        return alpha

    def _lstm_pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, K, item_hidden) — phi(items). Real items occupy the
               FIRST `length` positions of each row (chronological order,
               oldest first — matches _pad_windows' zero-padding-at-the-
               end convention, and the live window's natural deque
               order), padding trails at the end.
        mask : (B, K) — 1.0 real item, 0.0 padding (trailing).
        Returns (B, item_hidden) — final LSTM hidden state per row.

        Rows with zero real items get length clamped to 1 (pack_padded_
        sequence rejects length 0) — the LSTM runs one step over a
        zero-padding vector for those rows, but the result is discarded
        by batch_forward's is_empty override, same as every other mode.
        """
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]

    def _transformer_selfattend(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Self-attention stage for pool_mode="transformer" — items attend
        to each other (multi-head self-attention, `transformer_layers`
        stacked layers) before being pooled by a masked mean
        (_transformer_pool).

        h    : (B, K, item_hidden) — phi(items).
        mask : (B, K) — 1.0 real item, 0.0 padding.
        Returns (B, K, item_hidden) — contextualized item embeddings,
        same shape as h.
        """
        key_padding_mask = mask == 0  # True = padded, torch convention
        # A fully-padded row would make self-attention softmax over an
        # all-True mask (undefined) — batch_forward zeroes these rows
        # afterward anyway (is_empty), but guard here so the transformer
        # itself never sees one.
        all_empty = key_padding_mask.all(dim=1)
        if all_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_empty] = False
        return self.transformer(h, src_key_padding_mask=key_padding_mask)

    def _transformer_pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, K, item_hidden) — phi(items).
        mask : (B, K) — 1.0 real item, 0.0 padding.
        Returns (B, item_hidden) — masked mean over the transformer's
        contextualized item embeddings (items attend to each other via
        self-attention first, unlike "mean"/"attention" above, which
        never let items interact).
        """
        out = self._transformer_selfattend(h, mask)
        m = mask.unsqueeze(-1)
        summed = (out * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        items : (K, item_dim) tensor of (goal_vec..., success) for K
                recent outcomes (K can be anything, including 0).
        Returns: (knowledge_dim,) vector.
        """
        if items.shape[0] == 0:
            return torch.zeros(self.knowledge_dim, device=items.device)
        h = self.phi(items)
        if self.pool_mode == "mean":
            pooled = h.mean(dim=0)
        else:
            mask = torch.ones(1, items.shape[0], device=items.device)
            pool_fns = {"attention": self._attention_pool, "lstm": self._lstm_pool,
                       "transformer": self._transformer_pool}
            pooled = pool_fns[self.pool_mode](h.unsqueeze(0), mask).squeeze(0)
        return self.rho(pooled)

    def batch_forward(self, items: torch.Tensor, mask: torch.Tensor,
                      self_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Vectorized batch version — one forward pass for a whole minibatch
        of (zero-padded) windows, instead of one encoder call per example.

        items : (B, K, item_dim) zero-padded tensor
        mask  : (B, K) — 1.0 for real items, 0.0 for padding
        self_mask : (B, K) bool, optional — True at positions whose
                phi(item) output should be overwritten with the learned
                self_mask_embedding BEFORE pooling (main_leave_self_out_
                mode="mask_embed" — see self_mask_embedding's docstring).
                Independent of `mask`: a self-masked position still
                counts as a real (non-padding) item, it just gets a
                fixed learned representation instead of one derived
                from its real (goal, success) values.
        Returns: (B, knowledge_dim) tensor.
        """
        h = self.phi(items)
        if self_mask is not None:
            h = torch.where(self_mask.unsqueeze(-1), self.self_mask_embedding, h)
        m = mask.unsqueeze(-1)
        counts = m.sum(dim=1)
        if self.pool_mode == "mean":
            summed = (h * m).sum(dim=1)
            pooled = summed / counts.clamp(min=1.0)
        elif self.pool_mode == "attention":
            pooled = self._attention_pool(h, mask)
        elif self.pool_mode == "lstm":
            pooled = self._lstm_pool(h, mask)
        else:
            pooled = self._transformer_pool(h, mask)
        out = self.rho(pooled)

        is_empty = (counts.squeeze(-1) == 0)
        if is_empty.any():
            out = out.clone()
            out[is_empty] = 0.0
        return out


def load_competence_mlp_with_ks(state_dict: dict, goal_dim: int,
                                transformer_n_heads: int | None = None) -> "CompetenceMLPWithKnowledgeState":
    """
    Reconstruct a CompetenceMLPWithKnowledgeState from a saved state_dict,
    inferring hidden_dim / item_hidden / knowledge_dim from tensor shapes.
    `goal_dim` MUST be given explicitly — it can't be disambiguated from
    knowledge_dim by shape alone, since net.0.weight's input width is
    goal_dim + knowledge_dim for either. Each game has a fixed, known
    goal_dim (2 for minigrid_game, the embedding size for crafter_game),
    so the caller always has this available.

    pool_mode/n_heads are inferred, not passed: "knowledge_encoder.query"
    only exists in the state_dict for pool_mode="attention", and
    "knowledge_encoder.lstm.weight_ih_l0" only for pool_mode="lstm" (see
    DeepSetKnowledgeEncoder), so presence/shape fully disambiguates which
    mode + how many heads the checkpoint was trained with. Exception:
    pool_mode="transformer" — transformer_layers IS inferable (count of
    distinct "transformer.layers.N." indices present), but nhead is NOT
    (nn.MultiheadAttention's in_proj_weight shape doesn't depend on it),
    so it must be passed explicitly via `transformer_n_heads` for those
    checkpoints.
    """
    hidden_dim    = state_dict["net.0.weight"].shape[0]
    knowledge_dim = state_dict["net.0.weight"].shape[1] - goal_dim
    item_hidden   = state_dict["knowledge_encoder.phi.0.weight"].shape[0]
    transformer_layer_keys = [k for k in state_dict if k.startswith("knowledge_encoder.transformer.layers.")]
    if "knowledge_encoder.query" in state_dict:
        pool_mode = "attention"
        n_heads   = state_dict["knowledge_encoder.query"].shape[0]
        transformer_layers = 2
    elif "knowledge_encoder.lstm.weight_ih_l0" in state_dict:
        pool_mode, n_heads, transformer_layers = "lstm", 1, 2
    elif transformer_layer_keys:
        pool_mode = "transformer"
        transformer_layers = len({k.split(".")[3] for k in transformer_layer_keys})
        if transformer_n_heads is None:
            raise ValueError(
                "checkpoint is pool_mode='transformer' — pass transformer_n_heads "
                "explicitly (not inferable from the state_dict shapes)")
        n_heads = transformer_n_heads
    else:
        pool_mode, n_heads, transformer_layers = "mean", 1, 2
    model = CompetenceMLPWithKnowledgeState(
        goal_dim=goal_dim, hidden_dim=hidden_dim, item_hidden=item_hidden,
        knowledge_dim=knowledge_dim, pool_mode=pool_mode, n_heads=n_heads,
        transformer_layers=transformer_layers)
    # strict=False + explicit check, not plain strict=True: checkpoints
    # saved before self_mask_embedding was added (see its docstring)
    # legitimately lack that one key — everything else must still match
    # exactly, so any OTHER missing/unexpected key still raises.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"knowledge_encoder.self_mask_embedding"}
    if unexpected or (set(missing) - allowed_missing):
        raise RuntimeError(
            f"load_competence_mlp_with_ks: state_dict mismatch — "
            f"missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


class CompetenceMLPWithKnowledgeState(nn.Module):
    """
    Competence estimator conditioned on a knowledge-state vector:
      C(g | knowledge_state) = MLP( concat(goal_vec, knowledge_state) )
    """

    def __init__(self, goal_dim: int, hidden_dim: int = 64,
                item_hidden: int = 32, knowledge_dim: int = 16,
                pool_mode: str = "mean", n_heads: int = 1,
                transformer_layers: int = 2):
        super().__init__()
        self.goal_dim = goal_dim
        self.knowledge_encoder = DeepSetKnowledgeEncoder(
            goal_dim, item_hidden, knowledge_dim,
            pool_mode=pool_mode, n_heads=n_heads,
            transformer_layers=transformer_layers)
        self.net = nn.Sequential(
            nn.Linear(goal_dim + knowledge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, goal_vec: torch.Tensor,
               recent_outcomes: torch.Tensor) -> torch.Tensor:
        """
        goal_vec        : (B, goal_dim) goal representations
        recent_outcomes : (K, goal_dim+1) recent (goal_vec..., success)
                          outcomes — SHARED across the whole batch. One
                          knowledge_state is computed ONCE and broadcast
                          to all B goals.
        Returns: (B,) competence values.
        """
        B = goal_vec.shape[0]
        ks = self.knowledge_encoder(recent_outcomes)
        ks_b = ks.unsqueeze(0).expand(B, -1)
        x = torch.cat([goal_vec, ks_b], dim=-1)
        return self.net(x).squeeze(-1)

    def knowledge_state(self, recent_outcomes: torch.Tensor) -> torch.Tensor:
        return self.knowledge_encoder(recent_outcomes)


# ---------------------------------------------------------------------------
# MagellanALP  (learned MLP, ALP = |C_now - C_old|)
# ---------------------------------------------------------------------------

class MagellanALP:
    """
    MAGELLAN competence estimator (Gaven et al., 2025).

    ALP(g) = |C_θnow(g) - C_θold(g)|
    where θold is the oldest snapshot in the weight buffer (N snapshots).

    Parameters
    ----------
    valid_goals   : list of goals (any hashable)
    goal_vectors  : dict[goal, array-like] — fixed numeric representation
                    per goal, computed ONCE by the caller (normalized xy
                    for minigrid_game, an embedding/one-hot for
                    crafter_game).
    goal_dim      : length of each vector in goal_vectors.
    group_fn      : goal -> hashable group id, for get_alp_all_groups().
                    Default: identity.
    M             : outcome buffer size
    N             : weight snapshot count
    hidden_dim    : MLP hidden size
    lr            : MLP learning rate
    train_every   : retrain MLP every K episodes
    train_steps   : gradient steps per retrain
    eps_min       : minimum epsilon
    device        : torch device string
    alp_deadzone  : see _alp_scores_to_probs's docstring — if the max
                    ALP score is below this, fall back to uniform
                    instead of proportional sampling. 0.0 (default) =
                    deadzone never fires.
    balanced_training : if True, the competence-MLP's own training loss
                    is re-weighted so each goal PRESENT in outcome_buffer
                    contributes equally to the gradient, regardless of
                    how many times it was actually played (outcome_buffer
                    composition mirrors the curriculum's own sampling
                    distribution, so a goal sampled 5x more often gets
                    5x the raw gradient weight under the old unweighted
                    BCE — this compounds the "rich get richer" ALP-
                    fixation pathology at the competence-net-training
                    level, on top of the goal-sampling level that
                    alp_deadzone already addresses). False (default) =
                    old behavior (plain unweighted BCE).
    """

    def __init__(self, valid_goals: list, goal_vectors: dict, goal_dim: int,
                 group_fn=None, M: int = 200, N: int = 10,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 train_every: int = 10, train_steps: int = 20,
                 eps_min: float = 0.1, device: str = "cpu",
                 alp_deadzone: float = 0.0, balanced_training: bool = False,
                 candidate_subsample: int = 0,
                 **_):
        self.valid_goals = valid_goals
        self.goal_dim    = goal_dim
        self.group_fn    = group_fn or _identity_group
        self.M           = M
        self.N           = N
        self.train_every = train_every
        self.train_steps = train_steps
        self.eps_min     = eps_min
        self.device      = device
        self.alp_deadzone = alp_deadzone
        self.balanced_training = balanced_training
        self.candidate_subsample = candidate_subsample
        self.epsilon     = 1.0
        self._ep_count   = 0

        self.current_mlp = CompetenceMLP(goal_dim, hidden_dim).to(device)
        self.opt         = optim.Adam(self.current_mlp.parameters(), lr=lr)
        self.bce         = nn.BCELoss()

        import copy
        self.weight_buffer  = deque(maxlen=N)
        self.weight_buffer.append(copy.deepcopy(self.current_mlp.state_dict()))
        self.outcome_buffer = deque(maxlen=M)
        self._hidden_dim    = hidden_dim
        self._last_loss     = None

        self._goal_t = {
            g: torch.tensor(np.asarray(goal_vectors[g]), dtype=torch.float32, device=device)
            for g in valid_goals}

    def update(self, goal, success: bool):
        import copy
        self.outcome_buffer.append((goal, self._goal_t[goal], float(success)))
        self._ep_count += 1
        if self._ep_count % self.train_every == 0:
            self._train_mlp()
            self.weight_buffer.append(
                copy.deepcopy(self.current_mlp.state_dict()))

    def _train_mlp(self):
        if len(self.outcome_buffer) < 2:
            return
        gt = torch.stack([v for _, v, _ in self.outcome_buffer])
        yt = torch.tensor([r for _, _, r in self.outcome_buffer],
                          dtype=torch.float32, device=self.device)

        # Full-batch training (same batch reused every train_steps) — under
        # balanced_training, per-example weights make each PRESENT goal's
        # total contribution to the loss equal, regardless of how many raw
        # occurrences it has in outcome_buffer. Computed once outside the
        # gradient loop since the buffer contents don't change within it.
        weights = None
        if self.balanced_training:
            counts = defaultdict(int)
            for g, _, _ in self.outcome_buffer:
                counts[g] += 1
            w = torch.tensor([1.0 / counts[g] for g, _, _ in self.outcome_buffer],
                             dtype=torch.float32, device=self.device)
            weights = w / w.mean()

        self.current_mlp.train()
        losses = []
        for _ in range(self.train_steps):
            self.opt.zero_grad()
            preds = self.current_mlp(gt)
            if weights is not None:
                per_ex = F.binary_cross_entropy(preds, yt, reduction="none")
                loss = (per_ex * weights).mean()
            else:
                loss = self.bce(preds, yt)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        self.current_mlp.eval()
        self._last_loss = float(np.mean(losses))

    def get_last_loss(self):
        return self._last_loss

    @torch.no_grad()
    def _comp(self, mlp: nn.Module, goal) -> float:
        return float(mlp(self._goal_t[goal].unsqueeze(0)).item())

    @torch.no_grad()
    def competence(self, goal) -> float:
        return self._comp(self.current_mlp, goal)

    @torch.no_grad()
    def get_alp(self, goal) -> float:
        c_now = self.competence(goal)
        old_mlp = CompetenceMLP(self.goal_dim, self._hidden_dim).to(self.device)
        old_mlp.load_state_dict(self.weight_buffer[0])
        old_mlp.eval()
        return abs(c_now - self._comp(old_mlp, goal))

    def get_alp_all(self) -> dict:
        old_mlp = CompetenceMLP(self.goal_dim, self._hidden_dim).to(self.device)
        old_mlp.load_state_dict(self.weight_buffer[0])
        old_mlp.eval()
        with torch.no_grad():
            gt    = torch.stack([self._goal_t[g] for g in self.valid_goals])
            c_now = self.current_mlp(gt).cpu().numpy()
            c_old = old_mlp(gt).cpu().numpy()
        return {g: float(abs(cn - co))
                for g, cn, co in zip(self.valid_goals, c_now, c_old)}

    def get_alp_all_groups(self) -> dict:
        alp_dict = self.get_alp_all()
        group_alp = {}
        for g, alp in alp_dict.items():
            group_alp.setdefault(self.group_fn(g), []).append(alp)
        return {r: float(np.mean(v)) for r, v in group_alp.items()}

    @torch.no_grad()
    def get_competence_all(self) -> dict:
        gt    = torch.stack([self._goal_t[g] for g in self.valid_goals])
        comps = self.current_mlp(gt).cpu().numpy()
        return {g: float(c) for g, c in zip(self.valid_goals, comps)}

    def sample_goal(self):
        if random.random() < self.epsilon:
            return sample_uniform(self.valid_goals)
        candidates = _draw_candidates(self.valid_goals, self.candidate_subsample)
        if candidates is self.valid_goals:
            alp_dict = self.get_alp_all()
        else:
            saved, self.valid_goals = self.valid_goals, candidates
            try:
                alp_dict = self.get_alp_all()
            finally:
                self.valid_goals = saved
        scores = np.array([alp_dict[g] for g in candidates], dtype=np.float64)
        return _sample_by_weight(candidates, scores, 0.0, deadzone=self.alp_deadzone)

    def anneal_epsilon(self, episode: int, epsilon_period: int, eps_min: float = None):
        eps_min = eps_min if eps_min is not None else self.eps_min
        self.epsilon = eps_min + (1.0 - eps_min) * np.exp(-episode / max(epsilon_period, 1))

    def get_sampling_probs(self) -> dict:
        alp_dict = self.get_alp_all()
        scores   = np.array([alp_dict[g] for g in self.valid_goals],
                            dtype=np.float64)
        probs = _alp_scores_to_probs(scores, deadzone=self.alp_deadzone)
        n = len(self.valid_goals)
        mixed = (1 - self.epsilon) * probs + self.epsilon / n
        return {g: float(p) for g, p in zip(self.valid_goals, mixed)}

    def stats(self) -> dict:
        alp_dict = self.get_alp_all()
        alp_vals = list(alp_dict.values())
        with torch.no_grad():
            gt    = torch.stack([self._goal_t[g] for g in self.valid_goals])
            comps = self.current_mlp(gt).cpu().numpy()
        return {"alp_mean":        float(np.mean(alp_vals)),
                "alp_max":         float(np.max(alp_vals)),
                "competence_mean": float(comps.mean()),
                "competence_max":  float(comps.max()),
                "buffer_size":     len(self.outcome_buffer),
                "n_snapshots":     len(self.weight_buffer),
                "epsilon":         self.epsilon}


# ---------------------------------------------------------------------------
# MagellanKnowledgeState  (MLP conditioned on a learned knowledge state)
# ---------------------------------------------------------------------------

class MagellanKnowledgeState:
    """
    MAGELLAN variant where the competence estimator is conditioned on a
    "knowledge state" — a fixed-size summary of the agent's recent
    outcomes, produced by a DeepSet encoder. Fully environment-agnostic:
    the knowledge state has a fixed dimension regardless of how many
    goals the environment has, or what a goal even IS structurally.

      ks_now    = DeepSet(K most recent (goal_vec, success) outcomes)
      C(g | ks) = MLP(goal_vec, ks)
      ALP(g)    = |C_θ(g | ks_now) - C_θ(g | ks_old)|   (alp_mode="endpoints")
               or mean_i |C_θ(g | ks_{i+1}) - C_θ(g | ks_i)|
                  along ks_buffer + ks_now                   (alp_mode="path")
               or mean_i |C_θ(g | ks_now) - C_θ(g | ks_i)|
                  over ks_buffer snapshots                   (alp_mode="to_now")
               or |f(p_recent) - f(p_gradual)| with double EMA of
                  C(g|ks_s) along the ks_buffer trajectory   (alp_mode="omni_ks")

    Optional post-processing (off by default, except omni_ks):
      - alp_ema : EMA α along the ks trajectory. For endpoints/path/to_now,
                  optional pre-smoothing. For omni_ks, α of both EMA filters
                  (defaults to 0.1 when unset — OMNI uses 0.1).

    Internally, recent_window / train_buffer store (goal_id, success)
    only — goal_id is the env-agnostic hashable from valid_goals. Numeric
    goal_vectors are looked up from `_goal_t` when building DeepSet /
    competence tensors. Same math as storing full vectors, far less
    memory/CPU when goal_dim is large (e.g. Crafter Qwen 1024-d).

    Stationarity argument: MagellanALP's C_now vs C_old comparison runs
    two DIFFERENT weight snapshots of the competence MLP, so a change in
    the result conflates "the network's beliefs drifted" with "the
    agent's real competence changed" — the two are not separable.
    MagellanKnowledgeState instead keeps ONE set of weights (the current
    MLP, continuously trained) and varies the CONDITIONING INPUT: C(g |
    ks_old) and C(g | ks_now) are the SAME function evaluated at two
    different ks vectors. Any difference between them is then a genuine
    difference in what the current, fixed function believes about
    competence at ks_old vs ks_now, not an artifact of the function
    itself having changed — which is why this only needs a buffer of old
    ks vectors (to feed as alternate inputs to the current model), not
    old model weight snapshots like MagellanALP.

    Parameters
    ----------
    valid_goals   : list of goals (any hashable)
    goal_vectors  : dict[goal, array-like] — fixed numeric representation
                    per goal, computed ONCE by the caller.
    goal_dim      : length of each vector in goal_vectors.
    group_fn      : goal -> hashable group id, for get_alp_all_groups().
                    Default: identity.
    M             : max size of the training buffer
    N             : ks snapshot count (for ALP's ks_old)
    K             : size of the recent-outcomes window for ks_now —
                    raising this doesn't uniformly help: a global
                    recency window dilutes relevant signal with
                    outcomes on unrelated goals as K grows, so bigger
                    isn't automatically better (confirmed on the
                    minigrid_game/fetch_game K-sweeps).
    hidden_dim    : competence MLP hidden size
    item_hidden   : DeepSet per-item encoder hidden size
    knowledge_dim : size of the knowledge_state vector
    pool_mode     : "mean" (default), "attention", "lstm", or
                    "transformer" — see DeepSetKnowledgeEncoder's
                    docstring for the mechanism of each and their caveats
                    (untested convergence behavior, ALP-stability risk
                    from a drifting learned reduction). "lstm" is the
                    only mode sensitive to the window's chronological
                    order (recency/trend within the window) —
                    "mean"/"attention"/"transformer" all discard order
                    ("transformer" lets items attend to EACH OTHER via
                    self-attention before a masked mean, but that's still
                    permutation-equivariant, no positional encoding).
    n_heads       : number of independent attention queries when
                    pool_mode="attention" (ignored for "mean"/"lstm";
                    must be 1 for those), or nhead for pool_mode=
                    "transformer"'s self-attention layers. >1 lets the
                    pooling keep several independent weightings instead
                    of one compromise.
    transformer_layers : number of stacked self-attention layers, only
                    used for pool_mode="transformer" (ignored otherwise).
    lr            : learning rate
    train_every   : retrain every K episodes
    train_steps   : gradient steps per retrain
    train_batch   : minibatch size sampled from train_buffer per step
    alp_mode      : "endpoints" (default) = |C(now)-C(old)| ;
                    "path" = mean consecutive |ΔC| along ks_buffer+ks_now ;
                    "to_now" = mean_i |C(now)-C(ks_i)| over buffer snapshots ;
                    "omni_ks" = |f(p_recent)-f(p_gradual)| with double EMA
                    along the ks trajectory (OMNI-style, ks-buffer axis).
                    All four of the above are RETROSPECTIVE — they only
                    ever compare ks snapshots already observed in the
                    past; confirmed empirically to give EXACTLY
                    0.0 predicted competence for goals beyond the
                    current real training frontier, at every checkpoint
                    tested — no forward-looking generalization at all.
                    "prospective_self" / "prospective_global" (see
                    _prospective_scores) instead SIMULATE one step
                    ahead: for every candidate goal g, hypothetically
                    append (g, outcome) to the live window for BOTH
                    outcome in {0, 1} (never a continuous/out-of-
                    distribution value — the network has only ever been
                    trained on real binary outcomes), producing two
                    candidate next-step ks vectors; the two branches are
                    combined via their EXACT expectation, weighted by
                    p_g = C(g|ks_now) (no stochastic sampling, no extra
                    variance). "prospective_self" measures the resulting
                    change in g's OWN competence estimate only (closer
                    to the original ALP definition, just computed by
                    simulation instead of retrospective history) —
                    tested to collapse to the same
                    ~0.0 floor as the retrospective modes for
                    never-succeeded goals, for the same underlying
                    reason: if C(g|ks) is already saturated near 0, one
                    simulated data point barely moves it either way.
                    "prospective_global" instead measures the mean
                    change in EVERY goal's competence estimate (not just
                    g's own) — genuinely non-zero for far, unvisited
                    goals when tested post-hoc on an already-trained
                    checkpoint, but NEGATIVELY correlated with which of
                    those goals actually get mastered later (-0.39 on
                    the one checkpoint tested) — consistent with it
                    picking up "how unfamiliar is g's vector to the
                    recent window" (a novelty signal) rather than
                    genuine anticipated learnability. Untested so far:
                    whether using either prospective mode LIVE as the
                    actual sampling signal (not just evaluated post-hoc
                    on a frozen checkpoint trained under a different
                    mode) changes this via the resulting feedback loop
                    — that is what these two modes are for.
    alp_ema       : EMA α in (0,1]; 0 = off for endpoints/path/to_now.
                    For omni_ks, α for both EMAs (default 0.1 if unset).
    alp_p_theta   : pθ in OMNI's reweighting f(p) (default 0.1).
    eps_min       : minimum epsilon
    device        : torch device string
    balanced_training : if True, each retrain draws a minibatch weighted
                    to counteract goal-frequency imbalance in train_buffer
                    instead of uniform random sampling over buffer index —
                    under the old behavior a goal sampled 5x more often by
                    the curriculum supplies ~5x more of every minibatch,
                    so the competence net's own gradient signal inherits
                    (and compounds) whatever imbalance the curriculum's
                    goal-sampling already has. False (default) = old
                    behavior (plain np.random.choice over the whole
                    buffer, no reweighting at all).
    balance_alpha   : only used when balanced_training=True. Per-entry
                    sampling weight = 1 / count(its goal)^(1-balance_alpha),
                    so every buffer entry belonging to goal g contributes
                    weight/count(g) -- i.e. each GOAL's total selection
                    probability scales as count(g)^balance_alpha.
                    0.0 (default) = full equalization: every goal's total
                    probability is equal regardless of how many times it
                    was actually visited (see _stratified_indices).
                    1.0 = no correction at all: weight=1 per entry,
                    identical to balanced_training=False's plain uniform
                    draw over buffer index (a goal visited 5x more
                    supplies 5x more of the minibatch).
                    Values in between (e.g. 0.5, a sqrt-law dampening
                    akin to word2vec's frequency subsampling) let a goal
                    that is genuinely being visited a lot right now (e.g.
                    a real competence frontier, not just a stale lock-in)
                    keep SOME extra weight instead of being flattened to
                    the exact same probability as a goal visited once —
                    full equalization (alpha=0) can't distinguish "over-sampled
                    because of a self-reinforcing bias worth correcting"
                    from "over-sampled because it's currently the most
                    informative goal," and treats both identically.
    """

    def __init__(self, valid_goals: list, goal_vectors: dict, goal_dim: int,
                 group_fn=None, M: int = 50_000, N: int = 10, K: int = 30,
                 hidden_dim: int = 64, item_hidden: int = 32,
                 knowledge_dim: int = 16, pool_mode: str = "mean",
                 n_heads: int = 1, transformer_layers: int = 2, lr: float = 1e-3,
                 train_every: int = 10, train_steps: int = 20,
                 train_batch: int = 256,
                 aux_loss_weight: float = 0.0, aux_mode: str = "window",
                 cl_weight: float = 0.0, cl_temperature: float = 0.2,
                 cl_subsample_frac: float = 0.7,
                 alp_mode: str = "endpoints",
                 alp_ema: float = 0.0,
                 alp_p_theta: float = 0.1,
                 prospective_subsample: int = 50,
                 eps_min: float = 0.1, device: str = "cpu",
                 alp_deadzone: float = 0.0, balanced_training: bool = False,
                 balance_alpha: float = 0.0,
                 main_leave_self_out: bool = False,
                 main_leave_self_out_mode: str = "strip",
                 candidate_subsample: int = 0,
                 **_):
        """
        main_leave_self_out : if True, strips a training example's own
            goal from its window BEFORE encoding kst for the MAIN BCE
            loss (not just the aux_mode="leave_self_out" auxiliary head,
            which only regularizes the shared encoder indirectly and
            never forces the pathway actually used for real ALP/sampling
            decisions — net(goal_vec, kst) — to stop relying on it).
            Without this, if a goal has been played
            several times recently, some of THOSE earlier plays are
            often still sitting in the window when a later training
            example for the SAME goal is captured, letting net() satisfy
            the BCE loss by reading that goal's own recent occurrences
            straight back out of kst instead of learning to infer its
            competence from goal_vec's position relative to OTHER goals
            — i.e. no incentive to ever learn genuine structural
            generalization. Confirmed empirically: predicted competence
            for goals just beyond the current training frontier tracks
            the REAL (already-nonzero) rate almost exactly at every
            checkpoint, with zero anticipatory lift for goals still at
            0% real success — no forward-looking generalization at all.
            False (default) = original behavior.
            Does NOT change the live ks_now/ks_buffer/get_alp_all
            pathway (still one shared window/ks across all goals —
            extending self-masking to that side would be a bigger,
            not-yet-attempted redesign).

        main_leave_self_out_mode : only used when main_leave_self_out=True.
            "strip" (default): _strip_own_goal's original behavior —
                self-occurrences are DELETED from the window entirely,
                shortening it (remaining entries shift to fill the gap).
                Confirmed via the bigrooms mean-vs-lstm batch
                that this reshuffles which entries end up
                temporally adjacent, which hurt pool_mode="lstm"
                specifically (worse collapse rate than baseline, 5/8 vs
                2/8) even though it strongly helped pool_mode="mean"
                (collapse eliminated, 0/8) — working theory: lstm's
                recurrent state was using a goal's OWN past occurrences
                as an anchor to detect that ITS competence had
                plateaued (the direct trigger for deprioritizing an
                already-mastered goal), and losing both the outcome
                AND the temporal slot/spacing at once removed that
                signal more thoroughly than intended.
            "mask": keeps the window the SAME length and order (no
                entries removed, no reshuffling of temporal spacing) —
                every (g', s') entry whose g' equals the example's own
                goal g has its success value replaced with a fixed
                sentinel (_MASK_SUCCESS = -1.0, outside the real [0,1]
                range) instead of being deleted. The goal identity
                (goal_vec) at that position stays visible — only the
                OUTCOME is hidden — so a pooling mode that tracks
                recency/position (lstm) can still tell "my own goal was
                attempted around here" (preserving whatever plateau-
                detection cue it was using) without being able to read
                back the actual result, forcing genuine inference from
                other goals' evidence for the one thing (the outcome)
                that would otherwise be a free shortcut. Analogous to
                BERT's masked-language-modeling: position/context stays
                visible, only the thing being predicted is hidden.
                Caveat found after testing "mask" on lstm/transformer:
                the raw sentinel -1.0 still has to
                pass through phi()'s ordinary weights like any real
                value, and unlike mean pooling's gentle dilution, an
                order/interaction-sensitive pool_mode (lstm's recurrent
                state, transformer's self-attention) must actively
                process or attend to it as if it were a genuine event —
                plausibly injecting a disruptive out-of-distribution
                signal rather than a neutral no-op. "mask_embed" (below)
                is the fix attempt.
            "mask_embed": same length/order preservation as "mask" (the
                raw (goal, success) pairs are left completely UNCHANGED
                this time), but instead of corrupting the pair before
                phi(), a boolean self_mask is passed to the encoder's
                batch_forward, which overwrites phi(item)'s OUTPUT at
                those positions with a dedicated LEARNED embedding
                (DeepSetKnowledgeEncoder.self_mask_embedding) — closer
                to BERT's actual mechanism (a trainable [MASK] token
                embedding, not a hand-picked raw sentinel run through
                the same weights as real data). The raw success value
                becomes irrelevant either way (fully overwritten after
                phi()), so what "mask" bothers to write into the pair
                doesn't matter for this mode — only the pair's PRESENCE/
                position does.

        aux_loss_weight / cl_weight : both OFF (0.0) by default, opt-in.
        In an early ablation (a handful of seeds, not settled), the
        auxiliary recovery loss alone showed high ks usage AND was the
        best-performing variant, while the contrastive loss alone showed
        low ks usage in isolation but not when combined with the
        auxiliary loss.

        aux_loss_weight : weight of the auxiliary ks-recovery loss (an
            extra head decodes ks back into per-goal success rates for
            goals that appear in the window; goals absent from the window
            are masked out) relative to the main BCE loss.
        aux_mode : "window" (default, original behavior) — the aux head
            decodes ALL goals present in the SAME window used for the
            main loss's kst, so it can satisfy the loss by reading a
            goal's own recent occurrences straight back out of ks
            instead of learning genuine cross-goal inference.
            "leave_self_out" — for each training example, its own goal's
            entries are stripped from the window BEFORE encoding a
            SEPARATE kst just for the aux loss, and only that one goal is
            supervised (target = its mean success rate in the unmasked
            window) — forces the aux head to infer a goal's competence
            purely from what OTHER goals in the window are doing. Costs
            one extra encoder forward pass per retrain step (still
            skipped entirely when aux_loss_weight=0).
        cl_weight / cl_temperature / cl_subsample_frac : weight/params of
            the contrastive (NT-Xent) loss on ks — positive pair = two
            random subsamples of the same window, negatives = every other
            window in the minibatch. cl_weight=0 skips the extra encoder
            calls entirely (no cost when off).
        alp_mode / alp_ema / alp_p_theta : see class docstring.
        alp_deadzone : see _alp_scores_to_probs's docstring — if the max
            ALP score is below this, fall back to uniform instead of
            proportional sampling. 0.0 (default) = deadzone never fires.
        """
        if alp_mode not in ("endpoints", "path", "to_now", "omni_ks",
                            "prospective_self", "prospective_self_signed",
                            "prospective_global"):
            raise ValueError(
                f"alp_mode must be 'endpoints', 'path', 'to_now', 'omni_ks', "
                f"'prospective_self', 'prospective_self_signed', or "
                f"'prospective_global', got {alp_mode!r}")
        if not (0.0 <= alp_ema <= 1.0):
            raise ValueError(f"alp_ema must be in [0, 1], got {alp_ema}")
        if not (0.0 < alp_p_theta < 1.0):
            raise ValueError(f"alp_p_theta must be in (0, 1), got {alp_p_theta}")
        if aux_mode not in ("window", "leave_self_out"):
            raise ValueError(
                f"aux_mode must be 'window' or 'leave_self_out', got {aux_mode!r}")
        if main_leave_self_out_mode not in ("strip", "mask", "mask_embed"):
            raise ValueError(
                f"main_leave_self_out_mode must be 'strip', 'mask', or "
                f"'mask_embed', got {main_leave_self_out_mode!r}")

        self.valid_goals = valid_goals
        self.goal_dim    = goal_dim
        self.group_fn    = group_fn or _identity_group
        self.M           = M
        self.N           = N
        self.K           = K
        self.train_every = train_every
        self.train_steps = train_steps
        self.train_batch = train_batch
        self.aux_loss_weight   = aux_loss_weight
        self.aux_mode          = aux_mode
        self.cl_weight         = cl_weight
        self.cl_temperature    = cl_temperature
        self.cl_subsample_frac = cl_subsample_frac
        self.alp_mode      = alp_mode
        self.alp_ema       = float(alp_ema)
        self.alp_p_theta   = float(alp_p_theta)
        self.candidate_subsample = candidate_subsample
        self.prospective_subsample = prospective_subsample
        self.eps_min       = eps_min
        self.alp_deadzone  = alp_deadzone
        self.balanced_training = balanced_training
        self.balance_alpha = balance_alpha
        self.main_leave_self_out = main_leave_self_out
        self.main_leave_self_out_mode = main_leave_self_out_mode
        self.device      = device
        self.epsilon     = 1.0
        self._ep_count   = 0

        self.current_mlp = CompetenceMLPWithKnowledgeState(
            goal_dim, hidden_dim, item_hidden, knowledge_dim,
            pool_mode=pool_mode, n_heads=n_heads,
            transformer_layers=transformer_layers).to(device)
        self._goal_idx = {g: i for i, g in enumerate(valid_goals)}
        self._num_goals = len(valid_goals)
        # Extra head: ks -> per-goal success rates for goals present in the
        # window (masked BCE — goals not sampled in the window are ignored).
        self.aux_head = nn.Linear(knowledge_dim, self._num_goals).to(device)
        self.opt = optim.Adam(
            list(self.current_mlp.parameters()) + list(self.aux_head.parameters()),
            lr=lr)
        self.bce = nn.BCELoss()

        self.recent_window = deque(maxlen=K)

        self.ks_buffer = deque(maxlen=N)
        self.ks_buffer.append(np.zeros(knowledge_dim, dtype=np.float32))

        self.train_buffer = deque(maxlen=M)

        self._hidden_dim    = hidden_dim
        self._item_hidden   = item_hidden
        self._knowledge_dim = knowledge_dim
        self._last_loss     = None
        self._last_aux_loss = None
        self._last_cl_loss  = None

        self._goal_t = {
            g: torch.tensor(np.asarray(goal_vectors[g]), dtype=torch.float32, device=device)
            for g in valid_goals}

    def _pairs_to_items(self, pairs) -> torch.Tensor:
        """(goal, success) pairs -> (n, goal_dim+1) float tensor via `_goal_t`."""
        if not pairs:
            return torch.zeros((0, self.goal_dim + 1), device=self.device)
        gvecs = torch.stack([self._goal_t[g] for g, _ in pairs], dim=0)
        succ = torch.tensor([float(s) for _, s in pairs],
                            dtype=torch.float32, device=self.device).unsqueeze(-1)
        return torch.cat([gvecs, succ], dim=-1)

    def _recent_outcomes_tensor(self) -> torch.Tensor:
        return self._pairs_to_items(self.recent_window)

    def _window_for(self, goal) -> list:
        """The (goal, success) window used as TRAINING context for a
        specific goal — the plain global recency window, identical for
        every goal."""
        return list(self.recent_window)

    def _pad_windows(self, windows: list) -> tuple:
        """
        windows : list of B raw windows (each a list of up to self.K
                  (goal, success) pairs — goal is the hashable id).
        Returns (items, mask) for DeepSetKnowledgeEncoder.batch_forward:
          items : (B, K, goal_dim+1) zero-padded tensor
          mask  : (B, K)
        """
        B, K = len(windows), self.K
        items = torch.zeros(B, K, self.goal_dim + 1, dtype=torch.float32, device=self.device)
        mask  = torch.zeros(B, K, dtype=torch.float32, device=self.device)
        for i, w in enumerate(windows):
            n = min(len(w), K)
            if n > 0:
                items[i, :n] = self._pairs_to_items(w[:n])
                mask[i, :n] = 1.0
        return items, mask

    @staticmethod
    def _strip_own_goal(batch: list) -> list:
        """[(g, w, s) ...] -> windows with every (g', s') entry whose g'
        equals that example's OWN goal g removed — see
        main_leave_self_out's docstring. Shared by the main loss and by
        _leave_self_out_batch (which does the same stripping for the
        aux head, plus its own separate target/has_signal bookkeeping)."""
        return [[(gp, sp) for gp, sp in w if gp != g] for g, w, s in batch]

    _MASK_SUCCESS = -1.0

    @staticmethod
    def _mask_own_goal(batch: list) -> list:
        """[(g, w, s) ...] -> windows of the SAME length/order as w, but
        every (g', s') entry whose g' equals that example's OWN goal g
        has its success value replaced with the fixed sentinel
        _MASK_SUCCESS (outside the real [0, 1] range) instead of being
        removed — see main_leave_self_out_mode="mask"'s docstring for
        why this is the alternative to _strip_own_goal."""
        return [[(gp, MagellanKnowledgeState._MASK_SUCCESS if gp == g else sp)
                 for gp, sp in w] for g, w, s in batch]

    def _own_goal_positions(self, batch: list) -> torch.Tensor:
        """(B, K) bool tensor, True at positions (within each window's
        first min(len(w), K) real entries, matching _pad_windows'
        convention) whose goal equals that example's OWN goal g — the
        positions main_leave_self_out_mode="mask_embed" overwrites with
        DeepSetKnowledgeEncoder.self_mask_embedding AFTER phi(), leaving
        the raw (goal, success) pairs themselves untouched (unlike
        _mask_own_goal, which corrupts the pair itself BEFORE phi() —
        see main_leave_self_out_mode's docstring for why)."""
        B, K = len(batch), self.K
        out = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        for i, (g, w, s) in enumerate(batch):
            n = min(len(w), K)
            for j in range(n):
                gp, _ = w[j]
                if gp == g:
                    out[i, j] = True
        return out

    def _window_goal_success_targets(self, windows: list) -> tuple:
        """
        Per-window sparse success-rate targets for the auxiliary decoder.

        For each goal g that appears at least once in window w, target[g]
        is the mean success of g within w. Goals absent from w are masked
        out (no supervision — we do not know their rate from this window).
        """
        B, G = len(windows), self._num_goals
        aux_t = torch.zeros(B, G, dtype=torch.float32, device=self.device)
        aux_mask = torch.zeros(B, G, dtype=torch.float32, device=self.device)
        for i, w in enumerate(windows):
            if not w:
                continue
            succ_sum = defaultdict(float)
            counts = defaultdict(int)
            for g, s in w:
                j = self._goal_idx[g]
                succ_sum[j] += float(s)
                counts[j] += 1
            for j, cnt in counts.items():
                aux_t[i, j] = succ_sum[j] / cnt
                aux_mask[i, j] = 1.0
        return aux_t, aux_mask

    def _leave_self_out_batch(self, batch: list):
        """Per-example inputs/targets for aux_mode="leave_self_out".

        For each (goal, window, success) example in `batch`, strips ALL
        of that example's own goal's entries out of its window before
        encoding — under the default aux_mode="window", the window fed
        to the encoder already contains the target goal's own recent
        occurrences, so the aux head can satisfy the loss by reading
        those straight back out of ks instead of learning to infer a
        goal's competence from OTHER goals (the whole point of a shared
        knowledge state). Target is still the goal's
        mean success rate within its (unmasked) window; has_signal is 0
        for examples whose goal never appeared in their own window (no
        ground truth to supervise against).

        Returns (masked_windows, target (B,), has_signal (B,)) — caller
        pads masked_windows via _pad_windows and encodes it separately
        from the main kst.
        """
        masked_windows, targets, has_signal = [], [], []
        for goal, window, _ in batch:
            own = [s for g, s in window if g == goal]
            masked_windows.append([(g, s) for g, s in window if g != goal])
            if own:
                targets.append(sum(own) / len(own))
                has_signal.append(1.0)
            else:
                targets.append(0.5)
                has_signal.append(0.0)
        return (masked_windows,
                torch.tensor(targets, dtype=torch.float32, device=self.device),
                torch.tensor(has_signal, dtype=torch.float32, device=self.device))

    def _augmented_views(self, windows: list):
        """Two independent random subsamples per window (without
        replacement, keeping ~cl_subsample_frac of the real items) — the
        two "views" that form a positive pair for the contrastive loss.
        Empty/tiny windows just keep whatever they have.

        idx is sorted before indexing so each view keeps the surviving
        items in their original chronological order — a no-op for
        pool_mode="mean"/"attention" (order-invariant either way) but
        matters for pool_mode="lstm": without it, two views of the same
        window could look artificially dissimilar to an order-sensitive
        encoder purely from shuffled order, not from actual content
        differences, which would fight the contrastive objective."""
        def _subsample(w):
            if not w:
                return w
            k = max(1, int(round(len(w) * self.cl_subsample_frac)))
            idx = np.sort(np.random.choice(len(w), size=k, replace=False))
            return [w[i] for i in idx]

        view1 = [_subsample(w) for w in windows]
        view2 = [_subsample(w) for w in windows]
        return view1, view2

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Standard NT-Xent / InfoNCE (SimCLR-style), applied directly to
        ks (no separate projection head). z1[i]/z2[i] are the positive
        pair; every other vector in the batch is a negative for i."""
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z  = torch.cat([z1, z2], dim=0)

        sim = (z @ z.T) / self.cl_temperature
        sim.fill_diagonal_(float("-inf"))

        targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, targets)

    @torch.no_grad()
    def _current_ks(self) -> torch.Tensor:
        recent = self._recent_outcomes_tensor()
        return self.current_mlp.knowledge_state(recent).detach().clone()

    def update(self, goal, success: bool):
        # Snapshot BEFORE appending this episode: the train target is
        # (goal, success) conditioned on the prior window.
        recent_snapshot = self._window_for(goal)
        self.train_buffer.append((goal, recent_snapshot, float(success)))
        self.recent_window.append((goal, float(success)))

        self._ep_count += 1
        if self._ep_count % self.train_every == 0:
            self._train_mlp()
            self.ks_buffer.append(self._current_ks().cpu().numpy())

    def _stratified_indices(self, batch_size: int) -> np.ndarray:
        """Single importance-weighted draw over the WHOLE train_buffer
        (no replacement), counteracting goal-frequency imbalance instead
        of one plain uniform draw over buffer index — see balance_alpha's
        docstring in __init__ for the alpha continuum (0 = full
        equalization, 1 = no correction, identical to
        balanced_training=False).

        Per-entry weight = 1 / count(its goal)^(1-balance_alpha), so each
        GOAL's total selection probability scales as
        count(goal)^balance_alpha across the whole buffer.

        A single weighted draw over the full buffer, rather than a
        per-goal-quota-then-truncate scheme (draw ~batch_size//G per
        goal G present, then idx[:batch_size]), matters once the number
        of distinct goals G exceeds batch_size — which happens quickly
        in practice (eps_min's floor keeps feeding new goals into the
        buffer throughout training): a quota-then-truncate approach
        would keep only whichever ~batch_size goals were discovered
        EARLIEST and permanently exclude every goal discovered
        afterward from ever contributing a gradient again, the opposite
        of what balanced_training is meant to do. A single weighted
        draw has no such cliff: every present goal has a well-defined,
        nonzero selection probability regardless of how many distinct
        goals are present relative to batch_size."""
        n = len(self.train_buffer)
        if n == 0:
            return np.array([], dtype=int)
        counts = defaultdict(int)
        for b in self.train_buffer:
            counts[b[0]] += 1
        weights = np.array([1.0 / (counts[b[0]] ** (1.0 - self.balance_alpha))
                            for b in self.train_buffer])
        probs = weights / weights.sum()
        size = min(batch_size, n)
        return np.random.choice(n, size=size, replace=False, p=probs)

    def _train_mlp(self):
        n = len(self.train_buffer)
        if n < 2:
            return
        batch_size = min(self.train_batch, n)

        self.current_mlp.train()
        losses, aux_losses, cl_losses = [], [], []
        for _ in range(self.train_steps):
            if self.balanced_training:
                idx = self._stratified_indices(batch_size)
            else:
                idx = np.random.choice(n, size=batch_size, replace=False)
            batch = [self.train_buffer[i] for i in idx]

            gt  = torch.stack([self._goal_t[b[0]] for b in batch])
            yt  = torch.tensor([b[2] for b in batch],
                               dtype=torch.float32, device=self.device)

            self_mask = None
            if not self.main_leave_self_out:
                main_windows = [b[1] for b in batch]
            elif self.main_leave_self_out_mode == "mask":
                main_windows = self._mask_own_goal(batch)
            elif self.main_leave_self_out_mode == "mask_embed":
                main_windows = [b[1] for b in batch]  # raw pairs untouched
                self_mask = self._own_goal_positions(batch)
            else:
                main_windows = self._strip_own_goal(batch)
            items, mask = self._pad_windows(main_windows)
            kst = self.current_mlp.knowledge_encoder.batch_forward(
                items, mask, self_mask=self_mask)

            self.opt.zero_grad()
            x = torch.cat([gt, kst], dim=-1)
            preds = self.current_mlp.net(x).squeeze(-1)
            main_loss = self.bce(preds, yt)
            total_loss = main_loss

            # ── Auxiliary recovery loss ──────────────────────────────
            aux_loss_val = 0.0
            if self.aux_loss_weight > 0 and self.aux_mode == "window":
                # Decode ks -> per-goal success rates, but only for goals
                # that actually appear in each window (sparse masked
                # BCE). NOTE: kst here already saw each goal's own
                # occurrences, so this can be satisfied by reading them
                # straight back out — see aux_mode="leave_self_out" for
                # the variant that rules that shortcut out.
                aux_t, aux_mask = self._window_goal_success_targets(
                    [b[1] for b in batch])
                if aux_mask.sum() > 0:
                    aux_pred = torch.sigmoid(self.aux_head(kst))
                    bce = F.binary_cross_entropy(aux_pred, aux_t, reduction="none")
                    aux_loss = (bce * aux_mask).sum() / aux_mask.sum()
                    total_loss = total_loss + self.aux_loss_weight * aux_loss
                    aux_loss_val = aux_loss.item()
            elif self.aux_loss_weight > 0 and self.aux_mode == "leave_self_out":
                # Each example's own goal is stripped from its window
                # before encoding a SEPARATE kst for this loss, and only
                # that one goal is supervised — forces genuine
                # cross-goal inference (see aux_mode's docstring).
                lso_windows, lso_target, lso_has_signal = self._leave_self_out_batch(batch)
                if lso_has_signal.sum() > 0:
                    lso_items, lso_mask = self._pad_windows(lso_windows)
                    lso_kst = self.current_mlp.knowledge_encoder.batch_forward(
                        lso_items, lso_mask)
                    aux_pred_full = torch.sigmoid(self.aux_head(lso_kst))
                    goal_idx = torch.tensor(
                        [self._goal_idx[b[0]] for b in batch], device=self.device)
                    aux_pred = aux_pred_full.gather(1, goal_idx.unsqueeze(1)).squeeze(1)
                    bce = F.binary_cross_entropy(aux_pred, lso_target, reduction="none")
                    aux_loss = (bce * lso_has_signal).sum() / lso_has_signal.sum()
                    total_loss = total_loss + self.aux_loss_weight * aux_loss
                    aux_loss_val = aux_loss.item()

            # ── Contrastive (NT-Xent) loss — only pay the extra encoder
            # calls if it's actually turned on.
            cl_loss_val = 0.0
            if self.cl_weight > 0:
                view1, view2 = self._augmented_views([b[1] for b in batch])
                items1, mask1 = self._pad_windows(view1)
                items2, mask2 = self._pad_windows(view2)
                z1 = self.current_mlp.knowledge_encoder.batch_forward(items1, mask1)
                z2 = self.current_mlp.knowledge_encoder.batch_forward(items2, mask2)
                cl_loss = self._nt_xent(z1, z2)
                total_loss = total_loss + self.cl_weight * cl_loss
                cl_loss_val = cl_loss.item()

            total_loss.backward()
            self.opt.step()
            losses.append(main_loss.item())
            aux_losses.append(aux_loss_val)
            cl_losses.append(cl_loss_val)

        self.current_mlp.eval()
        self._last_loss     = float(np.mean(losses))
        self._last_aux_loss = float(np.mean(aux_losses)) if self.aux_loss_weight > 0 else None
        self._last_cl_loss  = float(np.mean(cl_losses)) if self.cl_weight > 0 else None

    def get_last_loss(self):
        return self._last_loss

    def get_last_aux_loss(self):
        return self._last_aux_loss

    def get_last_cl_loss(self):
        return self._last_cl_loss

    @torch.no_grad()
    def competence(self, goal) -> float:
        recent = self._pairs_to_items(self._window_for(goal))
        gvec = self._goal_t[goal].unsqueeze(0)
        return float(self.current_mlp(gvec, recent).item())

    def _ks_trajectory(self) -> torch.Tensor:
        """ks_buffer snapshots + live ks_now → (S, ks_dim)."""
        ks_now = self._current_ks()
        ks_list = [torch.from_numpy(k).to(self.device) for k in self.ks_buffer]
        ks_list.append(ks_now)
        return torch.stack(ks_list, dim=0)

    @staticmethod
    def _ema_along_traj(comps: np.ndarray, alpha: float) -> np.ndarray:
        """EMA along axis 0: out[t] = α·c[t] + (1-α)·out[t-1]."""
        out = np.empty_like(comps, dtype=np.float64)
        out[0] = comps[0]
        one_m = 1.0 - alpha
        for t in range(1, comps.shape[0]):
            out[t] = alpha * comps[t] + one_m * out[t - 1]
        return out

    def _alp_from_comps(self, comps: np.ndarray) -> np.ndarray:
        """Reduce competence trajectory (S, G_or_) → ALP along the S axis.

        comps[-1] is always C(· | ks_now); comps[:-1] are buffer snapshots.
        Mode-dependent reduction (see alp_mode).
        """
        if comps.ndim == 1:
            comps = comps[:, None]
            squeeze = True
        else:
            squeeze = False

        comps = comps.astype(np.float64, copy=False)

        if self.alp_mode == "omni_ks":
            alpha = self.alp_ema if self.alp_ema > 0.0 else 0.1
            if comps.shape[0] < 2:
                alp = np.zeros(comps.shape[1], dtype=np.float64)
            else:
                recent = self._ema_along_traj(comps, alpha)
                gradual = self._ema_along_traj(recent, alpha)
                alp = np.abs(
                    _omni_reweight(recent[-1], self.alp_p_theta)
                    - _omni_reweight(gradual[-1], self.alp_p_theta))
        else:
            if self.alp_ema > 0.0 and comps.shape[0] >= 2:
                comps = self._ema_along_traj(comps, self.alp_ema)

            if comps.shape[0] < 2:
                alp = (np.abs(comps[-1] - comps[0])
                       if comps.shape[0] == 1 else np.zeros(comps.shape[1]))
            elif self.alp_mode == "endpoints":
                alp = np.abs(comps[-1] - comps[0])
            elif self.alp_mode == "to_now":
                alp = np.mean(np.abs(comps[-1] - comps[:-1]), axis=0)
            else:
                alp = np.mean(np.abs(np.diff(comps, axis=0)), axis=0)

        alp = np.asarray(alp, dtype=np.float64)
        return alp.squeeze() if squeeze else alp

    @torch.no_grad()
    def _prospective_scores(self) -> torch.Tensor:
        """(G,) tensor, one score per goal in self.valid_goals order —
        see alp_mode="prospective_self"/"prospective_global"'s docstring
        for the full mechanism. Computed fresh every call (same cadence
        as the retrospective modes: ks_now genuinely
        changes every episode via recent_window, so there's no valid
        point in this trajectory to cache against).

        "prospective_global" measures the mean competence shift over a
        RANDOM SUBSAMPLE of goals (size self.prospective_subsample), not
        every goal — the full O(G^2) grid
        took 5.8s/call at G=1599 on CPU, which is infeasible every
        episode (and outright impossible at G=100,000, where G^2 is 10
        billion). Subsampling the g' axis decouples cost from G
        (O(G x subsample) instead of O(G^2)) at the price of a noisier
        estimate of the "global landscape shift" — a fresh subsample is
        drawn every call, so the noise averages out across the many
        calls made over training rather than biasing toward one fixed
        subset. "prospective_self" has no such cost problem (no g' loop
        at all, ~as cheap as the retrospective modes) and is unaffected.

        IMPORTANT — abs() placement differs by mode, and is NOT
        interchangeable:

        "prospective_self" (default): takes abs() of EACH outcome's own
        shift (|comp_self_1 - comp_now|, |comp_self_0 - comp_now|)
        BEFORE the p_g-weighted combination. Neither comp_self_0/1 come
        from an actual gradient step -- current_mlp.net's weights are
        100% frozen throughout this method; only the INPUT ks differs
        (re-encoded from a window with one hypothetical (g, outcome)
        entry appended). So "comp_future" here means "what the frozen
        predictor outputs given one more hypothetical observation",
        not "after real training". Since the outcome-1 branch is
        weighted by p_g = comp_now itself, the SIGNED weighted
        combination p_g*(comp1-comp_now) + (1-p_g)*(comp0-comp_now)
        collapses to ~E[comp_future] - comp_now =~ 0 for every goal
        (a martingale property: a self-consistent predictor's
        expectation of its own future belief, under its own current
        belief about the outcome, equals its current belief) --
        confirmed empirically (alp_mode="prospective_self_signed" test).
        abs() per outcome is what
        recovers a usable signal: the EXPECTED MAGNITUDE of belief
        flip, which is large for uncertain/actively-changing goals and
        small for mastered/untouched ones -- the correct one-step
        analogue of classical |current - past| ALP.

        "prospective_global" measures transfer to UNRELATED goals g',
        where no such martingale identity holds (nothing forces
        E[comp(g'|ks_after)] = comp(g'|ks_now) for g' != g), so a real
        signed transfer effect is possible and worth keeping signed. The
        SIGNED delta is averaged over g' first (cancelling zero-mean
        noise via the law of large numbers), THEN the final
        p_g-weighted combination is clamped at 0 -- NOT abs()'d per g'
        before averaging. abs()'ing per g' first would, by Jensen's
        inequality E[|X|] >= |E[X]|, leave a strictly positive noise
        floor (of order the estimation noise's std) for every goal g
        alike whenever the effect on a given g' is near-zero-mean
        noise, flattening any real distance/learnability structure
        (confirmed empirically: a flat, undifferentiated-by-distance
        score, 8/8 seed collapse, when tried that way). Clamping at 0
        rather than abs()'ing the final combination matters too, since
        get_sampling_probs() needs non-negative scores and a goal with
        net-negative expected global effect should get zero priority,
        not an inflated one from its magnitude."""
        enc = self.current_mlp.knowledge_encoder
        gt_all = torch.stack([self._goal_t[g] for g in self.valid_goals])  # (G, goal_dim)
        G = gt_all.shape[0]

        ks_now = self._current_ks()  # (knowledge_dim,)
        comp_now_all = self.current_mlp.net(
            torch.cat([gt_all, ks_now.unsqueeze(0).expand(G, -1)], dim=-1)).squeeze(-1)  # (G,)
        p_all = comp_now_all

        # hypothetical next-step window per candidate goal: drop the
        # oldest entry, append (g, outcome) — for BOTH outcome in {0,1},
        # never a continuous value (see docstring: stays in-distribution).
        tail = self._pairs_to_items(list(self.recent_window)[1:])
        tail_b = tail.unsqueeze(0).expand(G, -1, -1)
        zeros_col = torch.zeros(G, 1, device=self.device)
        ones_col = torch.ones(G, 1, device=self.device)
        new_item0 = torch.cat([gt_all, zeros_col], dim=-1).unsqueeze(1)
        new_item1 = torch.cat([gt_all, ones_col], dim=-1).unsqueeze(1)
        items0 = torch.cat([tail_b, new_item0], dim=1)  # (G, K, item_dim)
        items1 = torch.cat([tail_b, new_item1], dim=1)
        mask1 = torch.ones(G, items0.shape[1], device=self.device)

        ks1_0 = enc.batch_forward(items0, mask1)  # (G, knowledge_dim)
        ks1_1 = enc.batch_forward(items1, mask1)

        if self.alp_mode in ("prospective_self", "prospective_self_signed"):
            comp_self_0 = self.current_mlp.net(torch.cat([gt_all, ks1_0], dim=-1)).squeeze(-1)
            comp_self_1 = self.current_mlp.net(torch.cat([gt_all, ks1_1], dim=-1)).squeeze(-1)
            if self.alp_mode == "prospective_self":
                score0 = (comp_self_0 - comp_now_all).abs()
                score1 = (comp_self_1 - comp_now_all).abs()
            else:  # "prospective_self_signed" — diagnostic only, see
                # docstring: expected to collapse to ~0 everywhere by
                # the martingale property, kept to confirm that
                # empirically rather than just by argument.
                score0 = comp_self_0 - comp_now_all
                score1 = comp_self_1 - comp_now_all
            combined = p_all * score1 + (1 - p_all) * score0
            # clamp, not abs: _alp_scores_to_probs divides raw scores by
            # their sum with no clipping, so unclamped negative values
            # here would crash np.random.choice(p=...) the moment the
            # signed sum goes negative for some goal.
            return combined if self.alp_mode == "prospective_self" else combined.clamp(min=0.0)
        else:  # "prospective_global"
            # random subsample of the g' axis — see docstring: the full
            # O(G^2) grid is infeasible every episode (5.8s/call at
            # G=1599 on CPU, outright impossible at G=100,000).
            M = min(self.prospective_subsample, G)
            sub_idx = torch.randperm(G, device=self.device)[:M]
            gt_sub = gt_all[sub_idx]              # (M, goal_dim)
            comp_now_sub = comp_now_all[sub_idx]  # (M,)
            gt_b = gt_sub.unsqueeze(0).expand(G, -1, -1).reshape(G * M, -1)
            ks1_0_b = ks1_0.unsqueeze(1).expand(-1, M, -1).reshape(G * M, -1)
            ks1_1_b = ks1_1.unsqueeze(1).expand(-1, M, -1).reshape(G * M, -1)
            comp_grid_0 = self.current_mlp.net(
                torch.cat([gt_b, ks1_0_b], dim=-1)).squeeze(-1).reshape(G, M)
            comp_grid_1 = self.current_mlp.net(
                torch.cat([gt_b, ks1_1_b], dim=-1)).squeeze(-1).reshape(G, M)
            # signed delta averaged over g' FIRST (cancels zero-mean
            # noise), then clamp the combined expectation at 0 (not
            # abs()) — see docstring for why.
            score0 = (comp_grid_0 - comp_now_sub.unsqueeze(0)).mean(dim=1)
            score1 = (comp_grid_1 - comp_now_sub.unsqueeze(0)).mean(dim=1)
            return (p_all * score1 + (1 - p_all) * score0).clamp(min=0.0)

    @torch.no_grad()
    def get_alp(self, goal) -> float:
        if self.alp_mode in ("prospective_self", "prospective_self_signed",
                              "prospective_global"):
            scores = self._prospective_scores()
            idx = self.valid_goals.index(goal)
            return float(scores[idx])
        ks_stack = self._ks_trajectory()                 # (S, ks_dim)
        gvec = self._goal_t[goal].unsqueeze(0).expand(ks_stack.shape[0], -1)
        comps = self.current_mlp.net(
            torch.cat([gvec, ks_stack], dim=-1)).squeeze(-1).cpu().numpy()
        return float(self._alp_from_comps(comps))

    @torch.no_grad()
    def _competence_grid_over_ks(self) -> np.ndarray:
        """C(g|ks) for every goal x every position in the live ks_buffer
        (oldest trained snapshot first, live ks_now last, unreduced).
        Shape (S, G) — the grid get_alp_all() reduces over S internally."""
        ks_stack = self._ks_trajectory()                 # (S, ks_dim)
        gt = torch.stack([self._goal_t[g] for g in self.valid_goals])  # (G, d)
        S, G = ks_stack.shape[0], gt.shape[0]
        gt_b = gt.unsqueeze(0).expand(S, -1, -1).reshape(S * G, -1)
        ks_b = ks_stack.unsqueeze(1).expand(-1, G, -1).reshape(S * G, -1)
        comps = self.current_mlp.net(
            torch.cat([gt_b, ks_b], dim=-1)).squeeze(-1).reshape(S, G)
        return comps.cpu().numpy()

    def get_alp_all(self) -> dict:
        if self.alp_mode in ("prospective_self", "prospective_self_signed",
                              "prospective_global"):
            scores = self._prospective_scores()
            return {g: float(s) for g, s in zip(self.valid_goals, scores)}
        comps = self._competence_grid_over_ks()
        alp = self._alp_from_comps(comps)
        return {g: float(a) for g, a in zip(self.valid_goals, alp)}

    def get_competence_by_ks_index(self) -> np.ndarray:
        """Same (S, G) grid get_alp_all() reduces internally, exposed
        unreduced — competence of every goal at every position in the
        buffer (index 0 = oldest trained snapshot, index -1 = live ks_now),
        for logging/plotting instead of the single ALP scalar per goal."""
        return self._competence_grid_over_ks()

    def get_alp_all_groups(self) -> dict:
        alp_dict = self.get_alp_all()
        group_alp = {}
        for g, alp in alp_dict.items():
            group_alp.setdefault(self.group_fn(g), []).append(alp)
        return {r: float(np.mean(v)) for r, v in group_alp.items()}

    @torch.no_grad()
    def get_competence_all(self) -> dict:
        recent = self._recent_outcomes_tensor()
        gt    = torch.stack([self._goal_t[g] for g in self.valid_goals])
        comps = self.current_mlp(gt, recent).cpu().numpy()
        return {g: float(c) for g, c in zip(self.valid_goals, comps)}

    @torch.no_grad()
    def get_knowledge_state(self) -> np.ndarray:
        return self._current_ks().cpu().numpy()

    def get_raw_window_snapshot(self) -> list:
        """Materialize (goal_vec..., success) for logging — same dump
        shape as before the index-storage optimization."""
        out = []
        for g, s in self.recent_window:
            out.append((*self._goal_t[g].detach().cpu().tolist(), float(s)))
        return out

    def _alp_score_array(self) -> np.ndarray:
        alp_dict = self.get_alp_all()
        return np.array([alp_dict[g] for g in self.valid_goals], dtype=np.float64)

    def sample_goal(self):
        if random.random() < self.epsilon:
            return sample_uniform(self.valid_goals)
        candidates = _draw_candidates(self.valid_goals, self.candidate_subsample)
        if candidates is self.valid_goals:
            scores = self._alp_score_array()
        else:
            saved, self.valid_goals = self.valid_goals, candidates
            try:
                scores = self._alp_score_array()
            finally:
                self.valid_goals = saved
        return _sample_by_weight(candidates, scores, 0.0, deadzone=self.alp_deadzone)

    def anneal_epsilon(self, episode: int, epsilon_period: int, eps_min: float = None):
        eps_min = eps_min if eps_min is not None else self.eps_min
        self.epsilon = eps_min + (1.0 - eps_min) * np.exp(-episode / max(epsilon_period, 1))

    def get_sampling_probs(self) -> dict:
        scores = self._alp_score_array()
        probs = _alp_scores_to_probs(scores, deadzone=self.alp_deadzone)
        n = len(self.valid_goals)
        mixed = (1 - self.epsilon) * probs + self.epsilon / n
        return {g: float(p) for g, p in zip(self.valid_goals, mixed)}

    def stats(self) -> dict:
        alp_dict = self.get_alp_all()
        alp_vals = list(alp_dict.values())
        recent = self._recent_outcomes_tensor()
        with torch.no_grad():
            gt    = torch.stack([self._goal_t[g] for g in self.valid_goals])
            comps = self.current_mlp(gt, recent).cpu().numpy()
        return {"alp_mean":        float(np.mean(alp_vals)),
                "alp_max":         float(np.max(alp_vals)),
                "competence_mean": float(comps.mean()),
                "competence_max":  float(comps.max()),
                "train_buffer_size": len(self.train_buffer),
                "recent_window":   len(self.recent_window),
                "n_ks_snapshots":  len(self.ks_buffer),
                "epsilon":         self.epsilon,
                "alp_mode":        self.alp_mode}


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

CURRICULUM_REGISTRY = {
    "uniform":     Uniform,
    "online":      OnlineALP,
    "magellan":    MagellanALP,
    "magellan_ks": MagellanKnowledgeState,
}

_MLP_CURRICULA = {"magellan", "magellan_ks"}


def build_curriculum(name: str, valid_goals: list,
                     goal_vectors: dict = None, goal_dim: int = None,
                     group_fn=None, **kwargs):
    """
    Instantiate the correct curriculum.

    goal_vectors/goal_dim are required for the MLP-based curricula
    (magellan, magellan_ks) — not needed for uniform/online, which never
    build a neural net over goals.
    group_fn is always optional (default: identity — per-goal groups).
    Extra kwargs (M, eps_min, device, …) are forwarded to the constructor.
    """
    name = name.lower()
    if name not in CURRICULUM_REGISTRY:
        raise ValueError(f"Unknown curriculum '{name}'. "
                         f"Choose from {list(CURRICULUM_REGISTRY)}")
    cls = CURRICULUM_REGISTRY[name]
    if name == "uniform":
        return cls(valid_goals)
    if name in _MLP_CURRICULA:
        assert goal_vectors is not None and goal_dim is not None, \
            f"'{name}' needs goal_vectors and goal_dim"
        return cls(valid_goals, goal_vectors=goal_vectors, goal_dim=goal_dim,
                  group_fn=group_fn, **kwargs)
    return cls(valid_goals, group_fn=group_fn, **kwargs)   # online