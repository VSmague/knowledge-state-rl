"""
crafter_game/architecture.py — model pieces for Crafter, matching
minigrid_game/architecture.py's exact contract:
    Encoder(obs_batch) -> latent
    GoalRepr(goal_batch) -> vec
    GoalConditionedAC(latent, goal_vec) -> logits, value
    Agent.act(obs_batch, goal_batch) -> action, logp, entropy, value
    Agent.forward_value(obs_batch, goal_batch) -> value
    Agent.evaluate_actions(obs_batch, goal_batch, actions) -> logp, entropy, value
    build_agent(size, device, ...) -> Agent

Naming follows the shared convention used across every game module:
GoalRepr for the goal -> conditioning vector role (even though the
underlying mechanism differs a lot between games — learned/frozen
embedding here vs. a fixed coordinate normalization for minigrid_game),
and an Agent class bundling the 3-method interface core/'s generic PPO
loop calls regardless of which game it's training on: act (rollout
collection), forward_value (GAE bootstrap without wastefully sampling an
unused action), evaluate_actions (PPO update step, scoring already-taken
actions under the current policy).

GoalRepr.forward() accepts a list of achievement name strings, a
LongTensor of indices, or a mix — converts internally via GOAL_TO_IDX —
so a caller (a rollout buffer, e.g.) never has to know the internal
index encoding.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from crafter_game.env import N_ACTIONS, ACHIEVEMENTS


GOAL_TO_IDX = {g: i for i, g in enumerate(ACHIEVEMENTS)}


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """
    SB3-style NatureCNN (DQN/Atari architecture) for 64×64×3 obs.
    Same stack as stable_baselines3.common.torch_layers.NatureCNN:
    Conv 8×8/4 → Conv 4×4/2 → Conv 3×3/1 → flatten → Linear → features_dim.
    ~600k params.
    """

    def __init__(self, features_dim: int = 512):
        super().__init__()
        self.enc_dim = features_dim
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, 3, 64, 64)
            n_flatten = self.cnn(sample).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0
        return self.fc(self.cnn(x))


class GoalRepr(nn.Module):
    """
    Two modes:
      - Default (frozen_embeddings_path=None): a standard nn.Embedding,
        randomly initialized, trained end-to-end with everything else.
      - frozen_embeddings_path given: loads a precomputed (n_goals,
        goal_dim) array (e.g. from goal_llm_embed.py's Qwen embeddings)
        and uses it AS-IS, permanently frozen — stored via
        register_buffer, not nn.Parameter, so it's structurally
        impossible for it to end up in an optimizer's param list by
        accident (stronger guarantee than requires_grad=False, which is
        easy to bypass by mistake — same reasoning as minigrid_game/
        architecture.py's GoalRepr.scale buffer).

    forward() accepts a list of achievement name strings, a LongTensor of
    indices, or a numpy array of indices — converts internally.
    """

    def __init__(self, n_goals: int = 22, goal_dim: int = 16,
                frozen_embeddings_path: str = None):
        super().__init__()
        if frozen_embeddings_path is not None:
            weights = np.load(frozen_embeddings_path)
            assert weights.shape[0] == n_goals, (
                f"{frozen_embeddings_path} has {weights.shape[0]} rows, "
                f"expected {n_goals} (one per goal, same order as "
                f"crafter_game.env.ACHIEVEMENTS)")
            self.register_buffer("frozen_weights", torch.from_numpy(weights).float())
            self.goal_dim = weights.shape[1]
            self.frozen = True
            self.embed = None
        else:
            self.embed = nn.Embedding(n_goals, goal_dim)
            self.goal_dim = goal_dim
            self.frozen = False

    def _to_idx(self, goal_batch) -> torch.Tensor:
        device = self.frozen_weights.device if self.frozen else self.embed.weight.device
        if torch.is_tensor(goal_batch):
            return goal_batch.to(device).long()
        if len(goal_batch) > 0 and isinstance(goal_batch[0], str):
            return torch.tensor([GOAL_TO_IDX[g] for g in goal_batch],
                                dtype=torch.long, device=device)
        return torch.as_tensor(np.asarray(goal_batch), dtype=torch.long, device=device)

    def forward(self, goal_batch) -> torch.Tensor:
        idx = self._to_idx(goal_batch)
        if self.frozen:
            return self.frozen_weights[idx]
        return self.embed(idx)


class GoalConditionedAC(nn.Module):
    """separate_ac=False (default): a single shared trunk, split only at
    the final linear layer into actor/critic heads.

    separate_ac=True: fully independent actor and critic trunks, no
    shared parameters at all past Encoder/GoalRepr — the convention many
    actor-critic PPO implementations use (e.g. OMNI's). Rationale: with a
    shared trunk, pg_loss and v_loss both backprop through the SAME trunk
    parameters, so any imbalance between them (e.g. from entropy_coef or
    the reward/return scale) doesn't just compete in the scalar loss
    sum, it directly interferes at the gradient level on those shared
    weights. Opt-in rather than the default, since the shared-trunk
    behavior is still possibly better for some configurations."""

    def __init__(self, enc_dim: int, goal_dim: int, action_dim: int = N_ACTIONS,
                hidden_dim: int = 1024, separate_ac: bool = False):
        super().__init__()
        self.separate_ac = separate_ac
        in_dim = enc_dim + goal_dim
        if separate_ac:
            self.actor_trunk = nn.Sequential(
                layer_init(nn.Linear(in_dim, hidden_dim)), nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU())
            self.critic_trunk = nn.Sequential(
                layer_init(nn.Linear(in_dim, hidden_dim)), nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU())
        else:
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(in_dim, hidden_dim)), nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU())
        self.actor  = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1),          std=1.0)

    def forward(self, latent, goal_vec):
        x = torch.cat([latent, goal_vec], dim=-1)
        if self.separate_ac:
            logits = self.actor(self.actor_trunk(x))
            value  = self.critic(self.critic_trunk(x)).squeeze(-1)
        else:
            h = self.trunk(x)
            logits, value = self.actor(h), self.critic(h).squeeze(-1)
        return logits, value

    def act(self, latent, goal_vec):
        logits, val = self(latent, goal_vec)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), val


# ---------------------------------------------------------------------------
# Recurrent variant (LSTM) — separate from Agent/GoalConditionedAC above,
# not a flag on them, because recurrence isn't just an architecture
# choice: the caller (core/train_rnn.py, not core/train.py) has to own
# and thread hidden state through every rollout step and reset it
# per-env on episode boundaries, which needs its own training loop, not
# just its own network. Goal is injected INTO the LSTM's input
# (concat(image_embed, goal_vec) -> LSTMCell), matching OMNI's own
# model.py, rather than concatenated after the encoder like
# Agent/GoalConditionedAC above — so the hidden state itself can carry
# goal-conditioned "where am I in my multi-step plan" information across
# steps, which a memoryless feedforward policy can't represent except
# through whatever's directly visible in the current frame + inventory
# HUD. Verified with a real short run (see core/train_rnn.py's module
# docstring) — still only a smoke test at small scale, not evidence
# about sample efficiency or learning quality at a real training budget.
# ---------------------------------------------------------------------------

class RecurrentGoalConditionedAC(nn.Module):
    """Actor/critic heads operating on the LSTM's hidden state (not on
    encoder+goal directly — the LSTM in RecurrentAgent already consumed
    the goal as part of its input).

    separate_ac=False (default — same default as GoalConditionedAC's own
    flag, and controlled by the same --crafter_separate_ac CLI flag for
    both agents, so it means one consistent thing): a single shared
    2-layer trunk, split only at the final linear layer.
    separate_ac=True: independent 2-layer MLP heads, matching OMNI's
    model.py exactly. Neither choice has real evidence behind it yet for
    the recurrent case specifically — kept as a flag rather than picked;
    pass --crafter_separate_ac to reproduce OMNI's actual choice."""

    def __init__(self, hidden_size: int, action_dim: int = N_ACTIONS,
                head_dim: int = 128, separate_ac: bool = False):
        super().__init__()
        self.separate_ac = separate_ac
        if separate_ac:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(hidden_size, head_dim)), nn.ReLU(),
                layer_init(nn.Linear(head_dim, action_dim), std=0.01))
            self.critic = nn.Sequential(
                layer_init(nn.Linear(hidden_size, head_dim)), nn.ReLU(),
                layer_init(nn.Linear(head_dim, 1), std=1.0))
        else:
            self.trunk = nn.Sequential(
                layer_init(nn.Linear(hidden_size, head_dim)), nn.ReLU())
            self.actor  = layer_init(nn.Linear(head_dim, action_dim), std=0.01)
            self.critic = layer_init(nn.Linear(head_dim, 1),          std=1.0)

    def forward(self, h):
        if self.separate_ac:
            return self.actor(h), self.critic(h).squeeze(-1)
        x = self.trunk(h)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, h):
        logits, val = self(h)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), val


class RecurrentAgent:
    """OMNI-style recurrent agent. NOT the same 3-method contract as
    Agent above — act/forward_value/evaluate_actions all ALSO take a
    `hidden` = (h, c) tuple (each (B, hidden_size)) and return the
    updated hidden state alongside their usual outputs, since the
    caller must persist and reset it across rollout steps itself (see
    module-level comment above). init_hidden(batch_size) gives a
    zeroed starting state — call it fresh per env whenever that env's
    episode resets."""

    def __init__(self, enc_dim: int = 512, goal_dim: int = 16,
                hidden_size: int = 256, head_dim: int = 128,
                frozen_embeddings_path: str = None, device: str = "cpu",
                separate_ac: bool = False):
        self.encoder = Encoder(features_dim=enc_dim).to(device)
        self.goal_repr = GoalRepr(n_goals=len(ACHIEVEMENTS), goal_dim=goal_dim,
                                  frozen_embeddings_path=frozen_embeddings_path).to(device)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(enc_dim + self.goal_repr.goal_dim, hidden_size).to(device)
        self.ac = RecurrentGoalConditionedAC(hidden_size, N_ACTIONS, head_dim,
                                             separate_ac=separate_ac).to(device)
        self.device = device

    def init_hidden(self, batch_size: int):
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h, c

    def parameters(self):
        return (list(self.encoder.parameters()) +
                list(self.goal_repr.parameters()) +
                list(self.lstm.parameters()) +
                list(self.ac.parameters()))

    def _step(self, obs_batch, goal_batch, hidden):
        lat = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        x = torch.cat([lat, gvec], dim=-1)
        h, c = self.lstm(x, hidden)
        return h, (h, c)

    def act(self, obs_batch, goal_batch, hidden):
        h, new_hidden = self._step(obs_batch, goal_batch, hidden)
        a, logp, ent, val = self.ac.act(h)
        return a, logp, ent, val, new_hidden

    def forward_value(self, obs_batch, goal_batch, hidden):
        h, new_hidden = self._step(obs_batch, goal_batch, hidden)
        _, value = self.ac(h)
        return value, new_hidden

    def evaluate_actions(self, obs_batch, goal_batch, actions, hidden):
        h, new_hidden = self._step(obs_batch, goal_batch, hidden)
        logits, value = self.ac(h)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, new_hidden

    # -- Sequence-batched path — PPO-update-only speedup ---------------------
    # encoder/goal_repr don't depend on `hidden` at all (only self.lstm
    # does) — so for a call site that already has the WHOLE T-length
    # sequence available (core/train_rnn.py's ppo_update_recurrent,
    # replaying the fixed rollout buffer), there's no need to run the
    # CNN + goal_repr T separate times, once per small per-timestep
    # minibatch, inside that function's necessarily-sequential loop over
    # the LSTM. One big batched forward pass over all T*B frames does the
    # exact same computation (a reordering, not an approximation) far
    # more GPU-efficiently, leaving the
    # sequential loop to do only the cheap LSTM+heads step per timestep.
    # NOT used by act/forward_value/evaluate_actions above: rollout
    # collection and evaluate_all_recurrent only ever have ONE timestep's
    # observation available at a time (the env hasn't stepped further
    # yet), so there's no sequence to batch ahead of there.

    def encode_sequence(self, obs_seq, goal_seq):
        """obs_seq: (T, B, *obs_shape) tensor. goal_seq: (T, B) — anything
        goal_repr's _to_idx can flatten (e.g. a numpy object array of
        goal strings), in the SAME T-major-then-B order as
        obs_seq.reshape(T*B, ...), so the two align after flattening.
        Returns (T, B, enc_dim+goal_dim)."""
        T, B = obs_seq.shape[0], obs_seq.shape[1]
        obs_flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        lat_flat = self.encoder(obs_flat)
        goal_flat = np.asarray(goal_seq).reshape(-1).tolist()
        gvec_flat = self.goal_repr(goal_flat)
        x_flat = torch.cat([lat_flat, gvec_flat], dim=-1)
        return x_flat.reshape(T, B, -1)

    def evaluate_actions_from_features(self, x, actions, hidden):
        """Same job as evaluate_actions, but `x` is already the
        precomputed (encoder+goal_repr) feature vector for ONE timestep
        (a (B, enc_dim+goal_dim) slice of encode_sequence's output), not
        raw (obs, goal) — only the LSTM + heads run here."""
        h, c = self.lstm(x, hidden)
        logits, value = self.ac(h)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, (h, c)


def build_recurrent_agent(size: tuple = None, device: str = "cpu", **kwargs) -> RecurrentAgent:
    """`size` accepted (unused), same interface-parity reasoning as
    build_agent above."""
    return RecurrentAgent(device=device, **kwargs)


class Agent:
    """Bundles Encoder + GoalRepr + GoalConditionedAC. Same 3-method
    interface as every other game's Agent (act / forward_value /
    evaluate_actions — act = rollout collection, forward_value = GAE
    bootstrap, evaluate_actions = PPO update step). The PPO algorithm
    itself (GAE, clipping, epochs) is NOT here; it's game-agnostic and
    belongs in core/."""

    def __init__(self, enc_dim: int = 512, goal_dim: int = 16,
                hidden_dim: int = 1024,
                frozen_embeddings_path: str = None, device: str = "cpu",
                separate_ac: bool = False):
        self.encoder = Encoder(features_dim=enc_dim).to(device)
        self.goal_repr = GoalRepr(n_goals=len(ACHIEVEMENTS), goal_dim=goal_dim,
                                  frozen_embeddings_path=frozen_embeddings_path).to(device)
        self.ac = GoalConditionedAC(enc_dim, self.goal_repr.goal_dim,
                                    N_ACTIONS, hidden_dim, separate_ac=separate_ac).to(device)
        self.device = device

    def parameters(self):
        return (list(self.encoder.parameters()) +
                list(self.goal_repr.parameters()) +   # empty list if frozen — see GoalRepr docstring
                list(self.ac.parameters()))

    def act(self, obs_batch, goal_batch):
        lat = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        return self.ac.act(lat, gvec)

    def forward_value(self, obs_batch, goal_batch):
        lat = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        _, value = self.ac(lat, gvec)
        return value

    def evaluate_actions(self, obs_batch, goal_batch, actions):
        lat = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        logits, value = self.ac(lat, gvec)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def build_agent(size: tuple = None, device: str = "cpu", **kwargs) -> Agent:
    """`size` accepted (unused) for interface parity with every other
    game's build_agent(size, device) — core/'s generic training loop
    can call build_agent(size, device=...) uniformly regardless of
    game."""
    return Agent(device=device, **kwargs)


def describe_agent_hparams(args) -> str:
    """One-line description of agent hparams, printed by core/train.py at
    startup (crafter_game only — no other registered game module defines
    this hook, so core/train.py just doesn't print this line for them)."""
    return f"enc_dim={args.enc_dim}  hidden_dim={args.hidden_dim}"


def extra_agent_kwargs(args, frozen_embeddings_path) -> dict:
    """Extra build_agent() kwargs derived from crafter_game-only CLI args
    — folded into core/train.py's agent_kwargs via hasattr dispatch
    instead of an `args.game == "crafter_game"` check."""
    kwargs = dict(separate_ac=args.crafter_separate_ac)
    if frozen_embeddings_path:
        kwargs["frozen_embeddings_path"] = frozen_embeddings_path
    return kwargs