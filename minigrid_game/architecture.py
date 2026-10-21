"""
minigrid_game/architecture.py — model pieces for the 2D room-grid game.
Same contract as crafter_game/architecture.py (and any future game):
    Encoder(obs_batch) -> latent
    GoalRepr(goal_batch) -> vec
    GoalConditionedAC(latent, goal_vec) -> logits, value
    Agent.act(obs_batch, goal_batch) -> action, logp, entropy, value
    Agent.forward_value(obs_batch, goal_batch) -> value
    Agent.evaluate_actions(obs_batch, goal_batch, actions) -> logp, entropy, value
    build_agent(size, device, ...) -> Agent

Agent (x, y, dx, dy) is ALWAYS part of the observation (see
minigrid_game/env.py's module docstring) -- so PosDirRepr here is not
optional/gated behind a flag, it's just always part of the Agent.

Encoder assumes a FIXED (7, 7, 3) observation (MiniGrid's default
agent_view_size=7, unchanged), hence the fixed 64*4*4 flatten below.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from minigrid_game.env import N_ACTIONS, grid_dims, POS_DIR_DIM

IMG_SHAPE = (7, 7, 3)
IMG_FLAT_DIM = int(np.prod(IMG_SHAPE))


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    def __init__(self, enc_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 2)), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(64 * 4 * 4, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, enc_dim)),    nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0
        h = self.conv(x)
        return self.fc(h.reshape(h.shape[0], -1))


class GoalRepr(nn.Module):
    """Normalizes raw (x, y) goal cells into [0, 1] — no learned
    parameters."""

    def __init__(self, size: tuple):
        super().__init__()
        grid_w, grid_h = grid_dims(size)
        self.register_buffer("scale", torch.tensor([grid_w - 1, grid_h - 1], dtype=torch.float32))
        self.goal_dim = 2

    def forward(self, goal_batch) -> torch.Tensor:
        if not torch.is_tensor(goal_batch):
            goal_batch = torch.as_tensor(np.asarray(goal_batch), dtype=torch.float32)
        goal_batch = goal_batch.to(self.scale.device)
        return goal_batch.float() / self.scale


class PosDirRepr(nn.Module):
    """Normalizes the agent's own (x, y, dx, dy) — no learned
    parameters. Always present here (see module docstring), not gated
    behind a flag."""

    def __init__(self, size: tuple):
        super().__init__()
        grid_w, grid_h = grid_dims(size)
        self.register_buffer(
            "scale", torch.tensor([grid_w - 1, grid_h - 1], dtype=torch.float32))
        self.out_dim = 4

    def forward(self, pos_dir_batch) -> torch.Tensor:
        pos_dir_batch = pos_dir_batch.to(self.scale.device).float()
        pos = pos_dir_batch[:, :2] / self.scale
        dir_vec = pos_dir_batch[:, 2:4] - 1.0
        return torch.cat([pos, dir_vec], dim=-1)


class GoalConditionedAC(nn.Module):
    def __init__(self, enc_dim: int, goal_dim: int = 2, action_dim: int = N_ACTIONS,
                hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(enc_dim + goal_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim,  hidden_dim)), nn.ReLU())
        self.actor  = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1),          std=1.0)

    def forward(self, latent, goal_vec):
        h = self.trunk(torch.cat([latent, goal_vec], dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, latent, goal_vec):
        logits, val = self(latent, goal_vec)
        dist = Categorical(logits=logits)
        a    = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), val


class Agent:
    def __init__(self, size: tuple, enc_dim: int = 128, hidden_dim: int = 256,
                device: str = "cpu", **_):
        self.encoder = Encoder(enc_dim, hidden_dim).to(device)
        self.goal_repr = GoalRepr(size).to(device)
        self.pos_dir_repr = PosDirRepr(size).to(device)
        ac_enc_dim = enc_dim + self.pos_dir_repr.out_dim
        self.ac = GoalConditionedAC(ac_enc_dim, self.goal_repr.goal_dim,
                                    N_ACTIONS, hidden_dim).to(device)
        self.device = device

    def parameters(self):
        return (list(self.encoder.parameters()) +
               list(self.goal_repr.parameters()) +
               list(self.pos_dir_repr.parameters()) +
               list(self.ac.parameters()))

    def _split_obs(self, obs_batch):
        """obs_batch is (B, IMG_FLAT_DIM + POS_DIR_DIM) — the flattened
        image plus the raw (x, y, dx+1, dy+1) values Env._pack_obs
        always appends here — split back into the (B, *IMG_SHAPE) image
        Encoder expects and the (B, POS_DIR_DIM) pos/dir PosDirRepr
        expects."""
        img = obs_batch[:, :IMG_FLAT_DIM].reshape(-1, *IMG_SHAPE)
        pos_dir = obs_batch[:, IMG_FLAT_DIM:IMG_FLAT_DIM + POS_DIR_DIM]
        return img, pos_dir

    def _latent(self, obs_batch):
        img, pos_dir = self._split_obs(obs_batch)
        lat = self.encoder(img)
        return torch.cat([lat, self.pos_dir_repr(pos_dir)], dim=-1)

    def act(self, obs_batch, goal_batch):
        lat = self._latent(obs_batch)
        gvec = self.goal_repr(goal_batch)
        return self.ac.act(lat, gvec)

    def forward_value(self, obs_batch, goal_batch):
        lat = self._latent(obs_batch)
        gvec = self.goal_repr(goal_batch)
        _, value = self.ac(lat, gvec)
        return value

    def evaluate_actions(self, obs_batch, goal_batch, actions):
        lat = self._latent(obs_batch)
        gvec = self.goal_repr(goal_batch)
        logits, value = self.ac(lat, gvec)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def build_agent(size: tuple, device: str = "cpu", **kwargs) -> Agent:
    return Agent(size, device=device, **kwargs)
