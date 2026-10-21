"""
fetch_game/architecture.py — same 3-method Agent contract as every
other game module (MLP encoder, no learned GoalRepr — fetch_game's goal_vector
is already a fixed, structured numeric encoding, so there's nothing for
GoalRepr to learn here, unlike minigrid_game/crafter_game where it at
least normalizes or embeds raw goal data).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from fetch_game.env import N_ACTIONS, OBS_DIM, GOAL_DIM, goal_vector as _goal_vector


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    def __init__(self, enc_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, enc_dim)), nn.ReLU(),
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return self.net(x.float())


class GoalRepr(nn.Module):
    """No learned parameters — fetch_game's goal_vector is already a
    structured numeric encoding, so this is a pass-through to device/
    dtype only."""

    def __init__(self, size: tuple = None):
        super().__init__()
        self.goal_dim = GOAL_DIM
        self.register_buffer("_device_anchor", torch.zeros(1))

    def forward(self, goal_batch) -> torch.Tensor:
        if torch.is_tensor(goal_batch):
            return goal_batch.to(self._device_anchor.device).float()
        vecs = np.stack([_goal_vector(g) for g in goal_batch])
        return torch.as_tensor(vecs, dtype=torch.float32, device=self._device_anchor.device)


class GoalConditionedAC(nn.Module):
    def __init__(self, enc_dim: int, goal_dim: int = GOAL_DIM, action_dim: int = N_ACTIONS,
                hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(enc_dim + goal_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU())
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
    def __init__(self, size: tuple = None, enc_dim: int = 128, hidden_dim: int = 256,
                device: str = "cpu", **_):
        self.encoder   = Encoder(enc_dim, hidden_dim).to(device)
        self.goal_repr = GoalRepr(size).to(device)
        self.ac = GoalConditionedAC(enc_dim, self.goal_repr.goal_dim,
                                    N_ACTIONS, hidden_dim).to(device)
        self.device = device

    def parameters(self):
        return (list(self.encoder.parameters()) +
               list(self.goal_repr.parameters()) +
               list(self.ac.parameters()))

    def act(self, obs_batch, goal_batch):
        lat  = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        return self.ac.act(lat, gvec)

    def forward_value(self, obs_batch, goal_batch):
        lat  = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        _, value = self.ac(lat, gvec)
        return value

    def evaluate_actions(self, obs_batch, goal_batch, actions):
        lat  = self.encoder(obs_batch)
        gvec = self.goal_repr(goal_batch)
        logits, value = self.ac(lat, gvec)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def build_agent(size: tuple = None, device: str = "cpu", **kwargs) -> Agent:
    return Agent(size, device=device, **kwargs)
