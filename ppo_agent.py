from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model.GNN import PhysicsGNN
from model.actor import CelestialActor
from model.critic import CentralizedCritic

OBS_DIM = 17
ACTION_DIM = 6
NUM_BODIES = 3
SPATIAL_DIM = 2


class RunningMeanStd:
    """Streaming mean/variance for observation normalization."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = float(total_count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = (x - self.mean) / np.sqrt(self.var + 1e-8)
        y = np.clip(y, -clip, clip)
        return y.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class OrbitalPpoSystem(nn.Module):
    """CTDE policy/value system that reconstructs graph features from flat observations."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        if obs_dim != OBS_DIM:
            raise ValueError(f"Expected obs_dim={OBS_DIM}, got {obs_dim}")
        if action_dim != ACTION_DIM:
            raise ValueError(f"Expected action_dim={ACTION_DIM}, got {action_dim}")

        self.gnn = PhysicsGNN(node_in_dim=7, edge_in_dim=3)
        self.actors = nn.ModuleList([CelestialActor(action_dim=2) for _ in range(NUM_BODIES)])
        self.critic = CentralizedCritic()

    def _unflatten_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.dim() != 2 or obs.size(-1) != OBS_DIM:
            raise ValueError(f"Expected obs shape [B, {OBS_DIM}], got {tuple(obs.shape)}")

        batch = obs.shape[0]

        pos = obs[:, 0:6].view(batch, NUM_BODIES, SPATIAL_DIM)
        vel = obs[:, 6:12].view(batch, NUM_BODIES, SPATIAL_DIM)
        mass = obs[:, 12:15].view(batch, NUM_BODIES, 1)
        phase = obs[:, 15:17].view(batch, 1, SPATIAL_DIM).expand(batch, NUM_BODIES, SPATIAL_DIM)
        nodes = torch.cat((pos, vel, mass, phase), dim=-1)

        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        rel = pos_j - pos_i
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        edges = torch.cat((rel, dist), dim=-1)

        return nodes, edges

    def forward(
        self,
        obs: torch.Tensor,
        structured: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes, edges = self._unflatten_obs(obs)
        embeddings = self.gnn(nodes, edges)

        mean_parts = []
        std_parts = []
        for i, actor in enumerate(self.actors):
            mean_i, std_i = actor(embeddings[:, i, :])
            std_i = torch.clamp(std_i.expand_as(mean_i), min=1e-4)
            mean_parts.append(mean_i)
            std_parts.append(std_i)

        mean = torch.stack(mean_parts, dim=1)
        std = torch.stack(std_parts, dim=1)
        value = self.critic(torch.flatten(embeddings, start_dim=1)).squeeze(-1)

        if structured:
            return mean, std, value

        return mean.flatten(start_dim=1), std.flatten(start_dim=1), value


ActorCritic = OrbitalPpoSystem


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_logp: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor


def gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(mean, std)
    return dist.log_prob(actions).flatten(start_dim=1).sum(dim=-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Normal(torch.zeros_like(std), std)
    return dist.entropy().flatten(start_dim=1).sum(dim=-1)


__all__ = [
    "ActorCritic",
    "OrbitalPpoSystem",
    "PPOBatch",
    "RunningMeanStd",
    "gaussian_log_prob",
    "gaussian_entropy",
]
