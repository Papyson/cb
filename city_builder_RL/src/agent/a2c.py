from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from torch.distributions import Categorical

from ..config import Config
from ..features import FeatureExtractor
from ..models.nets import ActorCriticNet

class A2CAgent:
    def __init__(self, cfg: Config, model: ActorCriticNet, feat: FeatureExtractor):
        self.cfg = cfg
        self.model = model.to(cfg.DEVICE)
        self.feat = feat
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.LR)

    def _build_tensors(self, obs: dict):
        s = self.feat.state_features(obs)               # [S]
        X, ids, mask = self.feat.item_features(obs)     # [M,X], list, [M]
        return s.to(self.cfg.DEVICE), X.to(self.cfg.DEVICE), mask.to(self.cfg.DEVICE), ids

    def act(self, obs: dict) -> Dict[str, Any]:
        state_vec, item_mat, mask, ids = self._build_tensors(obs)
        if item_mat.shape[0] == 0 or mask.sum().item() == 0:
            return {"ids": ids, "action": None, "terminal_noop": True}

        tau = self.cfg.TAU if self.cfg.USE_TEMPERATURE else None
        logits, value = self.model(state_vec, item_mat, mask, tau=tau)
        dist = Categorical(logits=logits)
        a_idx_t = dist.sample()
        logp_t = dist.log_prob(a_idx_t)
        probs = dist.probs

        action_id = ids[int(a_idx_t.item())]
        return {
            "ids": ids,
            "action": action_id,
            "a_idx_t": a_idx_t,
            "logits": logits.detach(),
            "probs": probs.detach(),
            "logp_t": logp_t,
            "value_t": value,
            "state_vec": state_vec,
            "item_mat": item_mat,
            "mask": mask
        }

    def compute_loss(self, *, logp_t: torch.Tensor, value_t: torch.Tensor,
                     reward_t: float, value_next: torch.Tensor, entropy_t: torch.Tensor):
        r_t = torch.tensor(reward_t, dtype=torch.float32, device=value_t.device)
        with torch.no_grad():
            target = r_t + self.cfg.GAMMA * value_next
        advantage = target - value_t

        policy_loss = -(logp_t * advantage)
        value_loss = 0.5 * (target - value_t).pow(2)
        loss = policy_loss + self.cfg.VALUE_COEF * value_loss - self.cfg.ENTROPY_BONUS * entropy_t
        return loss, policy_loss.detach().item(), value_loss.detach().item(), advantage.detach().item()

    def update(self, total_loss: torch.Tensor):
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

    def adjust_temperature_from_pareto(self, tau: float, pareto_distance: float) -> float:
        if not self.cfg.USE_TEMPERATURE:
            return tau
        new_tau = tau * (1.0 + self.cfg.KAPPA_PARETO * float(pareto_distance))
        new_tau = float(np.clip(new_tau, self.cfg.TAU_MIN, self.cfg.TAU_MAX))
        return new_tau
