from typing import List, Optional
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(activation())
        layers.pop()  # remove final activation
        self.net = nn.Sequential(*layers)

        # Orthogonal init (good with ReLU); zero biases
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ActorCriticNet(nn.Module):
    """
    Two-tower A2C policy:
      - State tower:  s -> s_emb (state_dim -> 64 -> 64)
      - Item tower:   x_i -> e_i (item_dim -> 64 -> 64)
      - Policy head:  [e_i || s_emb] -> logit_i (128 -> 64 -> 1)
      - Value head:   s_emb -> V(s) (64 -> 64 -> 1)

    Init:
      - Hidden layers: orthogonal(ReLU gain)
      - Final policy layer: tiny gain (~0.01) for near-uniform initial logits
      - Final value layer: moderate gain (~1.0)
    """

    def __init__(self, state_dim: int, item_dim: int):
        super().__init__()
        self.state_tower = MLP(state_dim, [64, 64], 64)
        self.item_tower  = MLP(item_dim,  [64, 64], 64)
        self.policy_head = MLP(64 + 64,   [64],     1)
        self.value_head  = MLP(64,        [64],     1)
        self._init_final_layers()

    def _init_final_layers(self) -> None:
        # Last Linear layers of policy/value heads
        policy_last = self.policy_head.net[-1]
        value_last  = self.value_head.net[-1]
        assert isinstance(policy_last, nn.Linear) and isinstance(value_last, nn.Linear)

        nn.init.orthogonal_(policy_last.weight, gain=0.01)
        nn.init.zeros_(policy_last.bias)

        nn.init.orthogonal_(value_last.weight, gain=1.0)
        nn.init.zeros_(value_last.bias)

    def forward(
        self,
        state_vec: torch.Tensor,          # [S]
        item_mat: torch.Tensor,           # [M, X]
        mask: torch.Tensor,               # [M] in {0,1}
        tau: Optional[float] = None
    ):
        s_emb = self.state_tower(state_vec)                   # [64]
        if item_mat.ndim == 1:
            item_mat = item_mat.unsqueeze(0)
        e_mat = self.item_tower(item_mat)                     # [M,64]

        s_rep = s_emb.unsqueeze(0).expand(e_mat.size(0), -1)  # [M,64]
        fused = torch.cat([e_mat, s_rep], dim=-1)             # [M,128]

        logits = self.policy_head(fused).squeeze(-1)          # [M]
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask > 0.5, logits, torch.full_like(logits, neg_inf))
        if tau is not None and tau > 0:
            logits = logits / tau

        value = self.value_head(s_emb).squeeze(-1)            # scalar
        return logits, value