import os
import torch
from ..config import Config
from ..agent.a2c import A2CAgent
from ..models.nets import ActorCriticNet

def save_checkpoint(cfg: Config, model: ActorCriticNet, agent: A2CAgent):
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    path = os.path.join(cfg.CKPT_DIR, cfg.CKPT_NAME)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": agent.opt.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(ckpt, path)
    print(f"[CKPT] Saved checkpoint to {path}")

def load_checkpoint(path: str, model: ActorCriticNet, agent: A2CAgent):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    agent.opt.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[CKPT] Loaded checkpoint from {path}")
