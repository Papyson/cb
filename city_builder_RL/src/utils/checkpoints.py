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
        "optimizer_state_dict": getattr(agent, "opt", None).state_dict() if hasattr(agent, "opt") else None,
        "config": getattr(cfg, "__dict__", {}),
    }
    torch.save(ckpt, path)
    print(f"[CKPT] Saved checkpoint to {path}")

def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    """Ensure optimizer's internal tensors live on the same device as the model."""
    if opt is None:
        return
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)

def load_checkpoint(path: str, model: ActorCriticNet, agent: A2CAgent):
    ckpt = torch.load(path, map_location="cpu")

    # Load model weights: try strict first; if mismatch, fall back to non-strict with warning.
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[CKPT] Loaded model (strict) from {path}")
    except Exception as e:
        print(f"[CKPT] Warning: strict load failed ({e}). Falling back to non-strict.")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[CKPT] Loaded model (non-strict) from {path}")

    # Load optimizer state if present and agent has an optimizer
    opt_state = ckpt.get("optimizer_state_dict", None)
    if opt_state is not None and hasattr(agent, "opt") and agent.opt is not None:
        try:
            agent.opt.load_state_dict(opt_state)
            # Move optimizer states to the same device as the model parameters
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
            _move_optimizer_state_to_device(agent.opt, model_device)
            print(f"[CKPT] Loaded optimizer state from {path} (moved to device: {model_device})")
        except Exception as e:
            print(f"[CKPT] Warning: could not load optimizer state ({e}). "
                  f"Training will continue with a fresh optimizer.")

    # Optional: you can access cfg snapshot via ckpt.get("config") if needed
    print(f"[CKPT] Loaded checkpoint from {path}")
