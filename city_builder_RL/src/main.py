# city_builder_rl/main.py
from __future__ import annotations

import os
import torch
from pathlib import Path

from .config import parse_args_to_config, Config
from .env_client import EnvClient
from .features import FeatureExtractor
from .models.nets import ActorCriticNet
from .agent.a2c import A2CAgent
from .training.loops import train_loop, eval_loop
from .utils.checkpoints import load_checkpoint 
from .utils.plotting import plot_training_curves  

from citybuilder_env.io.logging import EpisodeLogger, make_run_dir  # type: ignore


# ----------------- Feature helpers -----------------
def build_feature_extractor_from_obs(obs: dict, cfg: Config) -> FeatureExtractor:
    remaining = obs.get("remaining", [])
    if len(remaining) > 0 and "v" in remaining[0]:
        K = len(remaining[0]["v"])
    else:
        K = 4
    initial_budget = float(obs.get("initial_budget", obs.get("budget", 1.0)))
    max_items_hint = int(len(remaining)) if len(remaining) > 0 else int(cfg.MAX_ITEMS_HINT_FLOOR)
    feat = FeatureExtractor(
        eps=float(cfg.EPS),
        initial_budget=initial_budget,
        num_objectives=int(K),
        max_items_hint=int(max_items_hint),
        use_advisors=bool(cfg.USE_ADVISORS),
    )
    feat.set_episode_pool_size(max_items_hint)
    return feat


def infer_dims_from_obs(feat: FeatureExtractor, obs: dict, cfg: Config) -> tuple[int, int]:
    # Ensure episode pool size is set for correct remaining_frac scaling
    remaining = obs.get("remaining", []) or []
    feat.set_episode_pool_size(len(remaining) if len(remaining) > 0 else int(cfg.MAX_ITEMS_HINT_FLOOR))
    # Rely on the extractor’s deterministic dim computation
    return feat.get_dims()


# ----------------- Utilities -----------------
def maybe_load_checkpoint_model_only(model: torch.nn.Module, ckpt_dir: str, ckpt_name: str, strict: bool = False) -> bool:
    """
    Eval-time loader: loads model weights only (no optimizer).
    """
    path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=strict)
    return True


def enable_grads(model: torch.nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


# ----------------- Main -----------------
def main():
    cfg, mode = parse_args_to_config()
    device = torch.device(cfg.DEVICE)

    # Seeding
    torch.manual_seed(int(cfg.SEED))

    # Env (mode-aware inside EnvClient)
    env = EnvClient(cfg, mode=mode)
    env.init()
    env.seed(int(cfg.SEED))

    # --- attach environment-native logger BEFORE first reset() ---
    logs_base = "./logs"
    run_dir = make_run_dir(logs_base, prefix=f"run-{mode}")
    logger = EpisodeLogger(run_dir)
    env.attach_logger(run_dir=str(run_dir), logger=logger)

    # First observation for feature sizing
    first = env.reset()
    obs0 = first["observation"]

    # Features / dims
    feat = build_feature_extractor_from_obs(obs0, cfg)
    state_dim, item_dim = infer_dims_from_obs(feat, obs0, cfg)
    print(f"[Init] USE_ADVISORS={cfg.USE_ADVISORS}  state_dim={state_dim}  item_dim={item_dim}  device={device}")

    # Model/Agent
    model = ActorCriticNet(state_dim=state_dim, item_dim=item_dim).to(device)
    agent = A2CAgent(cfg=cfg, model=model, feat=feat)  # agent also ensures device for tensors

    if mode == "train":
        # Optional resume: load model + optimizer
        if cfg.LOAD:
            path = os.path.join(cfg.CKPT_DIR, cfg.CKPT_NAME)
            if os.path.exists(path):
                load_checkpoint(path, model, agent)
                # ensure everything ends up on the target device
                model.to(device)
                # move optimizer state tensors to device
                for st in agent.opt.state.values():
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(device, non_blocking=True)
                print(f"[Train] resume checkpoint: loaded {path}")
            else:
                print(f"[Train] resume checkpoint: not found at {path}")

        model.train()
        enable_grads(model, True)

        # Run training and capture returns history
        history = train_loop(cfg=cfg, env=env, agent=agent, model=model, feat=feat)

        try:
            out_img = Path(run_dir) / "training_curve.png"
            plot_training_curves(
                returns_history=history["returns"],
                paretoD_history=history["paretoD"],
                entropy_history=history["entropy"],
                tau_history=history["tau"],
                out_path=str(out_img)
            )
        except Exception as e:
            print(f"[PLOT] Skipped plotting due to error: {e}")

    else:  # EVAL
        # Require a checkpoint: eval should be a pure rollout of a frozen model
        ok = maybe_load_checkpoint_model_only(model, cfg.CKPT_DIR, cfg.CKPT_NAME, strict=True)
        if not ok:
            raise FileNotFoundError(
                f"[Eval] checkpoint required but missing: {os.path.join(cfg.CKPT_DIR, cfg.CKPT_NAME)}"
            )

        model.to(device)
        model.eval()
        enable_grads(model, False)

        # If your agent supports an eval policy hook, prefer greedy inference
        if hasattr(agent, "set_eval_policy"):
            try:
                agent.set_eval_policy(greedy=True, temperature=None)
            except Exception:
                pass

        # Pure rollout: no grads, no optimizer, no checkpointing
        with torch.no_grad():
            stats = eval_loop(cfg=cfg, env=env, agent=agent, model=model, feat=feat)

        print(f"[Eval] avg_return={stats['avg_return']:.3f}  avg_len={stats['avg_len']:.2f}")
        # Optionally persist eval stats
        try:
            import json
            (Path(run_dir) / "eval_stats.json").write_text(json.dumps(stats, indent=2))
        except Exception:
            pass


if __name__ == "__main__":
    main()
