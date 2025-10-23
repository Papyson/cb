# city_builder_rl/main.py
from __future__ import annotations

import torch

from .config import parse_args_to_config, Config
from .env_client import EnvClient
from .features import FeatureExtractor
from .models.nets import ActorCriticNet
from .agent.a2c import A2CAgent
from .training.loops import train_loop, eval_loop


def build_feature_extractor_from_obs(obs: dict, cfg: Config) -> FeatureExtractor:
    remaining = obs.get("remaining", [])
    if len(remaining) > 0 and "v" in remaining[0]:
        K = len(remaining[0]["v"])
    else:
        # fallback: infer from config if provided elsewhere
        K = 4

    initial_budget = float(obs.get("initial_budget", obs.get("budget", 1.0)))
    max_items_hint = int(len(remaining)) if len(remaining) > 0 else int(cfg.MAX_ITEMS_HINT_FLOOR)

    feat = FeatureExtractor(
        eps=float(cfg.EPS),
        initial_budget=initial_budget,
        num_objectives=int(K),
        max_items_hint=int(max_items_hint),
    )
    feat.set_episode_pool_size(max_items_hint)
    return feat


def infer_dims_from_obs(feat: FeatureExtractor, obs: dict, cfg: Config) -> tuple[int, int]:
    s_vec = feat.state_features(obs)
    state_dim = int(s_vec.numel())

    X, ids, mask = feat.item_features(obs)
    if X.ndim == 2 and X.shape[0] > 0:
        item_dim = int(X.shape[1])
    else:
        item_dim = 1  # safe fallback when no items are remaining at t0

    return state_dim, item_dim


def main():
    # Parse CLI into a single Config + mode (train/eval)
    cfg, mode = parse_args_to_config()

    # Torch/seed
    torch.manual_seed(int(cfg.SEED))

    # Env (local)
    env = EnvClient(cfg)
    env.init()
    env.seed(int(cfg.SEED))

    first = env.reset()
    obs0 = first["observation"]

    # Features / dims
    feat = build_feature_extractor_from_obs(obs0, cfg)
    state_dim, item_dim = infer_dims_from_obs(feat, obs0, cfg)

    # Model/Agent
    model = ActorCriticNet(state_dim=state_dim, item_dim=item_dim)
    agent = A2CAgent(cfg=cfg, model=model, feat=feat)

    if mode == "train":
        _ = train_loop(cfg=cfg, env=env, agent=agent, model=model, feat=feat)
        # Optional: evaluate right after training
        stats = eval_loop(cfg=cfg, env=env, agent=agent, model=model, feat=feat)
        print(f"[Eval] avg_return={stats['avg_return']:.3f}  avg_len={stats['avg_len']:.2f}")
    else:
        stats = eval_loop(cfg=cfg, env=env, agent=agent, model=model, feat=feat)
        print(f"[Eval] avg_return={stats['avg_return']:.3f}  avg_len={stats['avg_len']:.2f}")


if __name__ == "__main__":
    main()
