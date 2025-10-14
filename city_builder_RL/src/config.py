from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Server
    BASE_URL: str = "http://localhost:5000"
    API_TIMEOUT: float = 30.0
    RETRIES: int = 3
    RETRY_BACKOFF_SEC: float = 0.5

    # Training
    NUM_EPISODES: int = 200
    GAMMA: float = 1.0
    LR: float = 3e-4
    ENTROPY_BONUS: float = 1e-3
    VALUE_COEF: float = 0.5
    MAX_STEPS_PER_EPISODE: int = 500

    # Exploration temperature
    USE_TEMPERATURE: bool = True
    TAU: float = 0.6
    TAU_MIN: float = 0.1
    TAU_MAX: float = 2.0
    KAPPA_PARETO: float = 0.5

    # Features
    EPS: float = 1e-9
    MAX_ITEMS_HINT_FLOOR: int = 64

    # Device / Seeds
    DEVICE: str = "cpu"
    SEED: int = 42

    # Checkpoint
    CKPT_DIR: str = "./checkpoints"
    CKPT_NAME: str = "actor_critic_citybuilder.pt"

    # Evaluation
    EVAL_EPISODES: int = 20

def parse_args_to_config() -> Config:
    import argparse
    p = argparse.ArgumentParser(description="City Builder RL (Actor–Critic)")

    sub = p.add_subparsers(dest="cmd", required=True)
    # shared
    def add_shared(sp):
        sp.add_argument("--base-url", default=Config.BASE_URL)
        sp.add_argument("--device", default=Config.DEVICE, choices=["cpu","cuda"])
        sp.add_argument("--seed", type=int, default=Config.SEED)
        sp.add_argument("--tau", type=float, default=Config.TAU)
        sp.add_argument("--gamma", type=float, default=Config.GAMMA)
        sp.add_argument("--lr", type=float, default=Config.LR)
        sp.add_argument("--entropy", type=float, default=Config.ENTROPY_BONUS)
        sp.add_argument("--ckpt-dir", default=Config.CKPT_DIR)
        sp.add_argument("--ckpt-name", default=Config.CKPT_NAME)
        sp.add_argument("--episodes", type=int)  # set below per mode
        sp.add_argument("--load", action="store_true", help="Load checkpoint if exists")

    sp_train = sub.add_parser("train")
    add_shared(sp_train)
    sp_train.set_defaults(episodes=Config.NUM_EPISODES)

    sp_eval = sub.add_parser("eval")
    add_shared(sp_eval)
    sp_eval.set_defaults(episodes=Config.EVAL_EPISODES)

    args = p.parse_args()

    cfg = Config(
        BASE_URL=args.base_url,
        DEVICE=args.device,
        SEED=args.seed,
        TAU=args.tau,
        GAMMA=args.gamma,
        LR=args.lr,
        ENTROPY_BONUS=args.entropy,
        CKPT_DIR=args.ckpt_dir,
        CKPT_NAME=args.ckpt_name,
    )
    if args.cmd == "train":
        cfg.NUM_EPISODES = args.episodes
    else:
        cfg.EVAL_EPISODES = args.episodes

    return cfg, args.cmd, args.load
