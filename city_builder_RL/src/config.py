# city_builder_rl/config.py
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    # ---------------- Core training ----------------
    NUM_EPISODES: int = 2
    EVAL_EPISODES: int = 2
    MAX_STEPS_PER_EPISODE: int = 15

    GAMMA: float = 1.0
    LR: float = 3e-4
    ENTROPY_BONUS: float = 1e-3
    VALUE_COEF: float = 0.5

    # ---------------- Exploration temperature ----------------
    USE_TEMPERATURE: bool = True
    TAU: float = 0.6
    TAU_MIN: float = 0.1
    TAU_MAX: float = 2.0
    KAPPA_PARETO: float = 0.5  # scaling for end-of-episode pareto_distance

    # ---------------- Features ----------------
    EPS: float = 1e-9                  # numerical eps for feature norms
    MAX_ITEMS_HINT_FLOOR: int = 64     # fallback when remaining is empty at t0

    # ---------------- Device / Seeding ----------------
    DEVICE: str = "cpu"                # "cpu" or "cuda"
    SEED: int = 12345

    # ---------------- Checkpointing ----------------
    CKPT_DIR: str = "./checkpoints"
    CKPT_NAME: str = "actor_critic_citybuilder.pt"
    LOAD: bool = False                 # load checkpoint if exists

    # ---------------- Local env (no API) ----------------
    ENV_CFG_PATH: Optional[str] = None       # path to YAML; None -> use package default
    CFG_FINGERPRINT: str = "v1"
    PARETO_WINDOW: int = 256                 # if you later want rolling stats in client


def parse_args_to_config() -> Tuple[Config, str]:
    """
    Parse CLI, return (cfg, mode) where mode in {"train", "eval"}.
    This replaces parsing in main.py to avoid duplication.
    """
    import argparse
    p = argparse.ArgumentParser(description="City Builder RL (local, no REST)")

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument("--device", default=Config.DEVICE, choices=["cpu", "cuda"])
        sp.add_argument("--seed", type=int, default=Config.SEED)
        sp.add_argument("--tau", type=float, default=Config.TAU)
        sp.add_argument("--gamma", type=float, default=Config.GAMMA)
        sp.add_argument("--lr", type=float, default=Config.LR)
        sp.add_argument("--entropy", type=float, default=Config.ENTROPY_BONUS)
        sp.add_argument("--value-coef", type=float, default=Config.VALUE_COEF)
        sp.add_argument("--max-steps", type=int, default=Config.MAX_STEPS_PER_EPISODE)
        sp.add_argument("--ckpt-dir", default=Config.CKPT_DIR)
        sp.add_argument("--ckpt-name", default=Config.CKPT_NAME)
        sp.add_argument("--cfg", dest="env_cfg_path", default=None, help="Path to environment YAML")
        sp.add_argument("--fingerprint", default=Config.CFG_FINGERPRINT)
        sp.add_argument("--load", action="store_true", help="Load checkpoint if exists")
        # temperature controls
        sp.add_argument("--use-temp", action="store_true", default=Config.USE_TEMPERATURE)
        sp.add_argument("--tau-min", type=float, default=Config.TAU_MIN)
        sp.add_argument("--tau-max", type=float, default=Config.TAU_MAX)
        sp.add_argument("--kappa-pareto", type=float, default=Config.KAPPA_PARETO)
        # feature controls
        sp.add_argument("--eps", type=float, default=Config.EPS)
        sp.add_argument("--items-hint-floor", type=int, default=Config.MAX_ITEMS_HINT_FLOOR)

    sp_train = sub.add_parser("train")
    add_shared(sp_train)
    sp_train.add_argument("--episodes", type=int, default=Config.NUM_EPISODES)

    sp_eval = sub.add_parser("eval")
    add_shared(sp_eval)
    sp_eval.add_argument("--episodes", type=int, default=Config.EVAL_EPISODES)

    args = p.parse_args()

    mode = args.cmd  # "train" or "eval"
    cfg = Config(
        DEVICE=args.device,
        SEED=int(args.seed),
        TAU=float(args.tau),
        GAMMA=float(args.gamma),
        LR=float(args.lr),
        ENTROPY_BONUS=float(args.entropy),
        VALUE_COEF=float(args.value_coef),
        MAX_STEPS_PER_EPISODE=int(args.max_steps),
        CKPT_DIR=args.ckpt_dir,
        CKPT_NAME=args.ckpt_name,
        LOAD=bool(args.load),
        ENV_CFG_PATH=args.env_cfg_path,
        CFG_FINGERPRINT=args.fingerprint,
        USE_TEMPERATURE=bool(args.use_temp),
        TAU_MIN=float(args.tau_min),
        TAU_MAX=float(args.tau_max),
        KAPPA_PARETO=float(args.kappa_pareto),
        EPS=float(args.eps),
        MAX_ITEMS_HINT_FLOOR=int(args.items_hint_floor),
    )

    if mode == "train":
        cfg.NUM_EPISODES = int(args.episodes)
    else:
        cfg.EVAL_EPISODES = int(args.episodes)

    return cfg, mode
