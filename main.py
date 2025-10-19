# main.py
"""
CityBuilderEnv demo runner 

What it shows
-------------
    Build the env from YAML via the 'make_env' factory
    Run a few episodes with a tiny policy
    Log steps/summaries; optionally save a replay.

Usage
----------
python -m citybuilder_env.main --episodes 3 --seed 1234 --logdir ./logs \
        -- policy advisor-greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

import citybuilder_env as cbe 

# --------- Simple demo policies -------------

def policy_advisor_greedy(obs: dict, mask: np.ndarray) -> int:
    feasible = np.flatnonzero(mask)
    if feasible.size == 0:
        return 0
    phi = obs.get("advisor_freq")
    if phi is None:
        return int(np.random.choice(feasible))
    objs = obs.get("objs_int")
    if objs is None:
        return int(feasible[np.argmax(phi[feasible])])
    # sort by (advisor freq desc, sum objectives desc)
    tie = objs[feasible, :].sum(axis=1)
    idx = int(np.lexsort((-tie, -phi[feasible]))[0])
    return int(feasible[idx])

def policy_random(_: dict, mask: np.ndarray) -> int:
    feasible = np.flatnonzero(mask)
    if feasible.size == 0:
        return 0
    return int(np.random.choice(feasible))

# ------------ runner --------------

def run(args: argparse.Namespace) -> int:
    env = cbe.make_env(seed=args.seed, cfg_path=args.cfg_path, cfg_fingerprint=args.cfg_fp)
    run_dir =- cbe.make_run_dir(args.logdir, prefix="citybuilder")
    logger = cbe.EpisodeLogger(run_dir)

    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        obs, mask, info = env.reset()
        logger.start_episode(info)

        done = False
        step_idx = 0
        actions_for_replay = []

        while not done:
            action = (
                policy_advisor_greedy(obs, mask)
                if args.policy == "advisor-greedy"
                else policy_random(obs, mask)
            )
            step = env.step(action)

            logger.log_step({
                "episode_id": info["episode_id"],
                "step_idx": step_idx,
                "item_id": action,
                "reward": float(step.reward),
                "budget_after_int": step.info.get("budget_after_int"),
                "done": step.done,
                "info": step.info,
            })

            actions_for_replay.append(int(action))
            obs, mask, done = step.observation, step.action_mask, step.done
            step_idx += 1

        summary = env.episode_summary()
        logger.end_episode(summary)

        if args.save_replay:
            trace = cbe.ReplayTrace(
                format_version=1,
                master_seed=args.seed,
                cfg_fingerprint=args.cfg_fp,
                episode_index=ep,
                actions=list(actions_for_replay),
                meta={"policy": args.policy},
            )
            out = Path(run_dir) / "replays" / f"episode_{ep:04d}.json"
            out.parents.mkdir(parents=True, exist_ok=True)
            cbe.save_replay(out, trace)

    logger.close()
    print(f"[logs] run directory: {run_dir}")
    return 0

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CityBuilderEnv demo runner")
    p.add_argument("--episodes", type=int, default=3, help="NUmber of episodes to run")
    p.add_argument("--seed", type=int, default=1234, help="Master seed")
    p.add_argument("--cfg-fp", dest="cfg_fp", type=str, default="v1", help="Config fingerprint tag")
    p.add_argument("--cfg-path", type=str, default="./config/default.yaml", help="Path to YAML config")
    p.add_argument("--logdir", type=str, default="./logs", help="Directory for logs")
    p.add_argument("--policy", type=str, choices=["advisor-greedy", "random"], default="advisor-greedy")
    p.add_argument("--save-replay", action="store_true", help="Save a replay JSON per episode")
    return p.parse_args(argv)

if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))