# city_builder_rl/training/loops.py
from typing import List, Any, Dict
import numpy as np
import torch
from torch.distributions import Categorical

from ..config import Config
from ..env_client import EnvClient
from ..features import FeatureExtractor
from ..agent.a2c import A2CAgent
from ..models.nets import ActorCriticNet
from ..utils.checkpoints import save_checkpoint


def _parse_reward(rew_field: Any) -> float:
    if isinstance(rew_field, dict):
        return float(rew_field.get("final", 0.0))
    return float(rew_field)


def _safe_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits robustly; if all logits are -inf (no valid actions),
    return 0.0 entropy.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    if not torch.isfinite(logits).any():
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    dist = Categorical(logits=logits)
    ent = dist.entropy()
    return ent.mean() if ent.ndim > 0 else ent


def train_loop(cfg: Config, env: EnvClient, agent: A2CAgent, model: ActorCriticNet, feat: FeatureExtractor) -> Dict[str, List[float]]:
    returns_history: List[float] = []
    paretoD_history: List[float] = []
    entropy_history: List[float] = []
    tau_history: List[float] = []

    env.init()
    env.seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)

    for ep in range(cfg.NUM_EPISODES):
        resp = env.reset()
        obs = resp["observation"]

        # per-episode N0 for remaining_frac
        feat.set_episode_pool_size(len(obs.get("remaining", [])))

        done = False
        steps = 0
        ep_return = 0.0
        ep_entropies: List[float] = []

        while not done and steps < cfg.MAX_STEPS_PER_EPISODE:
            # Agent produces tensors/logits; ensure device alignment when used here
            act_info = agent.act(obs)
            if act_info.get("terminal_noop"):
                break

            logits_t = act_info["logits"].to(device)
            entropy_t = _safe_entropy_from_logits(logits_t)
            ep_entropies.append(entropy_t.item())

            # step
            action_id = act_info["action"]
            resp = env.step(action_id)
            rew = _parse_reward(resp["reward"])
            obs_next = resp["observation"]
            done = bool(resp.get("done", False))

            # bootstrap
            with torch.no_grad():
                s_next, X_next, m_next, _ = agent._build_tensors(obs_next)
                s_next = s_next.to(device)
                X_next = X_next.to(device)
                m_next = m_next.to(device)

                if X_next.shape[0] == 0 or m_next.sum().item() == 0:
                    value_next = torch.tensor(0.0, dtype=torch.float32, device=device)
                else:
                    logits_next, value_next = model(
                        s_next, X_next, m_next,
                        tau=cfg.TAU if cfg.USE_TEMPERATURE else None
                    )

            loss, pi_loss, v_loss, adv_val = agent.compute_loss(
                logp_t=act_info["logp_t"],
                value_t=act_info["value_t"],
                reward_t=rew,
                value_next=value_next,
                entropy_t=entropy_t
            )
            agent.update(loss)

            obs = resp["observation"]
            steps += 1
            ep_return += rew

        # episode end: Pareto distance -> temperature modulation
        try:
            summary = env.summary()
            D = float(summary.get("pareto_distance", 0.0))
        except Exception:
            D = 0.0
        cfg.TAU = agent.adjust_temperature_from_pareto(cfg.TAU, D)

        returns_history.append(ep_return)
        paretoD_history.append(D)
        tau_history.append(cfg.TAU)
        if ep_entropies:
            entropy_history.append(np.mean(ep_entropies))
        else:
            entropy_history.append(0.0)

        print(f"[Train] Ep {ep+1:03d}/{cfg.NUM_EPISODES} steps={steps:03d} "
              f"ret={ep_return:7.3f} ParetoD={D:5.3f} tau={cfg.TAU:4.2f}")

        # periodic checkpointing
        if (ep + 1) % 50 == 0 or (ep + 1) == cfg.NUM_EPISODES:
            save_checkpoint(cfg, model, agent)

    return {
        "returns": returns_history,
        "paretoD": paretoD_history,
        "entropy": entropy_history,
        "tau": tau_history,
    }


def eval_loop(cfg: Config, env: EnvClient, agent: A2CAgent, model: ActorCriticNet, feat: FeatureExtractor):
    totals: List[float] = []
    lengths: List[int] = []

    env.seed(cfg.SEED + 999)

    device = torch.device(cfg.DEVICE)

    for _ in range(cfg.EVAL_EPISODES):
        resp = env.reset()
        obs = resp["observation"]

        feat.set_episode_pool_size(len(obs.get("remaining", [])))

        done = False
        ep_return = 0.0
        steps = 0

        prev_tau = cfg.TAU
        if cfg.USE_TEMPERATURE:
            cfg.TAU = max(cfg.TAU_MIN, cfg.TAU * 0.8)

        while not done and steps < cfg.MAX_STEPS_PER_EPISODE:
            act_info = agent.act(obs)
            if act_info.get("terminal_noop"):
                break
            resp = env.step(act_info["action"])
            rew = _parse_reward(resp["reward"])
            obs = resp["observation"]
            done = bool(resp.get("done", False))
            ep_return += rew
            steps += 1

        if cfg.USE_TEMPERATURE:
            cfg.TAU = prev_tau

        totals.append(ep_return)
        lengths.append(steps)

    return {"avg_return": float(np.mean(totals)), "avg_len": float(np.mean(lengths))}
