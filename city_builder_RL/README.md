# CityBuilder RL (Actor–Critic)

PyTorch Actor–Critic agent for the City Builder multi-objective knapsack game over HTTP.
The environment (C# server) exposes: `/init`, `/seed`, `/reset`, `/step`, `/episode/{id}/summary`.

## Quick Start

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Ensure your City Builder server is running (e.g. http://localhost:5000)

# 3) Train
python -m src.main train --base-url http://localhost:5000 --episodes 200

# 4) Evaluate
python -m src.main eval --base-url http://localhost:5000 --episodes 20 --load
```

> Tip: run `python -m src.main --help` to see all flags.

## Requirements

* Python 3.9+
* `torch`, `numpy`, `requests`, `matplotlib`, `tqdm` (see `requirements.txt`)

## Repo Layout

```
citybuilder-rl/
├─ README.md
├─ requirements.txt
├─ checkpoints/                # saved models
├─ src/
│  ├─ main.py                  # CLI (train/eval)
│  ├─ config.py                # config + arg parsing
│  ├─ env_client.py            # HTTP calls: /init /seed /reset /step /summary
│  ├─ features.py              # Feature extractor (per-episode N0 scaling)
│  ├─ models/nets.py           # ActorCriticNet + MLP
│  ├─ agent/a2c.py             # A2C agent logic
│  └─ training/loops.py        # train_loop, eval_loop
└─ docs/
   └─ INSTRUCTIONS.md          # full detailed guide 
```

## How It Works 

* **State tower** summarizes budget + remaining-pool stats + advisor breadth.
* **Item tower** encodes each candidate item (cost norm, per-objective normalized value, advisor support, value-per-cost).
* Masked softmax over per-item logits ⇒ **action probabilities**.
* **A2C** updates (policy + value + entropy) every step.
* At episode end, **Pareto distance** can modulate sampling temperature `tau`.

**Per-episode pool size**: we set `N0 = len(remaining)` at reset, and use
`remaining_frac = |R_t| / N0` for stable, episode-local scaling.

**Reproducibility**: call `/seed` once before training, and again before eval (with a different seed).

## Outputs

* Checkpoints in `./checkpoints/`
* Training curve saved as `returns.png`
* Console logs: episode steps, return, Pareto distance, and `tau`.

## Troubleshooting

* Connection errors → check `--base-url` and that the server is running.
* Unstable learning → try `--lr 1e-4` or increase `--entropy 5e-3`.
* CPU vs GPU → `--device cpu|cuda`.

## More Details

See **`docs/INSTRUCTIONS.md`** for deeper explanations (features, architecture, seeding, temperature modulation, etc.).

---

