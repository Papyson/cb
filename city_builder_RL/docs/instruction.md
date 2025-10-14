hh
Here’s a detailed `docs/INSTRUCTIONS.md` you can drop into your repo:

---

# City Builder RL — Detailed Instructions

This document explains how to set up, train, evaluate, and extend the PyTorch Actor–Critic agent that plays the **City Builder** multi-objective 0–1 knapsack game via HTTP.

The **environment** (a C# server) exposes the following endpoints:

* `POST /init`
* `POST /seed`
* `POST /reset`
* `POST /step`
* `GET  /episode/{episode_id}/summary`

The client discovers all defaults from `/init`; **no client-side env constants** are hard-coded.

---

## 1 Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # torch, numpy, requests, matplotlib, tqdm
```

Ensure your City Builder server is running (e.g., `http://localhost:5000`).

---

## 2 Quick Usage

### Train

```bash
python -m src.main train \
  --base-url http://localhost:5000 \
  --episodes 200 \
  --device cpu
```

### Evaluate (load checkpoint)

```bash
python -m src.main eval \
  --base-url http://localhost:5000 \
  --episodes 20 \
  --device cpu \
  --load
```

> Run `python -m src.main --help` to see all CLI flags.

Outputs:

* Checkpoints in `./checkpoints/`
* Training curve as `returns.png`
* Console logs: episode steps, return, Pareto distance, and temperature `tau`

---

## 3 CLI Flags (most useful)

* `--base-url` (str): Env server URL. Default: `http://localhost:5000`
* `--episodes` (int): Number of episodes (train/eval)
* `--seed` (int): RNG seed (used via `/seed`)
* `--device` (`cpu|cuda`): Compute device
* `--tau` (float): Policy sampling temperature (softmax scaling)
* `--gamma` (float): Discount factor (default `1.0`)
* `--lr` (float): Learning rate (default `3e-4`)
* `--entropy` (float): Entropy bonus (exploration; default `1e-3`)
* `--ckpt-dir`, `--ckpt-name`: Checkpoint destination/name
* `--load`: Load an existing checkpoint before running

---

## 4 Environment Contract (HTTP)

### `/init` (POST)

* **Request:** `{}` (empty; server decides defaults)
* **Response:**

  ```json
  {
    "env_id": "abcd1234",
    "status": "initialized",
    "config": {
      "initial_budget": 51,
      "objectives": 3,
      "advisor_backend": "ilp"
    }
  }
  ```

### `/seed` (POST)

* **Request:** `{"env_id": "...", "seed": 42}`
* **Purpose:** Deterministic episode stream for reproducibility.

### `/reset` (POST)

* **Request:** `{"env_id": "..." }`
* **Response (initial observation):**

  ```json
  {
    "episode_id": "ep1",
    "observation": {
      "budget": 51,
      "knapsack": [],             // optional; not required by the client
      "remaining": [
        { "id": 1, "cost": 2, "v": [2,1,4] },
        { "id": 2, "cost": 5, "v": [5,4,7] }
      ],
      "advisor_sets": {
        "R1": [ ... ],
        "R2": [ ... ],
        "R3": [ ... ]
      }
    },
    "done": false
  }
  ```

### `/step` (POST)

* **Request:**

  ```json
  {
    "env_id": "...",
    "episode_id": "ep1",
    "action": { "item_id": 2 }
  }
  ```
* **Response:**

  ```json
  {
    "observation": { ...updated... },
    "reward": { "base": 0.83, "phi": 0.6667, "beta": 0.1, "final": 0.8967 },
    "done": false
  }
  ```

### `/episode/{episode_id}/summary` (GET)

* **Response:**

  ```json
  {
    "episode_id": "ep1",
    "final_knapsack": [2,7,9],
    "objective_sums_raw": [27,13,35],
    "pareto_distance": 0.12
  }
  ```

> **Note:** The client only *requires* `budget`, `remaining`, and `advisor_sets` in observations, and `reward.final` (or a numeric reward). Everything else is optional.

---

## 5 Features & Architecture

### Two-Tower Network

* **State Tower** (`state_features → s_emb`): summarizes the *whole state* once per step.
* **Item Tower** (`item_features → e_i`): encodes each candidate item independently.
* **Policy Head:** scores `[e_i || s_emb]` → logit per item; masked softmax yields action probs.
* **Value Head:** predicts scalar $V(s_t)$ from `s_emb`.

### State Features (shared per step)

Let $K$ = number of objectives.

We use **episode-local scaling** with $N_0 = |\mathcal{R}_0|$ (the item count on the first observation after `/reset`):

* `budget_frac` $= B_t / B_0$
* `remaining_frac` $= |\mathcal R_t| / N_0$
* `vmax[1..K]` over remaining (columnwise max of objectives matrix $V$)
* `vmean[1..K]` over remaining
* `vstd[1..K]` over remaining
* `advisor_frac[1..K]` where

  $$
  \text{advisor\_frac}^{(k)} = \frac{|\hat S_t^{(k)}|}{|\mathcal R_t|}
  $$

> This reflects **market/horizon** and **advisor breadth**; it’s Markov-sufficient for your current additive reward.

### Item Features (per candidate action)

For item $i$ with cost $C_i$, objective vector $v_i \in \mathbb{R}^K$:

* `cost_norm` $= C_i / (B_t + \varepsilon)$
* `Vnorm[1..K]` $= v_i^{(k)} / \max_{j \in \mathcal R_t} v_j^{(k)}$
* `phi` $=$ fraction of advisors including $i$:

  $$
  \phi_t(i) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\{ i \in \hat S_t^{(k)} \}
  $$
* `ratio[1..K]` $= v_i^{(k)} / (C_i + \varepsilon)$

The network builds one logit per **feasible** item and masks infeasible ones (cost > budget) to probability 0.

---

## 6 Training Algorithm (A2C, per-step updates)

At time $t$:

1. **Forward pass:**

   * Compute logits over feasible items; masked softmax → Categorical policy.
   * Sample action $a_t$, compute $\log \pi(a_t|s_t)$ and value $V(s_t)$.
2. **Environment step:** call `/step` with `item_id`; get `reward`, `observation'`.
3. **Bootstrap:** compute $V(s_{t+1})$ (0 if terminal).
4. **Targets & advantage:**

   $$
   \text{target}_t = r_t + \gamma V(s_{t+1}) \quad;\quad A_t = \text{target}_t - V(s_t)
   $$
5. **Losses:**

   * Policy: $-\log \pi(a_t|s_t) \cdot A_t$
   * Value:  $\frac{1}{2}(\text{target}_t - V(s_t))^2$
   * Entropy bonus: encourages exploration
6. **Update:** backprop & optimizer step. Repeat until `done`.

### Exploration via Temperature

We scale logits by `tau` (softmax temperature) before sampling; smaller `tau` = greedier. At episode end, we can modulate:

$$
\tau \leftarrow \text{clip}\big(\tau \cdot (1 + \kappa \cdot \text{ParetoDistance}), \tau_{\min}, \tau_{\max}\big)
$$

This nudges exploration when the final knapsack is far from the frontier.

---

## 7 Reproducibility

* Call `/seed` once before training with a fixed seed (e.g., 42).
* Call `/seed` again before evaluation with a different fixed seed (e.g., 999) to create a **common eval stream** for model comparisons.
* For full traceability, log `{seed, episode_id}` (and optionally a catalog hash if your server provides one).

---

## 8 Checkpoints & Plots

* The script periodically saves `./checkpoints/actor_critic_citybuilder.pt` (configurable).
* Training curve saved as `returns.png`.
* To resume: `python -m src.main train --load ...`

---

## 9 Extending / Modifying

* **Progress features (optional):** If you add path-dependent mechanics later (diversity, capacities, synergies), include small progress features (spent fraction, picks count, cumulative objective sums) in `state_features`. This can be done **client-side** without API changes by caching the initial catalog and tracking what you pick.
* **Larger models:** Increase hidden sizes or add layers in `models/nets.py`.
* **Different RL algo:** Swap out A2C for PPO/REINFORCE by changing `agent/` and `training/loops.py`.
* **Server contracts:** If you add new fields (e.g., explicit `knapsack` or `objectives_so_far`), the client can consume them but does not require them.

---

## 10 Troubleshooting

* **ConnectionError / timeouts:** Check `--base-url` and that the server is reachable.
* **Infeasible actions only:** Ensure server returns valid `remaining` with some `cost ≤ budget`; client masks infeasible rows.
* **Unstable learning:** Lower `--lr` to `1e-4`; raise `--entropy` to `5e-3`; verify rewards are within a sensible range.
* **CUDA issues:** Use `--device cpu` or reduce model size in `nets.py`.

---


