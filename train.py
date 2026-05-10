"""
train.py
========
Train a Q-learning agent on the Coral Reef environment.

Usage:
    python train.py --config configs/qlearning_v1.yaml

MLOps outputs:
  - experiments/results_run1.csv   — per-run metrics
  - policies/policy_v1.pkl          — snapshot at episode 400
  - policies/policy_v2.pkl          — converged final policy
  - experiments/plots/reward_over_episodes.png
  - experiments/plots/coral_health_comparison.png
"""

import argparse
import csv
import json
import os
import pickle
import random
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import yaml
from tqdm import tqdm

from sim.environment import CoralReefEnv, N_STATES, N_ACTIONS, INTERVENTION_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixed-Schedule Baseline Policy
# ─────────────────────────────────────────────────────────────────────────────

class FixedSchedulePolicy:
    """Intervene (apply shading = action 2) every `interval` steps."""

    def __init__(self, interval: int = 10):
        self.interval = interval
        self._step = 0

    def select_action(self, state: int) -> int:  # noqa: ARG002
        self._step += 1
        return 2 if (self._step % self.interval == 0) else 0

    def reset(self) -> None:
        self._step = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Q-Learning Agent
# ─────────────────────────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = 42,
    ):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.lr         = lr
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # ── Action selection ──────────────────────────────────────────────────── #
    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    # ── Q-table update ────────────────────────────────────────────────────── #
    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        best_next = np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.lr * (td_target - self.Q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Serialisation ─────────────────────────────────────────────────────── #
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "epsilon": self.epsilon, "lr": self.lr}, f)
        print(f"  [save] Policy saved -> {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(N_STATES, N_ACTIONS, **kwargs)
        agent.Q = data["Q"]
        agent.epsilon = data["epsilon"]
        return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: CoralReefEnv, agent: QLearningAgent, max_steps: int, train: bool = True):
    state = env.reset()
    total_reward = 0.0
    final_info = {}

    for _ in range(max_steps):
        action = agent.select_action(state) if train else agent.greedy_action(state)
        next_state, reward, done, info = env.step(action)

        if train:
            agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        final_info = info

        if done:
            break

    return total_reward, final_info


def run_fixed_episode(env: CoralReefEnv, policy: FixedSchedulePolicy, max_steps: int):
    state = env.reset()
    policy.reset()
    total_reward = 0.0
    final_info = {}

    for _ in range(max_steps):
        action = policy.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        final_info = info
        if done:
            break

    return total_reward, final_info


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_over_episodes(rewards: list, plot_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    eps = np.arange(1, len(rewards) + 1)
    smoothed = _smooth(rewards, window=30)
    offset = len(rewards) - len(smoothed)

    ax.plot(eps, rewards, color="#30a3ff", alpha=0.25, linewidth=0.8, label="Raw reward")
    ax.plot(eps[offset:], smoothed, color="#00e5ff", linewidth=2.2, label="Smoothed (w=30)")

    ax.set_title("Q-Learning: Reward over Episodes", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Episode", color="#8b949e")
    ax.set_ylabel("Total Reward", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    out = Path(plot_dir) / "reward_over_episodes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {out}")


def plot_coral_health_comparison(
    rl_health: list,
    fixed_health: list,
    plot_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    eps_rl    = np.arange(1, len(rl_health) + 1)
    eps_fixed = np.arange(1, len(fixed_health) + 1)

    sm_rl    = _smooth(rl_health,    window=20)
    sm_fixed = _smooth(fixed_health, window=20)

    ax.plot(eps_rl,    rl_health,    color="#00e5ff", alpha=0.2, linewidth=0.7)
    ax.plot(eps_rl[len(rl_health) - len(sm_rl):], sm_rl,
            color="#00e5ff", linewidth=2.2, label="RL Adaptive Policy")

    ax.plot(eps_fixed, fixed_health, color="#ff6b6b", alpha=0.2, linewidth=0.7)
    ax.plot(eps_fixed[len(fixed_health) - len(sm_fixed):], sm_fixed,
            color="#ff6b6b", linewidth=2.2, label="Fixed Schedule Policy")

    ax.set_title("Coral Health: RL vs Fixed Schedule Policy", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Evaluation Episode", color="#8b949e")
    ax.set_ylabel("Avg Coral Health Score", color="#8b949e")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    out = Path(plot_dir) / "coral_health_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  MLOps Logger
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_id", "episodes", "avg_reward", "avg_coral_health",
    "epsilon", "learning_rate", "interventions_deployed",
]


def save_mlops_tracking(base_dir: str, run_id: str, results: dict, config: dict) -> None:
    """Save a per-run CSV and JSON log for experiment tracking and reproducibility."""
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"results_{run_id}.csv"
    json_path = out_dir / f"log_{run_id}.json"
    
    # 1. Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        
        # Filter results dict to match CSV fields
        csv_row = {k: results.get(k, None) for k in CSV_FIELDS}
        writer.writerow(csv_row)
        
    # 2. Save JSON with full tracking info
    tracking_data = {
        "metadata": {"run_id": run_id},
        "metrics": results,
        "parameters": config,
    }
    with open(json_path, "w") as f:
        json.dump(tracking_data, f, indent=4)
        
    print(f"  [mlops] Saved run tracking -> {csv_path}")
    print(f"  [mlops] Saved run tracking -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Coral Reef RL Agent")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent_cfg = cfg["agent"]
    exp_cfg   = cfg["experiment"]
    log_cfg   = cfg["logging"]
    rew_cfg   = cfg["rewards"]

    set_seed(exp_cfg["seed"])

    episodes   = agent_cfg["episodes"]
    max_steps  = agent_cfg["max_steps_per_episode"]
    save_v1_at = log_cfg["save_policy_v1_at"]

    print("=" * 60)
    print("  Adaptive Coral Reef Monitoring - Q-Learning Trainer")
    print("=" * 60)
    print(f"  Config       : {args.config}")
    print(f"  Episodes     : {episodes}")
    print(f"  Learning rate: {agent_cfg['learning_rate']}")
    print(f"  Epsilon      : {agent_cfg['epsilon']}  (decay {agent_cfg['epsilon_decay']})")
    print(f"  Gamma        : {agent_cfg['discount_factor']}")
    print("=" * 60)

    env = CoralReefEnv(seed=exp_cfg["seed"])

    agent = QLearningAgent(
        n_states       = N_STATES,
        n_actions      = N_ACTIONS,
        lr             = agent_cfg["learning_rate"],
        gamma          = agent_cfg["discount_factor"],
        epsilon        = agent_cfg["epsilon"],
        epsilon_decay  = agent_cfg["epsilon_decay"],
        epsilon_min    = agent_cfg["epsilon_min"],
        seed           = exp_cfg["seed"],
    )

    # ── Training ─────────────────────────────────────────────────────────── #
    rewards_hist       = []
    health_hist_rl     = []
    total_interventions = 0

    for ep in tqdm(range(1, episodes + 1), desc="Training", unit="ep", ncols=70):
        total_reward, info = run_episode(env, agent, max_steps, train=True)
        agent.decay_epsilon()

        rewards_hist.append(total_reward)
        health_hist_rl.append(info.get("coral_health", 0.0))
        total_interventions += info.get("interventions", 0)

        # Save policy v1 snapshot
        if ep == save_v1_at:
            agent.save(log_cfg["policy_v1_path"])

    # Save final (converged) policy v2
    agent.save(log_cfg["policy_v2_path"])

    # ── Fixed-baseline evaluation ─────────────────────────────────────────── #
    fixed_policy = FixedSchedulePolicy(interval=10)
    eval_episodes = min(200, episodes)
    health_hist_fixed = []
    fixed_interv_count = 0
    fixed_unnecessary  = 0
    fixed_early_warn   = 0
    fixed_early_total  = 0

    for _ in range(eval_episodes):
        _, info = run_fixed_episode(env, fixed_policy, max_steps)
        health_hist_fixed.append(info.get("coral_health", 0.0))
        fixed_interv_count += info.get("interventions", 0)
        fixed_early_warn   += info.get("early_warn_caught", 0)
        fixed_early_total  += info.get("early_warn_total", 1)

    # ── RL greedy evaluation ──────────────────────────────────────────────── #
    rl_reward_eval    = []
    rl_health_eval    = []
    rl_interv_count   = 0
    rl_unnecessary    = 0
    rl_early_warn     = 0
    rl_early_total    = 0

    for _ in range(eval_episodes):
        r, info = run_episode(env, agent, max_steps, train=False)
        rl_reward_eval.append(r)
        rl_health_eval.append(info.get("coral_health", 0.0))
        rl_interv_count += info.get("interventions", 0)
        rl_early_warn   += info.get("early_warn_caught", 0)
        rl_early_total  += info.get("early_warn_total", 1)

    # ── Metrics ──────────────────────────────────────────────────────────── #
    avg_reward       = float(np.mean(rl_reward_eval))
    avg_coral_health = float(np.mean(rl_health_eval))

    run_id = f"{exp_cfg['run_id']}_{uuid.uuid4().hex[:6]}"
    
    results = {
        "run_id":                run_id,
        "episodes":              episodes,
        "avg_reward":            round(avg_reward, 4),
        "avg_coral_health":      round(avg_coral_health, 4),
        "epsilon":               round(agent.epsilon, 5),
        "learning_rate":         agent_cfg["learning_rate"],
        "interventions_deployed": rl_interv_count,
    }
    
    # Save per-run MLOps tracking
    save_mlops_tracking(
        base_dir="experiments",
        run_id=run_id,
        results=results,
        config=cfg
    )

    # ── Plots ─────────────────────────────────────────────────────────────── #
    plot_reward_over_episodes(rewards_hist, log_cfg["plot_dir"])
    plot_coral_health_comparison(health_hist_rl, health_hist_fixed, log_cfg["plot_dir"])

    # ── Comparison Table ─────────────────────────────────────────────────── #
    rl_surv   = avg_coral_health * 100
    fix_surv  = float(np.mean(health_hist_fixed)) * 100

    rl_unnec_pct  = (rl_unnecessary  / max(rl_interv_count,  1)) * 100
    fix_unnec_pct = (fixed_unnecessary / max(fixed_interv_count, 1)) * 100

    rl_ew_pct   = (rl_early_warn  / max(rl_early_total,  1)) * 100
    fix_ew_pct  = (fixed_early_warn / max(fixed_early_total, 1)) * 100

    print()
    print("=" * 60)
    print("  BASELINE COMPARISON: Fixed Schedule vs RL Adaptive")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Fixed':>8} {'RL':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    print(f"  {'Avg Coral Health Score':<35} {fix_surv:>7.1f}% {rl_surv:>7.1f}%")
    print(f"  {'Unnecessary Interventions (%)':<35} {fix_unnec_pct:>7.1f}% {rl_unnec_pct:>7.1f}%")
    print(f"  {'Early Warnings Caught (%)':<35} {fix_ew_pct:>7.1f}% {rl_ew_pct:>7.1f}%")
    print(f"  {'Overall Reef Survival Rate (%)':<35} {fix_surv:>7.1f}% {rl_surv:>7.1f}%")
    print("=" * 60)
    print(f"\n  run_id          : {run_id}")
    print(f"  avg_reward (RL) : {avg_reward:.2f}")
    print(f"  Final epsilon   : {agent.epsilon:.5f}")
    print("\nTraining complete!\n")


if __name__ == "__main__":
    main()
