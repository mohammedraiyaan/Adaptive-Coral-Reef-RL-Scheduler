"""
evaluate.py
===========
Evaluate a saved Q-learning policy against the fixed-schedule baseline.

Usage:
    python evaluate.py --policy policies/policy_v2.pkl --episodes 200
    python evaluate.py --policy policies/policy_v1.pkl --episodes 200 --compare
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from sim.environment import CoralReefEnv, N_STATES, N_ACTIONS, INTERVENTION_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
#  Load policy Q-table
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  [load] Policy loaded <- {path}  (epsilon={data.get('epsilon', 'n/a'):.4f})")
    return data["Q"]


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def greedy_action(Q: np.ndarray, state: int) -> int:
    return int(np.argmax(Q[state]))


def eval_rl_policy(Q: np.ndarray, env: CoralReefEnv, episodes: int, max_steps: int):
    rewards, healths, intervs, ew_caught, ew_total = [], [], [], [], []
    unnecessary = 0

    for _ in range(episodes):
        state = env.reset()
        total_r = 0.0
        ep_interv = 0
        ep_unnec  = 0

        for _ in range(max_steps):
            action = greedy_action(Q, state)
            state, reward, done, info = env.step(action)
            total_r   += reward
            if action in INTERVENTION_ACTIONS:
                ep_interv += 1
                if info["bleach"] == 0 and info["temp"] == 0:
                    ep_unnec += 1
            if done:
                break

        rewards.append(total_r)
        healths.append(info["coral_health"])
        intervs.append(ep_interv)
        ew_caught.append(info["early_warn_caught"])
        ew_total.append(max(info["early_warn_total"], 1))
        unnecessary += ep_unnec

    return {
        "avg_reward":        float(np.mean(rewards)),
        "avg_health":        float(np.mean(healths)),
        "total_interventions": int(sum(intervs)),
        "unnecessary_pct":   unnecessary / max(sum(intervs), 1) * 100,
        "early_warn_pct":    sum(ew_caught) / sum(ew_total) * 100,
        "survival_rate":     float(np.mean(healths)) * 100,
    }


def eval_fixed_policy(env: CoralReefEnv, episodes: int, max_steps: int, interval: int = 10):
    rewards, healths, intervs, ew_caught, ew_total = [], [], [], [], []
    unnecessary = 0
    step_global = 0

    for _ in range(episodes):
        state = env.reset()
        total_r   = 0.0
        ep_interv = 0
        ep_unnec  = 0

        for _ in range(max_steps):
            step_global += 1
            action = 2 if (step_global % interval == 0) else 0
            state, reward, done, info = env.step(action)
            total_r += reward
            if action in INTERVENTION_ACTIONS:
                ep_interv += 1
                if info["bleach"] == 0 and info["temp"] == 0:
                    ep_unnec += 1
            if done:
                break

        rewards.append(total_r)
        healths.append(info["coral_health"])
        intervs.append(ep_interv)
        ew_caught.append(info["early_warn_caught"])
        ew_total.append(max(info["early_warn_total"], 1))
        unnecessary += ep_unnec

    return {
        "avg_reward":        float(np.mean(rewards)),
        "avg_health":        float(np.mean(healths)),
        "total_interventions": int(sum(intervs)),
        "unnecessary_pct":   unnecessary / max(sum(intervs), 1) * 100,
        "early_warn_pct":    sum(ew_caught) / sum(ew_total) * 100,
        "survival_rate":     float(np.mean(healths)) * 100,
    }


def print_comparison(rl: dict, fixed: dict) -> None:
    print()
    print("=" * 65)
    print("  EVALUATION REPORT: Fixed Schedule Policy vs RL Adaptive Policy")
    print("=" * 65)
    hdr = f"  {'Metric':<38} {'Fixed':>10} {'RL':>10}"
    print(hdr)
    print(f"  {'-'*38} {'-'*10} {'-'*10}")

    metrics = [
        ("Avg Reward per Episode",       "avg_reward",        ".2f",  ""),
        ("Avg Coral Health Score",        "avg_health",        ".3f",  ""),
        ("Total Interventions Deployed",  "total_interventions",".0f", ""),
        ("Unnecessary Interventions (%)", "unnecessary_pct",   ".1f",  "%"),
        ("Early Warnings Caught (%)",     "early_warn_pct",    ".1f",  "%"),
        ("Overall Reef Survival Rate (%)", "survival_rate",    ".1f",  "%"),
    ]

    for label, key, fmt, unit in metrics:
        fv = format(fixed[key], fmt) + unit
        rv = format(rl[key],    fmt) + unit
        print(f"  {label:<38} {fv:>10} {rv:>10}")

    print("=" * 65)
    delta = rl["survival_rate"] - fixed["survival_rate"]
    sign  = "+" if delta >= 0 else ""
    print(f"\n  RL policy survival improvement: {sign}{delta:.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Coral Reef RL Policy")
    parser.add_argument("--policy",   required=True,      help="Path to .pkl policy file")
    parser.add_argument("--episodes", type=int, default=200, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed",     type=int, default=0,   help="RNG seed")
    parser.add_argument("--compare",  action="store_true",   help="Also run fixed-schedule baseline")
    args = parser.parse_args()

    env = CoralReefEnv(seed=args.seed)
    Q   = load_policy(args.policy)

    print(f"\n  Evaluating {args.policy}  |  {args.episodes} episodes  |  max_steps={args.max_steps}")

    rl_metrics = eval_rl_policy(Q, env, args.episodes, args.max_steps)

    if args.compare:
        fixed_metrics = eval_fixed_policy(env, args.episodes, args.max_steps)
        print_comparison(rl_metrics, fixed_metrics)
    else:
        print()
        print("  ── RL Policy Metrics ──")
        for k, v in rl_metrics.items():
            print(f"  {k:<35}: {v:.3f}")
        print()


if __name__ == "__main__":
    main()
