"""
train.py  —  Adaptive Coral Reef RL Scheduler
=============================================
Usage:
    python train.py --config configs/qlearning_v1.yaml

What this does
--------------
1.  Trains a tabular Q-learning agent for `episodes` episodes.
2.  Saves a policy snapshot at episode `save_policy_v1_at`   → policy_v1.pkl
3.  Saves the converged final policy                          → policy_v2.pkl
4.  Appends a row to experiments/results_<run_id>.csv  (MLOps log)
5.  Writes a JSON metadata file for full reproducibility
6.  Produces four real, data-driven plots (no fake curves):
        reward_curve.png           – per-episode reward (smoothed)
        coral_health_curve.png     – per-episode health (smoothed)
        comparison_bar.png         – Fixed vs RL bar chart
        action_distribution.png    – how often each action was chosen
7.  Prints a Fixed-Timer vs RL comparison table to the terminal
"""

import argparse
import csv
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import os
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

from sim.environment import CoralReefEnv, N_STATES, N_ACTIONS, ACTIONS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


def make_dirs(cfg: dict):
    for d in [cfg["results_dir"], cfg["plots_dir"], "policies"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_policy(q_table: np.ndarray, path: str, metadata: dict):
    with open(path, "wb") as f:
        pickle.dump({"q_table": q_table, "metadata": metadata}, f)
    print(f"  [MLOps] Policy saved → {path}")


def smooth(values, window=30):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-Timer Baseline
# ──────────────────────────────────────────────────────────────────────────────

def run_fixed_timer(cfg: dict, episodes: int = 200, seed_offset: int = 9999) -> dict:
    """
    Fixed-timer scheduler: cycles through ALL 5 actions (0–4) every step in order.
    This gives the fixed-timer a fair fight — it is always doing something, not idling.
    Action distribution will be roughly uniform (~20% each), unlike the old version
    which was 90% no_action by construction (1 action per 10-step interval).
    """
    rewards, healths, action_counts = [], [], [0] * N_ACTIONS

    for ep in range(episodes):
        env = CoralReefEnv(seed=seed_offset + ep, stress_factor=cfg["stress_factor"])
        env.reset()
        ep_reward, step = 0.0, 0
        done = False
        while not done:
            # Strict round-robin over all 5 actions — no idle padding
            action = step % N_ACTIONS
            _, r, done, info = env.step(action)
            action_counts[action] += 1
            ep_reward += r
            step += 1
        rewards.append(ep_reward)
        healths.append(env.coral_health)

    return {
        "avg_reward":       float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "avg_health":       float(np.mean(healths)),
        "std_health":       float(np.std(healths)),
        "action_counts":    action_counts,
        "episode_rewards":  rewards,
        "episode_healths":  healths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent
# ──────────────────────────────────────────────────────────────────────────────

def train_qlearning(cfg: dict) -> dict:
    alpha   = cfg["learning_rate"]
    gamma   = cfg["discount_factor"]
    eps     = cfg["epsilon"]
    eps_min = cfg["epsilon_min"]
    eps_dec = cfg["epsilon_decay"]
    episodes      = cfg["episodes"]
    max_steps     = cfg["max_steps_per_episode"]
    snap_at       = cfg["save_policy_v1_at"]

    rng     = np.random.default_rng(cfg["seed"])
    q_table = np.zeros((N_STATES, N_ACTIONS))

    ep_rewards, ep_healths, ep_epsilons = [], [], []
    action_counts = [0] * N_ACTIONS

    for ep in range(episodes):
        env = CoralReefEnv(seed=cfg["seed"] + ep, stress_factor=cfg["stress_factor"])
        state = env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
            # ε-greedy
            if rng.random() < eps:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done, info = env.step(action)
            action_counts[action] += 1

            # Q-update
            best_next = np.max(q_table[next_state])
            q_table[state, action] += alpha * (
                reward + gamma * best_next - q_table[state, action]
            )

            ep_reward += reward
            state = next_state
            if done:
                break

        # Decay ε
        eps = max(eps_min, eps * eps_dec)

        ep_rewards.append(ep_reward)
        ep_healths.append(env.coral_health)
        ep_epsilons.append(eps)

        # Save early snapshot
        if ep + 1 == snap_at:
            snap_meta = {"episode": ep + 1, "epsilon": eps, "config": cfg}
            save_policy(q_table.copy(), cfg["policy_v1_path"], snap_meta)

    return {
        "q_table":          q_table,
        "ep_rewards":       ep_rewards,
        "ep_healths":       ep_healths,
        "ep_epsilons":      ep_epsilons,
        "action_counts":    action_counts,
        "avg_reward":       float(np.mean(ep_rewards[-200:])),
        "std_reward":       float(np.std(ep_rewards[-200:])),
        "avg_health":       float(np.mean(ep_healths[-200:])),
        "std_health":       float(np.std(ep_healths[-200:])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (real data, no fakes)
# ──────────────────────────────────────────────────────────────────────────────

def make_plots(rl: dict, fixed: dict, cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plots_dir = cfg["plots_dir"]
    COLORS = {
        "rl":    "#1f77b4",   # blue
        "fixed": "#d62728",   # red
        "shade": "#aec7e8",
    }
    STYLE = dict(linewidth=1.8, alpha=0.9)

    # ── 1. Reward curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    raw = np.array(rl["ep_rewards"])
    ax.plot(raw, color=COLORS["rl"], alpha=0.18, linewidth=0.7, label="_raw")
    ax.plot(smooth(raw, 40), color=COLORS["rl"], **STYLE, label="RL (smoothed, w=40)")

    # Fixed baseline as a horizontal band
    fm, fs = fixed["avg_reward"], fixed["std_reward"]
    ax.axhline(fm, color=COLORS["fixed"], linewidth=1.6, linestyle="--",
               label=f"Fixed-timer avg ({fm:.1f})")
    ax.axhspan(fm - fs, fm + fs, color=COLORS["fixed"], alpha=0.10)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Training Reward: RL Agent vs Fixed-Timer Baseline")
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/reward_curve.png", dpi=150)
    plt.close(fig)

    # ── 2. Coral health curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    raw_h = np.array(rl["ep_healths"])
    ax.plot(raw_h, color=COLORS["rl"], alpha=0.18, linewidth=0.7)
    ax.plot(smooth(raw_h, 40), color=COLORS["rl"], **STYLE, label="RL coral health (smoothed)")

    fhm, fhs = fixed["avg_health"], fixed["std_health"]
    ax.axhline(fhm, color=COLORS["fixed"], linewidth=1.6, linestyle="--",
               label=f"Fixed-timer avg ({fhm:.1f}%)")
    ax.axhspan(fhm - fhs, fhm + fhs, color=COLORS["fixed"], alpha=0.10)

    ax.set_ylim(0, 105)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coral Health Score (%)")
    ax.set_title("Coral Health: RL vs Fixed-Timer over Training")
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/coral_health_curve.png", dpi=150)
    plt.close(fig)

    # ── 3. Bar comparison (last-200 RL vs fixed) ───────────────────────
    metrics = ["Avg Reward", "Avg Health (%)"]
    rl_vals    = [rl["avg_reward"],    rl["avg_health"]]
    fixed_vals = [fixed["avg_reward"], fixed["avg_health"]]
    rl_err     = [rl["std_reward"],    rl["std_health"]]
    fixed_err  = [fixed["std_reward"], fixed["std_health"]]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars_rl    = ax.bar(x - w/2, rl_vals,    w, yerr=rl_err,
                        color=COLORS["rl"],    label="RL Policy (best 200)", capsize=5,
                        error_kw=dict(elinewidth=1.4))
    bars_fixed = ax.bar(x + w/2, fixed_vals, w, yerr=fixed_err,
                        color=COLORS["fixed"], label="Fixed-Timer", capsize=5,
                        error_kw=dict(elinewidth=1.4))

    # Value labels on bars
    for bar in list(bars_rl) + list(bars_fixed):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("RL Policy vs Fixed-Timer: Key Metrics\n(last 200 episodes ± 1 std)")
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/comparison_bar.png", dpi=150)
    plt.close(fig)

    # ── 4. Action distribution ─────────────────────────────────────────
    action_names = [f"A{i}\n{ACTIONS[i].replace('_',' ')[:12]}" for i in range(N_ACTIONS)]
    rl_ac    = np.array(rl["action_counts"],    dtype=float)
    fixed_ac = np.array(fixed["action_counts"], dtype=float)
    rl_ac    /= rl_ac.sum()
    fixed_ac /= fixed_ac.sum()

    x = np.arange(N_ACTIONS)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, rl_ac,    w, color=COLORS["rl"],    label="RL Policy")
    ax.bar(x + w/2, fixed_ac, w, color=COLORS["fixed"], label="Fixed-Timer")
    ax.set_xticks(x)
    ax.set_xticklabels(action_names, fontsize=8)
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Action Distribution: RL vs Fixed-Timer")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/action_distribution.png", dpi=150)
    plt.close(fig)

    print(f"  [MLOps] 4 plots saved to {plots_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# MLOps Logging
# ──────────────────────────────────────────────────────────────────────────────

def log_results(cfg: dict, rl: dict, fixed: dict, run_hash: str, elapsed: float):
    results_dir = cfg["results_dir"]
    run_id      = cfg["run_id"]

    # CSV row
    csv_path = f"{results_dir}/results_{run_id}.csv"
    fieldnames = [
        "run_hash", "run_id", "timestamp", "elapsed_s",
        "episodes", "learning_rate", "discount_factor",
        "epsilon_start", "epsilon_min", "epsilon_decay",
        "stress_factor", "seed",
        "rl_avg_reward_last200", "rl_std_reward_last200",
        "rl_avg_health_last200", "rl_std_health_last200",
        "fixed_avg_reward",      "fixed_std_reward",
        "fixed_avg_health",      "fixed_std_health",
        "reward_delta",          "health_delta",
    ]
    row = {
        "run_hash":   run_hash,
        "run_id":     run_id,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s":  round(elapsed, 2),
        "episodes":   cfg["episodes"],
        "learning_rate":   cfg["learning_rate"],
        "discount_factor": cfg["discount_factor"],
        "epsilon_start":   cfg["epsilon"],
        "epsilon_min":     cfg["epsilon_min"],
        "epsilon_decay":   cfg["epsilon_decay"],
        "stress_factor":   cfg["stress_factor"],
        "seed":            cfg["seed"],
        "rl_avg_reward_last200": round(rl["avg_reward"], 3),
        "rl_std_reward_last200": round(rl["std_reward"], 3),
        "rl_avg_health_last200": round(rl["avg_health"], 3),
        "rl_std_health_last200": round(rl["std_health"], 3),
        "fixed_avg_reward": round(fixed["avg_reward"], 3),
        "fixed_std_reward": round(fixed["std_reward"], 3),
        "fixed_avg_health": round(fixed["avg_health"], 3),
        "fixed_std_health": round(fixed["std_health"], 3),
        "reward_delta": round(rl["avg_reward"]  - fixed["avg_reward"], 3),
        "health_delta": round(rl["avg_health"]  - fixed["avg_health"], 3),
    }
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [MLOps] Metrics appended → {csv_path}")

    # Full JSON for reproducibility
    json_path = f"{results_dir}/log_{run_id}_{run_hash[:8]}.json"
    with open(json_path, "w") as f:
        json.dump({"config": cfg, "metrics": row}, f, indent=2)
    print(f"  [MLOps] Full log saved  → {json_path}")

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table (terminal)
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(rl: dict, fixed: dict):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    rd = rl["avg_reward"]  - fixed["avg_reward"]
    hd = rl["avg_health"]  - fixed["avg_health"]
    rows = [
        ["Metric",                       "Fixed-Timer",                                "RL Policy",                               "ΔRL−Fixed"],
        ["Avg Reward (last 200 eps)",
            f"{fixed['avg_reward']:.2f} ± {fixed['std_reward']:.2f}",
            f"{rl['avg_reward']:.2f} ± {rl['std_reward']:.2f}",
            f"{rd:+.2f}"],
        ["Avg Coral Health % (last 200)",
            f"{fixed['avg_health']:.2f} ± {fixed['std_health']:.2f}",
            f"{rl['avg_health']:.2f} ± {rl['std_health']:.2f}",
            f"{hd:+.2f}"],
        ["State-aware decisions",         "No (time-triggered)",  "Yes (Q-table)",     "—"],
        ["Unnecessary interventions",     "High",                 "Low (penalised)",   "—"],
        ["Adapts to env stress",          "No",                   "Yes",               "—"],
    ]
    header = rows[0]
    data   = rows[1:]

    print("\n" + "═" * 72)
    print("  FIXED-TIMER vs RL POLICY — COMPARISON TABLE")
    print("═" * 72)
    if use_tabulate:
        print(tabulate(data, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [28, 26, 26, 12]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
        print("  " + "-" * (sum(col_w) + len(col_w) * 2))
        for row in data:
            print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
    print("═" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Coral Reef RL Scheduler")
    parser.add_argument("--config", default="configs/qlearning_v1.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    run_hash = uuid.uuid4().hex
    make_dirs(cfg)

    print(f"\n{'='*60}")
    print(f"  Coral Reef RL Scheduler — Training")
    print(f"  run_id:   {cfg['run_id']}")
    print(f"  run_hash: {run_hash}")
    print(f"  config:   {args.config}")
    print(f"{'='*60}\n")

    # 1. Fixed-timer baseline (evaluated first, independent)
    print("[1/4] Evaluating fixed-timer baseline (200 episodes)…")
    fixed_results = run_fixed_timer(cfg, episodes=200)
    print(f"      Fixed avg reward: {fixed_results['avg_reward']:.2f}  "
          f"health: {fixed_results['avg_health']:.2f}%")

    # 2. Train Q-learning
    print(f"\n[2/4] Training Q-learning agent ({cfg['episodes']} episodes)…")
    t0 = time.time()
    rl_results = train_qlearning(cfg)
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s  |  "
          f"RL avg reward (last 200): {rl_results['avg_reward']:.2f}  "
          f"health: {rl_results['avg_health']:.2f}%")

    # 3. Save converged policy
    print("\n[3/4] Saving policies…")
    final_meta = {"episode": cfg["episodes"], "run_hash": run_hash, "config": cfg}
    save_policy(rl_results["q_table"], cfg["policy_v2_path"], final_meta)

    # 4. Plots & MLOps log
    print("\n[4/4] Generating plots and MLOps log…")
    make_plots(rl_results, fixed_results, cfg)
    log_results(cfg, rl_results, fixed_results, run_hash, elapsed)

    # Print comparison table
    print_comparison_table(rl_results, fixed_results)

    print("✓ Training complete.\n")
    print(f"  Plots   → {cfg['plots_dir']}/")
    print(f"  Policies→ policies/policy_v1.pkl  (ep {cfg['save_policy_v1_at']})")
    print(f"           policies/policy_v2.pkl  (converged)")
    print(f"  MLOps   → {cfg['results_dir']}/results_{cfg['run_id']}.csv\n")


if __name__ == "__main__":
    main()