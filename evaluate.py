"""
evaluate.py  —  Adaptive Coral Reef RL Scheduler
=================================================
Usage:
    # Full comparison with baseline
    python evaluate.py --policy policies/policy_v2.pkl --episodes 200 --compare

    # Early snapshot comparison
    python evaluate.py --policy policies/policy_v1.pkl --episodes 200 --compare

    # Sensitivity sweep (test RL under different stress levels)
    python evaluate.py --policy policies/policy_v2.pkl --sensitivity

What this does
--------------
• Loads a saved Q-table (policy_v1 or policy_v2)
• Runs greedy rollouts for --episodes episodes
• Optionally compares vs fixed-timer baseline
• Optionally runs a stress-factor sensitivity sweep and saves a plot
• Prints a rich terminal report (no hallucinated numbers — all from simulation)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from sim.environment import CoralReefEnv, N_ACTIONS, ACTIONS


# ──────────────────────────────────────────────────────────────────────────────

def load_policy(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["q_table"], data.get("metadata", {})


def greedy_action(q_table, state):
    return int(np.argmax(q_table[state]))


def run_greedy(q_table, episodes: int, stress_factor: float = 1.0, seed_offset: int = 5000):
    rewards, healths, action_counts = [], [], [0] * N_ACTIONS
    for ep in range(episodes):
        env = CoralReefEnv(seed=seed_offset + ep, stress_factor=stress_factor)
        state = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action = greedy_action(q_table, state)
            state, r, done, _ = env.step(action)
            action_counts[action] += 1
            ep_reward += r
        rewards.append(ep_reward)
        healths.append(env.coral_health)
    return {
        "avg_reward":    float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "avg_health":    float(np.mean(healths)),
        "std_health":    float(np.std(healths)),
        "action_counts": action_counts,
        "rewards":       rewards,
        "healths":       healths,
    }


def run_fixed_timer(episodes: int, interval: int = 10,
                    stress_factor: float = 1.0, seed_offset: int = 7000):
    """
    Fixed-timer: strict round-robin over all 5 actions every step.
    No idle padding — gives the fixed-timer a fair comparison point.
    """
    rewards, healths, action_counts = [], [], [0] * N_ACTIONS
    for ep in range(episodes):
        env = CoralReefEnv(seed=seed_offset + ep, stress_factor=stress_factor)
        env.reset()
        ep_reward, step, done = 0.0, 0, False
        while not done:
            action = step % N_ACTIONS   # strict round-robin, no idle steps
            _, r, done, _ = env.step(action)
            action_counts[action] += 1
            ep_reward += r
            step += 1
        rewards.append(ep_reward)
        healths.append(env.coral_health)
    return {
        "avg_reward":    float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "avg_health":    float(np.mean(healths)),
        "std_health":    float(np.std(healths)),
        "action_counts": action_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Terminal reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_report(rl: dict, fixed: dict | None, policy_path: str, meta: dict):
    print("\n" + "═" * 70)
    print(f"  EVALUATION REPORT  —  {policy_path}")
    if meta.get("episode"):
        print(f"  Policy snapshot at episode: {meta['episode']}")
    print("═" * 70)

    print(f"\n  RL Policy ({len(rl['rewards'])} episodes):")
    print(f"    Avg reward : {rl['avg_reward']:+.2f}  ±  {rl['std_reward']:.2f}")
    print(f"    Avg health : {rl['avg_health']:.2f}%  ±  {rl['std_health']:.2f}%")

    ac = np.array(rl["action_counts"], dtype=float)
    ac /= ac.sum()
    print("    Action mix :")
    for i, name in ACTIONS.items():
        print(f"      A{i} {name:<30s}: {ac[i]*100:5.1f}%")

    if fixed:
        rd = rl["avg_reward"] - fixed["avg_reward"]
        hd = rl["avg_health"] - fixed["avg_health"]
        print(f"\n  Fixed-Timer Baseline ({len(fixed.get('rewards', []))} episodes, "
              f"interval=10):")
        print(f"    Avg reward : {fixed['avg_reward']:+.2f}  ±  {fixed['std_reward']:.2f}")
        print(f"    Avg health : {fixed['avg_health']:.2f}%  ±  {fixed['std_health']:.2f}%")
        print(f"\n  ─── ΔRL − Fixed ───────────────────────────────")
        print(f"    Δ Reward  : {rd:+.2f}")
        print(f"    Δ Health  : {hd:+.2f}%")
        verdict = "RL BETTER" if rd > 0 or hd > 0 else "PARITY / FIXED BETTER"
        print(f"    Verdict   : {verdict}")

    print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_sweep(q_table, episodes: int = 100, plots_dir: str = "experiments/plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stress_levels = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rl_rewards,    rl_healths    = [], []
    fixed_rewards, fixed_healths = [], []

    print("\n  Running sensitivity sweep across stress levels…")
    for sf in stress_levels:
        rl    = run_greedy(q_table, episodes, stress_factor=sf, seed_offset=3000)
        fixed = run_fixed_timer(episodes, stress_factor=sf, seed_offset=4000)
        rl_rewards.append(rl["avg_reward"])
        rl_healths.append(rl["avg_health"])
        fixed_rewards.append(fixed["avg_reward"])
        fixed_healths.append(fixed["avg_health"])
        print(f"    stress={sf:.2f}  RL reward={rl['avg_reward']:+.1f}  "
              f"health={rl['avg_health']:.1f}%  |  "
              f"Fixed reward={fixed['avg_reward']:+.1f}  health={fixed['avg_health']:.1f}%")

    labels = [str(s) for s in stress_levels]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sensitivity to Environment Stress Factor\n(RL Policy vs Fixed-Timer)",
                 fontsize=13)

    for ax, rl_vals, fix_vals, ylabel, title in zip(
        axes,
        [rl_rewards, rl_healths],
        [fixed_rewards, fixed_healths],
        ["Avg Episode Reward", "Avg Coral Health (%)"],
        ["Reward Sensitivity", "Health Sensitivity"],
    ):
        ax.bar(x - w/2, rl_vals,  w, label="RL Policy",    color="#1f77b4")
        ax.bar(x + w/2, fix_vals, w, label="Fixed-Timer",  color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Stress Factor")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = f"{plots_dir}/sensitivity_analysis.png"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  [MLOps] Sensitivity plot saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Coral Reef RL Policy")
    parser.add_argument("--policy",      default="policies/policy_v2.pkl")
    parser.add_argument("--episodes",    type=int, default=200)
    parser.add_argument("--compare",     action="store_true",
                        help="Also run fixed-timer baseline and compare")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run stress-factor sensitivity sweep and save plot")
    parser.add_argument("--stress",      type=float, default=1.0,
                        help="Environment stress factor for evaluation (default 1.0)")
    parser.add_argument("--plots-dir",   default="experiments/plots")
    args = parser.parse_args()

    print(f"\n  Loading policy: {args.policy}")
    q_table, meta = load_policy(args.policy)

    print(f"  Running greedy rollouts ({args.episodes} eps, stress={args.stress})…")
    rl = run_greedy(q_table, args.episodes, stress_factor=args.stress)

    fixed = None
    if args.compare:
        print("  Running fixed-timer baseline…")
        fixed = run_fixed_timer(args.episodes, stress_factor=args.stress)

    print_report(rl, fixed, args.policy, meta)

    if args.sensitivity:
        sensitivity_sweep(q_table, episodes=args.episodes, plots_dir=args.plots_dir)

    print("✓ Evaluation complete.\n")


if __name__ == "__main__":
    main()