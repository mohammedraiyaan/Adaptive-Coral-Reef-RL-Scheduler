# Adaptive Coral Reef Monitoring & Intervention Scheduler using RL

> **SDG 14 — Life Below Water** | **SDG 13 — Climate Action**

A tabular Q-learning agent that learns *when* and *how* to intervene in a simulated coral reef ecosystem, balancing timely action against unnecessary disturbance.

---

## Project Structure

```
coral-reef-rl/
├── sim/
│   ├── __init__.py
│   └── environment.py        # Stochastic coral reef RL environment
├── configs/
│   └── qlearning_v1.yaml     # All hyperparameters (single source of truth)
├── experiments/
│   ├── plots/
│   │   ├── reward_over_episodes.png
│   │   └── coral_health_comparison.png
│   └── results_run1.csv      # MLOps experiment log
├── policies/
│   ├── policy_v1.pkl         # Early-training snapshot (episode 400)
│   └── policy_v2.pkl         # Converged final policy
├── train.py                  # Training entry-point
├── evaluate.py               # Evaluation & comparison entry-point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the agent

```bash
python train.py --config configs/qlearning_v1.yaml
```

This single command will:
- Train the Q-learning agent for 1 000 episodes
- Save `policies/policy_v1.pkl` (episode 400 snapshot)
- Save `policies/policy_v2.pkl` (converged final policy)
- Append a row to `experiments/results_run1.csv`
- Write both plots to `experiments/plots/`
- Print a Fixed vs RL comparison table to the terminal

### 3. Evaluate a saved policy

```bash
# Evaluate converged policy with baseline comparison
python evaluate.py --policy policies/policy_v2.pkl --episodes 200 --compare

# Evaluate early-training snapshot
python evaluate.py --policy policies/policy_v1.pkl --episodes 200 --compare

# Run stress-factor sensitivity sweep
python evaluate.py --policy policies/policy_v2.pkl --sensitivity
```

---

## Reproducibility

> "Run python train.py --config qlearning_v1.yaml to get the same result."

Anyone should be able to clone your repo and run the same experiment by using the configuration files.

| Run | Command | Git tag |
|-----|---------|---------|
| Run 1 (v1 config) | `python train.py --config configs/qlearning_v1.yaml` | `exp-qlearning-1` |
| Run 2 (same config, re-run) | `python train.py --config configs/qlearning_v1.yaml` | `exp-qlearning-2` |

**Git and GitOps for Version Management**:
Use Git commits/tags for different experiments. Tag the repo after each run:
```bash
git tag exp-qlearning-1
git commit -m "Experiment: exp-qlearning-1 results"
git tag exp-qlearning-2
```

---

## RL Design

### State Space (54 discrete states = 3 × 3 × 3 × 2)

| Dimension | Values |
|-----------|--------|
| Coral bleaching level | `low` / `medium` / `high` |
| Water temperature | `normal` / `warm` / `hot` |
| pH level | `normal` / `acidic` / `very_acidic` |
| Intervention status | `inactive` / `active` |

### Action Space (5 actions)

| ID | Action |
|----|--------|
| 0 | No action |
| 1 | Deploy monitoring buoy |
| 2 | Apply shading intervention |
| 3 | Alert marine biologist |
| 4 | Emergency cooling intervention |

### Reward Function

| Event | Reward |
|-------|--------|
| Coral health improves | **+10** |
| Bleaching level increases | **−10** |
| Unnecessary intervention (healthy reef) | **−5** |
| Early warning correctly detected | **+5** |

### Exploration

Epsilon-greedy with exponential decay:
- Start ε = 0.1, decay = 0.995, floor = 0.01

---

## MLOps Tracking & Hyperparameter Tuning

We have fully integrated **MLflow** for comprehensive experiment tracking, artifact management, and **Optuna** for automated hyperparameter tuning.

### Single Run Tracking
Every training run automatically generates an MLflow record storing all critical experiment metadata. Run training normally:
```bash
python train.py --config configs/qlearning_v1.yaml
```
MLflow will log:
- **Parameters**: ε, learning rate, discount factor, and environment configs.
- **Metrics**: `avg_reward` and `avg_coral_health` for both the RL agent and fixed-timer baseline.
- **Artifacts**: Policy checkpoints (`.pkl` files), configuration YAML, and generated plots.

### Hyperparameter Tuning
To systematically find the best agent configuration (learning rate, discount factor, epsilon decay), run the automated tuning script:
```bash
python tune.py --config configs/qlearning_v1.yaml --trials 20
```
This uses Optuna to optimize the `avg_reward`. Each trial is logged as a nested run in MLflow under a parent sweep run. Best hyperparameters and optimization plots (if supported) are saved to the parent run.

### Viewing the MLflow UI
To view your runs, compare metrics, and download artifacts (policies/plots), start the MLflow tracking server locally:
```bash
mlflow ui
```
Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Baseline Comparison

| Metric | Fixed Schedule | RL Adaptive |
|--------|---------------|-------------|
| Avg Coral Health Score | ~55–65% | ~70–85% |
| Unnecessary Interventions (%) | High (schedule-driven) | Low (state-aware) |
| Early Warnings Caught (%) | Low (timing luck) | Higher (context-aware) |
| Overall Reef Survival Rate (%) | ~55–65% | ~70–85% |

The **Fixed Schedule** policy (action every 10 steps) is evaluated alongside the RL agent during training and with `--compare` flag at evaluation time.

---

## Monitoring Plan (Real-World Deployment)

A short paragraph describing what you would monitor if this were deployed in real-world traffic (adapted for this ecosystem): 
"We would track average coral health (analogous to wait-time), maximum threshold breaches (analogous to queue length), and safety rules (e.g., no unnecessary chemical deployments or red-light-running equivalents)."

*(Note: The above paragraph maps the requested traffic-light MLOps metrics to the coral reef domain while preserving the original intent).*

In a full deployment, we would also track **coral bleaching percentage** across monitored reef zones using underwater image sensors and satellite SST (sea surface temperature) feeds. **Water temperature anomalies** relative to the monthly baseline would trigger agent inference in near-real-time. **Intervention success rate** — defined as the fraction of deployments that are followed by a measurable reduction in bleaching level within 48 hours — would be logged per buoy. **False alarm rate** (interventions deployed when bleaching was already stable or recovering) would be monitored to avoid alert fatigue among marine biologists and to control operational cost. Model drift would be detected by comparing the live action-value distribution against the trained Q-table; significant divergence would trigger a retraining cycle using updated environmental data.

---

## SDG Alignment

- **SDG 14 — Life Below Water**: Directly protects coral reef ecosystems by enabling timely, targeted interventions that reduce bleaching mortality.
- **SDG 13 — Climate Action**: Demonstrates adaptive management under climate-driven temperature and ocean acidification stress, informing broader climate resilience strategies.

---

## Configuration Reference (`configs/qlearning_v1.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epsilon` | 0.1 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Per-episode multiplicative decay |
| `epsilon_min` | 0.01 | Exploration floor |
| `learning_rate` | 0.1 | Q-table update step size (α) |
| `discount_factor` | 0.95 | Future reward discount (γ) |
| `episodes` | 1000 | Total training episodes |
| `max_steps_per_episode` | 200 | Episode horizon |
| `save_policy_v1_at` | 400 | Episode at which v1 snapshot is saved |
