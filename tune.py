"""
tune.py  —  Hyperparameter tuning with Optuna and MLflow
======================================================
Usage:
    python tune.py --config configs/qlearning_v1.yaml --trials 20
"""

import argparse
import mlflow
import optuna
import uuid
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import load_config, make_dirs, train_qlearning

def objective(trial, base_cfg):
    # Sample hyperparameters
    cfg = base_cfg.copy()
    
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    cfg["discount_factor"] = trial.suggest_float("discount_factor", 0.8, 0.99)
    cfg["epsilon_decay"] = trial.suggest_float("epsilon_decay", 0.9, 0.999)
    
    # We shouldn't overwrite the global snapshot locations simultaneously in parallel (if n_jobs>1)
    # But since it's sequential, it's fine. We'll disable saving the policy during tuning to save time.
    cfg["save_policy_v1_at"] = -1 # disable
    
    run_hash = uuid.uuid4().hex
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as run:
        mlflow.log_params({
            "learning_rate": cfg["learning_rate"],
            "discount_factor": cfg["discount_factor"],
            "epsilon_decay": cfg["epsilon_decay"],
            "episodes": cfg["episodes"]
        })
        
        # Train Q-learning
        rl_results = train_qlearning(cfg)
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "rl_avg_reward": rl_results['avg_reward'],
            "rl_avg_health": rl_results['avg_health'],
        })
        
        return rl_results['avg_reward']

def main():
    parser = argparse.ArgumentParser(description="Tune Hyperparameters")
    parser.add_argument("--config", default="configs/qlearning_v1.yaml")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    make_dirs(cfg)

    experiment_name = cfg.get("name", "coral_reef_qlearning")
    mlflow.set_experiment(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"  Hyperparameter Tuning (Optuna + MLflow)")
    print(f"  Trials: {args.trials}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name="optuna_sweep") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, cfg), n_trials=args.trials)

        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Avg Reward): {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        mlflow.log_params({"best_" + k: v for k, v in trial.params.items()})
        mlflow.log_metric("best_reward", trial.value)
        
        # Save the optuna study visualizations
        try:
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
            
            fig_hist = plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(f"{cfg['plots_dir']}/optuna_history.png", dpi=150)
            mlflow.log_artifact(f"{cfg['plots_dir']}/optuna_history.png")
            plt.close()
            
            fig_imp = plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(f"{cfg['plots_dir']}/optuna_importances.png", dpi=150)
            mlflow.log_artifact(f"{cfg['plots_dir']}/optuna_importances.png")
            plt.close()
        except ImportError:
            print("Could not plot optuna graphs. Make sure scikit-learn is installed for param importances.")
            
    print("\n✓ Tuning complete. Best parameters logged to MLflow parent run.")

if __name__ == "__main__":
    main()
