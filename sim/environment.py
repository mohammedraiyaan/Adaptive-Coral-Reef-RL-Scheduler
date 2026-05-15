"""
Coral Reef RL Environment
State space: 54 discrete states = 3 (bleaching) × 3 (temp) × 3 (pH) × 2 (intervention_active)
Action space: 5 actions
"""
import numpy as np


# --- State dimension maps ---
BLEACHING_LEVELS  = ["low", "medium", "high"]          # 0, 1, 2
TEMP_LEVELS       = ["normal", "warm", "hot"]           # 0, 1, 2
PH_LEVELS         = ["normal", "acidic", "very_acidic"] # 0, 1, 2
INTERVENTION_STATUS = ["inactive", "active"]            # 0, 1

ACTIONS = {
    0: "no_action",
    1: "deploy_monitoring_buoy",
    2: "apply_shading",
    3: "alert_marine_biologist",
    4: "emergency_cooling",
}
N_ACTIONS = len(ACTIONS)

# Derived constants
N_BLEACHING = len(BLEACHING_LEVELS)
N_TEMP      = len(TEMP_LEVELS)
N_PH        = len(PH_LEVELS)
N_INT       = len(INTERVENTION_STATUS)
N_STATES    = N_BLEACHING * N_TEMP * N_PH * N_INT  # 54


def encode_state(bleaching, temp, pH, intervention):
    return bleaching * (N_TEMP * N_PH * N_INT) + temp * (N_PH * N_INT) + pH * N_INT + intervention


def decode_state(state_idx):
    intervention = state_idx % N_INT
    state_idx //= N_INT
    pH = state_idx % N_PH
    state_idx //= N_PH
    temp = state_idx % N_TEMP
    bleaching = state_idx // N_TEMP
    return bleaching, temp, pH, intervention


class CoralReefEnv:
    """
    Stochastic coral reef monitoring environment.

    Transition dynamics
    -------------------
    - Without intervention, temperature & pH drift slowly toward worse states.
    - With an appropriate action the environment has a chance to recover.
    - Actions cost varies: high-impact interventions risk negative side effects
      if applied unnecessarily (healthy reef penalised).

    Parameters
    ----------
    seed : int | None
        Fixed seed for reproducibility.
    stress_factor : float
        Multiplies the baseline probability of conditions worsening.
        Default 1.0  →  normal stress.  >1 = harsher environment.
    """

    def __init__(self, seed: int | None = 42, stress_factor: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.stress_factor = stress_factor
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        # Start in a mildly stressed but not hopeless state
        self.bleaching   = self.rng.integers(0, 2)   # low or medium
        self.temp        = self.rng.integers(0, 2)   # normal or warm
        self.pH          = self.rng.integers(0, 2)   # normal or acidic
        self.intervention_active = 0
        self.step_count  = 0
        self.coral_health = self._compute_health()
        return self._state()

    # ------------------------------------------------------------------
    def _state(self):
        return encode_state(self.bleaching, self.temp, self.pH, self.intervention_active)

    # ------------------------------------------------------------------
    def _compute_health(self) -> float:
        """
        Coral health [0–100] is penalised by each stressor dimension.
        Reflects cumulative ecosystem stress, not a single metric.
        """
        health = 100.0
        health -= self.bleaching * 20   # 0 / 20 / 40
        health -= self.temp      * 10   # 0 / 10 / 20
        health -= self.pH        * 10   # 0 / 10 / 20
        return max(0.0, health)

    # ------------------------------------------------------------------
    def step(self, action: int):
        """Returns (next_state, reward, done, info)."""
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"
        self.step_count += 1

        prev_bleaching = self.bleaching
        reward = 0.0

        # ---- Action effects -------------------------------------------
        if action == 0:  # no_action — environment drifts freely
            pass

        elif action == 1:  # deploy_monitoring_buoy — mild benefit
            # Gives +5 if bleaching is medium or high (early warning)
            if self.bleaching >= 1:
                reward += 5.0

        elif action == 2:  # apply_shading — reduces temp stress
            if self.temp >= 1:
                # Cooling effect: chance to reduce temp
                if self.rng.random() < 0.55:
                    self.temp = max(0, self.temp - 1)
                reward += 3.0
            else:
                # Unnecessary shading on normal temp
                reward -= 5.0

        elif action == 3:  # alert_marine_biologist — triggers coordinated response
            if self.bleaching >= 1 or self.pH >= 1:
                reward += 4.0
                self.intervention_active = 1
            else:
                reward -= 5.0  # False alarm cost

        elif action == 4:  # emergency_cooling — high impact, high risk
            if self.bleaching == 2 or self.temp == 2:
                if self.rng.random() < 0.65:
                    self.bleaching = max(0, self.bleaching - 1)
                    self.temp      = max(0, self.temp - 1)
                reward += 8.0
            else:
                # Chemical disturbance on healthy reef
                reward -= 8.0
                self.pH = min(N_PH - 1, self.pH + 1)  # Side effect: acidification

        # ---- Stochastic environment drift (climate stress) ------------
        sf = self.stress_factor

        # Temperature can warm up stochastically
        if self.rng.random() < 0.08 * sf:
            self.temp = min(N_TEMP - 1, self.temp + 1)
        # ... or recover slightly
        elif self.rng.random() < 0.04:
            self.temp = max(0, self.temp - 1)

        # pH acidification drift
        if self.rng.random() < 0.06 * sf:
            self.pH = min(N_PH - 1, self.pH + 1)
        elif self.rng.random() < 0.03:
            self.pH = max(0, self.pH - 1)

        # Bleaching driven by temp and pH
        bleach_prob = (0.04 + 0.06 * self.temp + 0.05 * self.pH) * sf
        if self.rng.random() < bleach_prob:
            self.bleaching = min(N_BLEACHING - 1, self.bleaching + 1)
        recovery_prob = 0.05 if self.intervention_active else 0.02
        if self.rng.random() < recovery_prob:
            self.bleaching = max(0, self.bleaching - 1)

        # Intervention status fades after ~10 steps
        if self.intervention_active and self.rng.random() < 0.10:
            self.intervention_active = 0

        # ---- Reward from health change ---------------------------------
        new_health = self._compute_health()
        health_delta = new_health - self.coral_health
        if health_delta > 0:
            reward += 10.0   # health improved
        elif health_delta < 0:
            reward -= 10.0   # health worsened

        self.coral_health = new_health

        done = self.step_count >= 200  # episode horizon

        info = {
            "bleaching":  BLEACHING_LEVELS[self.bleaching],
            "temp":       TEMP_LEVELS[self.temp],
            "pH":         PH_LEVELS[self.pH],
            "intervention_active": bool(self.intervention_active),
            "coral_health": self.coral_health,
            "action_name": ACTIONS[action],
        }
        return self._state(), reward, done, info