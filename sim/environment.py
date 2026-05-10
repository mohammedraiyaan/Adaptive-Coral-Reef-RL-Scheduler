"""
sim/environment.py
==================
Coral Reef Monitoring & Intervention Environment

State Space (discrete, 3×3×3×2 = 54 unique states):
  - Coral bleaching level : 0=low  | 1=medium | 2=high
  - Water temperature      : 0=normal | 1=warm  | 2=hot
  - pH level               : 0=normal | 1=acidic | 2=very_acidic
  - Intervention status    : 0=inactive | 1=active

Action Space (5 discrete actions):
  0 - No action
  1 - Deploy monitoring buoy
  2 - Apply shading intervention
  3 - Alert marine biologist
  4 - Emergency cooling intervention

Reward Design:
  +10  coral health improves
  -10  bleaching level increases
   -5  unnecessary intervention deployed
   +5  early warning detected correctly (warm/acidic → alert before crisis)
"""

import random
import numpy as np
from typing import Tuple, Dict, Any


# --------------------------------------------------------------------------- #
#  Constants & Mappings                                                         #
# --------------------------------------------------------------------------- #
BLEACHING_LEVELS   = {0: "low",       1: "medium",     2: "high"}
TEMPERATURE_LEVELS = {0: "normal",    1: "warm",       2: "hot"}
PH_LEVELS          = {0: "normal",    1: "acidic",     2: "very_acidic"}
INTERVENTION_STATE = {0: "inactive",  1: "active"}

ACTIONS = {
    0: "no_action",
    1: "deploy_monitoring_buoy",
    2: "apply_shading",
    3: "alert_marine_biologist",
    4: "emergency_cooling",
}

N_BLEACH   = len(BLEACHING_LEVELS)    # 3
N_TEMP     = len(TEMPERATURE_LEVELS)  # 3
N_PH       = len(PH_LEVELS)           # 3
N_INTERV   = len(INTERVENTION_STATE)  # 2
N_STATES   = N_BLEACH * N_TEMP * N_PH * N_INTERV  # 54
N_ACTIONS  = len(ACTIONS)             # 5

# Intervention actions (non-zero actions are "active" interventions)
INTERVENTION_ACTIONS = {1, 2, 3, 4}

# Cooling / shading actions that can directly help
ACTIVE_INTERVENTIONS = {2, 4}   # shading, emergency cooling
ALERT_ACTIONS        = {1, 3}   # monitoring buoy, alert biologist


# --------------------------------------------------------------------------- #
#  State Encoding / Decoding                                                   #
# --------------------------------------------------------------------------- #
def encode_state(bleach: int, temp: int, ph: int, interv: int) -> int:
    """Encode 4-component state into a single integer index."""
    return bleach * (N_TEMP * N_PH * N_INTERV) + temp * (N_PH * N_INTERV) + ph * N_INTERV + interv


def decode_state(state_idx: int) -> Tuple[int, int, int, int]:
    """Decode a state index into its 4 components."""
    interv = state_idx % N_INTERV
    state_idx //= N_INTERV
    ph = state_idx % N_PH
    state_idx //= N_PH
    temp = state_idx % N_TEMP
    bleach = state_idx // N_TEMP
    return bleach, temp, ph, interv


# --------------------------------------------------------------------------- #
#  Environment                                                                  #
# --------------------------------------------------------------------------- #
class CoralReefEnv:
    """
    Tabular RL environment simulating a coral reef ecosystem.

    The reef's health degrades stochastically based on temperature and pH
    anomalies.  Agents must intervene intelligently to minimise bleaching
    while avoiding spurious (costly) interventions.
    """

    # ---------------------------------------------------------------------- #
    #  Construction & Reset                                                    #
    # ---------------------------------------------------------------------- #
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Internal state components
        self.bleach:  int = 0
        self.temp:    int = 0
        self.ph:      int = 0
        self.interv:  int = 0

        # Episode bookkeeping
        self.step_count:           int   = 0
        self.coral_health:         float = 1.0   # 0.0 (dead) → 1.0 (pristine)
        self.interventions_made:   int   = 0
        self.early_warnings_total: int   = 0
        self.early_warnings_caught: int  = 0

    def reset(self) -> int:
        """Reset to a random initial state.  Returns encoded state index."""
        self.bleach  = self.rng.randint(0, 1)   # start low or medium
        self.temp    = self.rng.randint(0, 1)   # start normal or warm
        self.ph      = self.rng.randint(0, 1)
        self.interv  = 0                        # always start inactive

        self.step_count             = 0
        self.coral_health           = 1.0 - self.bleach * 0.3
        self.interventions_made     = 0
        self.early_warnings_total   = 0
        self.early_warnings_caught  = 0

        return self._encoded_state()

    # ---------------------------------------------------------------------- #
    #  Step                                                                    #
    # ---------------------------------------------------------------------- #
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Parameters
        ----------
        action : int  — index from ACTIONS dict

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        prev_bleach = self.bleach
        reward      = 0.0

        # ------------------------------------------------------------------ #
        #  1. Environment dynamics — stochastic transitions                    #
        # ------------------------------------------------------------------ #
        self._advance_environment()

        # ------------------------------------------------------------------ #
        #  2. Track early-warning opportunity                                  #
        # ------------------------------------------------------------------ #
        # An "early warning window" exists when temp=warm AND bleach<high
        # A correct early warning = sending alert BEFORE crisis (bleach==2)
        early_warning_opportunity = (self.temp >= 1) and (prev_bleach < 2)
        if early_warning_opportunity:
            self.early_warnings_total += 1

        # ------------------------------------------------------------------ #
        #  3. Compute reward                                                   #
        # ------------------------------------------------------------------ #
        reward, did_early_warn = self._compute_reward(
            action, prev_bleach, early_warning_opportunity
        )
        if did_early_warn:
            self.early_warnings_caught += 1

        # ------------------------------------------------------------------ #
        #  4. Apply action effects on reef state                               #
        # ------------------------------------------------------------------ #
        self._apply_action_effects(action)

        # ------------------------------------------------------------------ #
        #  5. Update coral health score                                        #
        # ------------------------------------------------------------------ #
        #  Health is a smooth proxy: decreases with bleaching, improves slowly #
        health_target = 1.0 - self.bleach * 0.35
        self.coral_health += 0.1 * (health_target - self.coral_health)
        self.coral_health  = float(np.clip(self.coral_health, 0.0, 1.0))

        if action in INTERVENTION_ACTIONS:
            self.interventions_made += 1
            self.interv = 1
        else:
            self.interv = 0

        # ------------------------------------------------------------------ #
        #  6. Termination condition                                            #
        # ------------------------------------------------------------------ #
        self.step_count += 1
        done = (self.coral_health <= 0.0) or (self.bleach >= 2 and
               self.temp >= 2 and self.ph >= 2)

        info = {
            "bleach":            self.bleach,
            "temp":              self.temp,
            "ph":                self.ph,
            "interv":            self.interv,
            "coral_health":      self.coral_health,
            "interventions":     self.interventions_made,
            "early_warn_total":  self.early_warnings_total,
            "early_warn_caught": self.early_warnings_caught,
            "action_name":       ACTIONS[action],
        }

        return self._encoded_state(), reward, done, info

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                         #
    # ---------------------------------------------------------------------- #
    def _encoded_state(self) -> int:
        return encode_state(self.bleach, self.temp, self.ph, self.interv)

    def _advance_environment(self) -> None:
        """Stochastic transition of temperature, pH, and bleaching."""
        # Temperature — drifts up with 20% prob per step, recovers with 10%
        r = self.rng.random()
        if r < 0.20 and self.temp < 2:
            self.temp += 1
        elif r > 0.90 and self.temp > 0:
            self.temp -= 1

        # pH — acidification: rises with 15% prob when temp is elevated
        r = self.rng.random()
        acidify_prob = 0.10 + 0.05 * self.temp
        if r < acidify_prob and self.ph < 2:
            self.ph += 1
        elif r > 0.92 and self.ph > 0:
            self.ph -= 1

        # Bleaching — driven by temperature + pH stress
        stress = (self.temp + self.ph) / 4.0          # 0.0 → 1.0
        r = self.rng.random()
        bleach_up_prob   = stress * 0.40
        bleach_down_prob = (1.0 - stress) * 0.15
        if r < bleach_up_prob and self.bleach < 2:
            self.bleach += 1
        elif r < bleach_up_prob + bleach_down_prob and self.bleach > 0:
            self.bleach -= 1

    def _compute_reward(
        self,
        action: int,
        prev_bleach: int,
        early_warning_opportunity: bool,
    ) -> Tuple[float, bool]:
        """Return (reward, did_early_warn)."""
        reward = 0.0
        did_early_warn = False

        # Bleaching got worse → punish
        if self.bleach > prev_bleach:
            reward -= 10.0

        # Bleaching improved → reward
        elif self.bleach < prev_bleach:
            reward += 10.0

        # Early-warning bonus: alert/buoy deployed before crisis
        if early_warning_opportunity and action in ALERT_ACTIONS and prev_bleach < 2:
            reward += 5.0
            did_early_warn = True

        # Unnecessary intervention: intervening when reef is healthy
        if action in INTERVENTION_ACTIONS and self.bleach == 0 and self.temp == 0:
            reward -= 5.0

        return reward, did_early_warn

    def _apply_action_effects(self, action: int) -> None:
        """Actions can nudge the environment state beneficially."""
        if action == 2:   # shading — reduces temperature stress
            if self.rng.random() < 0.40 and self.temp > 0:
                self.temp -= 1
        elif action == 4:  # emergency cooling — strong temp reduction
            if self.rng.random() < 0.70 and self.temp > 0:
                self.temp -= 1
                if self.rng.random() < 0.30 and self.bleach > 0:
                    self.bleach -= 1

    # ---------------------------------------------------------------------- #
    #  Properties                                                               #
    # ---------------------------------------------------------------------- #
    @property
    def observation_space_n(self) -> int:
        return N_STATES

    @property
    def action_space_n(self) -> int:
        return N_ACTIONS

    def state_labels(self) -> Dict[str, str]:
        return {
            "bleach":  BLEACHING_LEVELS[self.bleach],
            "temp":    TEMPERATURE_LEVELS[self.temp],
            "ph":      PH_LEVELS[self.ph],
            "interv":  INTERVENTION_STATE[self.interv],
        }

    def __repr__(self) -> str:
        lbl = self.state_labels()
        return (
            f"CoralReefEnv("
            f"bleach={lbl['bleach']}, temp={lbl['temp']}, "
            f"ph={lbl['ph']}, interv={lbl['interv']}, "
            f"health={self.coral_health:.2f})"
        )
