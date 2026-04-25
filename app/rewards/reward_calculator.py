"""
Multi-Component Sparse Reward Calculator
==========================================
Replaces the single dense reward with a structured, multi-component approach
designed to prevent reward hacking while providing informative learning signals.

Components:
    1. Action Correctness (0.40)  — Sparse, only for truly correct actions
    2. Explanation Quality (0.30) — Separate evaluation of reasoning quality
    3. Time Efficiency    (0.10)  — Pressure to act quickly
    4. Prevention Bonus   (0.10)  — Rewarding early threat containment
    5. Investigation Depth(0.10)  — Rewarding proper investigation before action

Design Principles:
    - Sparse rewards: hard to achieve, meaningful when earned
    - Component isolation: agents can't hack one to compensate for another
    - Randomized weights: small jitter prevents overfitting to exact weights
    - Clamped output: always in [-1.0, 1.0]
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional, Tuple


class MultiComponentRewardCalculator:
    """
    Calculates multi-component sparse rewards for cybersecurity IR actions.

    Each component is scored independently and combined with configurable
    (and optionally jittered) weights to prevent reward hacking.
    """

    # Base component weights — sum to 1.0
    DEFAULT_WEIGHTS = {
        "action_correctness":   0.40,
        "explanation_quality":  0.30,
        "time_efficiency":      0.10,
        "prevention_bonus":     0.10,
        "investigation_depth":  0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        jitter_enabled: bool = True,
        jitter_magnitude: float = 0.03,
        seed: Optional[int] = None,
    ):
        """
        Args:
            weights: Override default component weights.
            jitter_enabled: Add small random jitter to weights each step.
            jitter_magnitude: Max jitter per weight (default ±3%).
            seed: RNG seed for reproducible jitter.
        """
        self._base_weights = dict(weights or self.DEFAULT_WEIGHTS)
        self._jitter_enabled = jitter_enabled
        self._jitter_magnitude = jitter_magnitude
        self._rng = _random.Random(seed)

        # Tracking
        self._step_count = 0
        self._component_history: List[Dict[str, float]] = []
        self._cumulative_components = {k: 0.0 for k in self._base_weights}

    def reset(self, seed: Optional[int] = None):
        """Reset calculator state for a new episode."""
        self._step_count = 0
        self._component_history.clear()
        self._cumulative_components = {k: 0.0 for k in self._base_weights}
        if seed is not None:
            self._rng = _random.Random(seed)

    def calculate(
        self,
        action_result: Dict[str, Any],
        xai_scores: Dict[str, float],
        step: int,
        max_steps: int,
        threat_contained: bool,
        containment_step: Optional[int],
        queried_hosts: int,
        total_hosts: int,
        action_type: str,
        action_history: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the multi-component reward.

        Returns:
            (final_reward, component_breakdown) where final_reward ∈ [-1.0, 1.0]
        """
        self._step_count += 1

        # ── 1. Action Correctness (sparse) ─────────────────────────────
        correctness = action_result.get("correctness", 0.0)
        if correctness >= 0.7:
            action_score = 0.4 + (correctness - 0.7) * 0.5  # 0.4 → 0.55
        elif correctness >= 0.5:
            action_score = 0.2
        elif correctness > 0.0:
            action_score = 0.05
        else:
            action_score = -0.2  # Penalty for completely wrong actions

        # Extra penalty for harmful actions
        if action_result.get("harmful", False):
            action_score -= 0.15

        # ── 2. Explanation Quality ─────────────────────────────────────
        explanation_score = xai_scores.get("explanation_quality", 0.0)

        # ── 3. Time Efficiency ─────────────────────────────────────────
        progress_ratio = step / max(max_steps, 1)
        time_factor = max(0.0, 1.0 - progress_ratio * 0.5)

        # Urgency penalty in final 20% of episode
        if progress_ratio > 0.8:
            time_factor *= 0.7

        # ── 4. Prevention Bonus (sparse — only when threat contained) ──
        prevention_bonus = 0.0
        if threat_contained and containment_step is not None:
            # Earlier containment = higher bonus
            containment_progress = containment_step / max(max_steps, 1)
            prevention_bonus = 0.3 * (1.0 - containment_progress)

        # ── 5. Investigation Depth ─────────────────────────────────────
        investigation_score = 0.0
        if total_hosts > 0:
            query_coverage = queried_hosts / total_hosts
            investigation_score = min(1.0, query_coverage * 1.5)

            # Bonus for investigating BEFORE containment actions
            containment_actions = {"isolate_host", "block_ip", "disable_account"}
            if action_type in containment_actions and query_coverage < 0.3:
                investigation_score *= 0.5  # Penalty for acting without investigation

        # ── Step Cost ──────────────────────────────────────────────────
        step_cost = -0.01  # Small per-step cost to discourage stalling

        # ── Combine with (optionally jittered) weights ─────────────────
        weights = self._get_weights()

        components = {
            "action_correctness":   action_score,
            "explanation_quality":  explanation_score,
            "time_efficiency":      time_factor,
            "prevention_bonus":     prevention_bonus,
            "investigation_depth":  investigation_score,
        }

        weighted_sum = sum(
            weights[k] * components[k] for k in components
        )
        final_reward = max(-1.0, min(1.0, weighted_sum + step_cost))

        # ── Track ──────────────────────────────────────────────────────
        breakdown = {
            **components,
            "step_cost":       step_cost,
            "weights_used":    weights,
            "weighted_sum":    round(weighted_sum, 4),
            "final_reward":    round(final_reward, 4),
        }
        self._component_history.append(breakdown)
        for k, v in components.items():
            self._cumulative_components[k] += v

        return final_reward, breakdown

    def get_summary(self) -> Dict[str, float]:
        """Get cumulative component averages for the episode."""
        steps = max(self._step_count, 1)
        return {
            k: round(v / steps, 4)
            for k, v in self._cumulative_components.items()
        }

    def _get_weights(self) -> Dict[str, float]:
        """Return weights with optional jitter for anti-hacking."""
        if not self._jitter_enabled:
            return dict(self._base_weights)

        jittered = {}
        total = 0.0
        for k, v in self._base_weights.items():
            jitter = self._rng.uniform(
                -self._jitter_magnitude, self._jitter_magnitude
            )
            jittered[k] = max(0.01, v + jitter)
            total += jittered[k]

        # Renormalize to sum to 1.0
        for k in jittered:
            jittered[k] = round(jittered[k] / total, 4)

        return jittered
