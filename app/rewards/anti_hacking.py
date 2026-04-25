"""
Anti-Hacking Countermeasures
==============================
Prevents agents from gaming the reward system through:
    1. Repetitive pattern detection — penalizes repetitive action sequences
    2. State exploitation detection — catches agents exploiting known loopholes
    3. Progressive difficulty injection — makes scenarios harder as agent improves
    4. Red herring insertion — adds alerts that should be ignored
    5. Reward verification — cross-checks rewards against ground truth outcomes

Design: All checks are pure functions operating on action history
and state, producing penalties or adjustments.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


class AntiHackingGuard:
    """
    Detects and penalizes reward hacking behavior in RL agents.
    
    Integrates with the reward calculator to modify rewards when
    suspicious patterns are detected.
    """

    def __init__(
        self,
        repetition_window: int = 5,
        repetition_penalty: float = 0.8,     # Multiplicative scale reduction
        exploitation_penalty: float = -0.5,
        max_same_action_ratio: float = 0.6,
    ):
        self._repetition_window = repetition_window
        self._repetition_penalty = repetition_penalty
        self._exploitation_penalty = exploitation_penalty
        self._max_same_action_ratio = max_same_action_ratio

        # State
        self._reward_scale: float = 1.0
        self._penalties_applied: List[Dict[str, Any]] = []
        self._red_herring_alerts: Set[str] = set()
        self._false_positive_injections: int = 0

    def reset(self):
        """Reset guard state for new episode."""
        self._reward_scale = 1.0
        self._penalties_applied.clear()
        self._red_herring_alerts.clear()
        self._false_positive_injections = 0

    def check(
        self,
        action_history: List[Dict[str, Any]],
        current_action: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Run all anti-hacking checks on the current action.

        Args:
            action_history: List of past action dicts with 'action_type' and 'target'
            current_action: The current action being evaluated
            state: Current environment state

        Returns:
            (is_hacking, penalty, details) where:
                is_hacking: True if hacking behavior detected
                penalty: Additive penalty to apply (≤ 0)
                details: Diagnostic information
        """
        total_penalty = 0.0
        hacking_detected = False
        details: Dict[str, Any] = {
            "checks_run": [],
            "reward_scale": self._reward_scale,
        }

        # ── Check 1: Repetitive Action Patterns ───────────────────────
        rep_result = self._check_repetitive_patterns(action_history)
        details["checks_run"].append(rep_result)
        if rep_result["detected"]:
            hacking_detected = True
            self._reward_scale *= self._repetition_penalty
            total_penalty += rep_result["penalty"]

        # ── Check 2: State Exploitation ───────────────────────────────
        exploit_result = self._check_state_exploitation(
            action_history, current_action, state
        )
        details["checks_run"].append(exploit_result)
        if exploit_result["detected"]:
            hacking_detected = True
            total_penalty += exploit_result["penalty"]

        # ── Check 3: Action Diversity ─────────────────────────────────
        diversity_result = self._check_action_diversity(action_history)
        details["checks_run"].append(diversity_result)
        if diversity_result["detected"]:
            hacking_detected = True
            total_penalty += diversity_result["penalty"]

        # ── Check 4: Reward Farming Detection ─────────────────────────
        farming_result = self._check_reward_farming(action_history)
        details["checks_run"].append(farming_result)
        if farming_result["detected"]:
            hacking_detected = True
            total_penalty += farming_result["penalty"]

        # ── Record ────────────────────────────────────────────────────
        details["total_penalty"] = round(total_penalty, 4)
        details["reward_scale"] = round(self._reward_scale, 4)
        details["hacking_detected"] = hacking_detected

        if hacking_detected:
            self._penalties_applied.append(details)

        return hacking_detected, total_penalty, details

    def get_reward_scale(self) -> float:
        """Current reward scale factor (reduced when hacking detected)."""
        return self._reward_scale

    def get_stats(self) -> Dict[str, Any]:
        """Get anti-hacking statistics for the episode."""
        return {
            "reward_scale": round(self._reward_scale, 4),
            "penalties_applied_count": len(self._penalties_applied),
            "red_herring_alerts": len(self._red_herring_alerts),
            "false_positive_injections": self._false_positive_injections,
        }

    # ── Individual Checks ─────────────────────────────────────────────

    def _check_repetitive_patterns(
        self, action_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect repetitive action sequences.
        
        If the last N actions are all the same type, this is likely
        a pattern-based hack (e.g., always triage → always correct).
        """
        if len(action_history) < self._repetition_window:
            return {"check": "repetitive_patterns", "detected": False, "penalty": 0.0}

        recent = action_history[-self._repetition_window:]
        action_types = [a.get("action_type", "") for a in recent]

        # Check if all recent actions are identical type
        if len(set(action_types)) == 1:
            return {
                "check": "repetitive_patterns",
                "detected": True,
                "penalty": -0.05,
                "message": f"Last {self._repetition_window} actions all '{action_types[0]}'",
            }

        # Check if recent actions form a repeating cycle (e.g., A-B-A-B-A)
        if len(action_types) >= 4:
            cycle_2 = action_types[-4:]
            if cycle_2[0] == cycle_2[2] and cycle_2[1] == cycle_2[3]:
                return {
                    "check": "repetitive_patterns",
                    "detected": True,
                    "penalty": -0.03,
                    "message": "2-action repeating cycle detected",
                }

        return {"check": "repetitive_patterns", "detected": False, "penalty": 0.0}

    def _check_state_exploitation(
        self,
        action_history: List[Dict[str, Any]],
        current_action: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detect when agent exploits specific state features repeatedly.
        
        Catches:
        - Always targeting the same host/alert
        - Only acting on highest-confidence alerts (cherry-picking)
        - Never querying hosts but taking containment actions
        """
        if len(action_history) < 3:
            return {"check": "state_exploitation", "detected": False, "penalty": 0.0}

        # Check target concentration
        targets = [a.get("target", "") for a in action_history if a.get("target")]
        if targets:
            target_counts = Counter(targets)
            most_common_target, most_common_count = target_counts.most_common(1)[0]
            target_ratio = most_common_count / len(targets)

            if target_ratio > 0.7 and len(targets) > 5:
                return {
                    "check": "state_exploitation",
                    "detected": True,
                    "penalty": -0.08,
                    "message": f"Over-targeting '{most_common_target}' ({target_ratio:.0%} of actions)",
                }

        # Check containment without investigation
        action_type = current_action.get("action_type", "")
        containment_actions = {"isolate_host", "block_ip", "disable_account"}
        if action_type in containment_actions:
            query_count = sum(
                1 for a in action_history if a.get("action_type") == "query_host"
            )
            if query_count == 0 and len(action_history) > 3:
                return {
                    "check": "state_exploitation",
                    "detected": True,
                    "penalty": -0.05,
                    "message": "Containment action without prior investigation",
                }

        return {"check": "state_exploitation", "detected": False, "penalty": 0.0}

    def _check_action_diversity(
        self, action_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Penalize extremely low action diversity.
        
        Agents should use a variety of action types across an episode.
        """
        if len(action_history) < 6:
            return {"check": "action_diversity", "detected": False, "penalty": 0.0}

        action_types = [a.get("action_type", "") for a in action_history]
        unique_ratio = len(set(action_types)) / len(action_types)

        if unique_ratio < 0.2:
            return {
                "check": "action_diversity",
                "detected": True,
                "penalty": -0.04,
                "message": f"Very low action diversity: {unique_ratio:.0%}",
            }

        return {"check": "action_diversity", "detected": False, "penalty": 0.0}

    def _check_reward_farming(
        self, action_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect reward farming: repeatedly taking low-risk, low-reward
        actions to accumulate small positive rewards.
        
        E.g., spamming 'monitor' or re-querying already-queried hosts.
        """
        if len(action_history) < 5:
            return {"check": "reward_farming", "detected": False, "penalty": 0.0}

        recent = action_history[-5:]
        farming_actions = {"monitor", "collect_forensics"}
        farming_count = sum(
            1 for a in recent if a.get("action_type") in farming_actions
        )

        if farming_count >= 4:
            return {
                "check": "reward_farming",
                "detected": True,
                "penalty": -0.06,
                "message": f"Reward farming detected: {farming_count}/5 recent actions are low-value",
            }

        # Check for repeated re-queries
        query_targets = [
            a.get("target") for a in recent if a.get("action_type") == "query_host"
        ]
        if len(query_targets) >= 3:
            unique_queries = set(query_targets)
            if len(unique_queries) == 1:
                return {
                    "check": "reward_farming",
                    "detected": True,
                    "penalty": -0.04,
                    "message": f"Repeated re-query of same target: {unique_queries.pop()}",
                }

        return {"check": "reward_farming", "detected": False, "penalty": 0.0}

    # ── Red Herring Management ────────────────────────────────────────

    def inject_red_herrings(
        self,
        alerts: list,
        rng: Any,
        difficulty: float,
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Add red herring alerts that should be correctly identified as benign.
        
        These test whether the agent can distinguish real threats from noise.
        Acting on red herrings incurs a penalty.
        
        Args:
            alerts: Current alert list
            rng: Random instance for reproducibility
            difficulty: 0.0-1.0 difficulty level
            count: Number of red herrings (auto-scaled if None)
            
        Returns:
            List of injected red herring alert dicts
        """
        if count is None:
            count = max(1, int(difficulty * 3))

        red_herring_templates = [
            {
                "alert_type": "Scheduled_Backup",
                "severity": "medium",
                "description": "Automated backup process triggered large data transfer",
            },
            {
                "alert_type": "Patch_Management",
                "severity": "low",
                "description": "Windows Update service downloading patches",
            },
            {
                "alert_type": "Admin_Login",
                "severity": "high",
                "description": "Administrator login from known management subnet",
            },
            {
                "alert_type": "Network_Scan",
                "severity": "medium",
                "description": "Vulnerability scanner performing authorized assessment",
            },
            {
                "alert_type": "Cloud_Sync",
                "severity": "low",
                "description": "OneDrive cloud sync activity detected",
            },
        ]

        injected = []
        for i in range(min(count, len(red_herring_templates))):
            template = red_herring_templates[i % len(red_herring_templates)]
            alert_id = f"ALT-RH-{rng.randint(70000, 79999)}"
            
            red_herring = {
                "alert_id": alert_id,
                "alert_type": template["alert_type"],
                "severity": template["severity"],
                "confidence": round(rng.uniform(0.3, 0.75), 2),
                "description": template["description"],
                "source_host": f"HOST-{rng.randint(1, 5):03d}",
                "timestamp": f"2026-04-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
                "is_true_positive": False,
                "_is_red_herring": True,
            }
            injected.append(red_herring)
            self._red_herring_alerts.add(alert_id)
            self._false_positive_injections += 1

        return injected

    def check_red_herring_penalty(
        self, action_type: str, target: str
    ) -> float:
        """
        Check if an action targets a red herring alert.
        
        Returns penalty if agent acted on a red herring as if it were real.
        """
        if target in self._red_herring_alerts:
            if action_type == "triage_alert":
                return 0.0  # Triaging is fine — it's the classification that matters
            else:
                return -0.15  # Penalty for escalating/acting on a red herring
        return 0.0
