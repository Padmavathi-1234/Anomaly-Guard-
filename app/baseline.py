"""
AnomalyGuard — Baseline Agent Evaluation

Provides RandomAgent and RuleBasedAgent for benchmarking.
These establish lower bounds that trained RL agents must exceed.

Design notes:
    - Agents receive Observation model objects (not dicts)
    - Actions use Action model (not raw strings)
    - step() returns 5-tuple (obs, reward, terminated, truncated, info)
    - Baselines do NOT use justification (worst case for explanation scoring)
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .models import Action, Observation, NetworkHost, SIEMAlert


# ═══════════════════════════════════════════════════════════════════
# Baseline Agents
# ═══════════════════════════════════════════════════════════════════

class RandomAgent:
    """
    Completely random action selection.
    Establishes absolute lower bound - any trained agent must beat this.

    Note: Does not provide justifications, so explanation score = 0.0
    This means max possible score ≈ 0.60 (action only) even if correct.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def choose_action(self, obs: Observation) -> Action:
        """Choose random valid action from observation."""
        action_types = obs.available_actions if obs.available_actions else ["monitor"]

        action_type = self.rng.choice(action_types)

        # Generate valid target for action
        target = self._get_random_target(action_type, obs)

        params = {}
        if action_type == "triage_alert":
            params["classification"] = self.rng.choice(
                ["true_positive", "false_positive"]
            )

        return Action(
            action_type=action_type,
            target=target,
            parameters=params,
            justification=None,
        )

    def _get_random_target(self, action_type: str, obs: Observation) -> str:
        """Get a random valid target for the action type."""
        if action_type in ("triage_alert",) and obs.alerts:
            alert = self.rng.choice(obs.alerts)
            return alert.alert_id

        if action_type in (
            "isolate_host", "query_host", "restore_host",
            "collect_forensics",
        ) and obs.hosts:
            host = self.rng.choice(obs.hosts)
            return host.host_id

        if action_type == "block_ip" and obs.threat_intel:
            ips = obs.threat_intel.malicious_ips
            if ips:
                return self.rng.choice(ips)

        if action_type == "disable_account" and obs.hosts:
            all_accounts: List[str] = []
            for h in obs.hosts:
                all_accounts.extend(h.accounts)
            if all_accounts:
                return self.rng.choice(all_accounts)

        if action_type == "patch_vulnerability" and obs.hosts:
            all_cves: List[str] = []
            for h in obs.hosts:
                all_cves.extend(h.vulnerabilities)
            if all_cves:
                return self.rng.choice(all_cves)

        if action_type == "remove_persistence" and obs.hosts:
            all_pers: List[str] = []
            for h in obs.hosts:
                all_pers.extend(h.persistence)
            if all_pers:
                return self.rng.choice(all_pers)

        return ""


class RuleBasedAgent:
    """
    Heuristic-driven agent using simple priority rules.

    Priority order:
        1. Query unqueried hosts (gather intelligence first)
        2. Triage critical alerts (detection phase)
        3. Isolate compromised hosts (containment phase)
        4. Remove persistence (eradication phase)
        5. Restore isolated hosts (recovery phase)
        6. Triage remaining alerts
        7. Monitor (fallback)

    Note: Sees only what partial observability allows - must query first.
    Does not provide justifications, so explanation score = 0.0
    """

    def choose_action(self, obs: Observation) -> Action:
        """Choose action based on heuristic priority rules."""

        available = set(obs.available_actions) if obs.available_actions else {"monitor"}

        # Priority 1: Query unqueried hosts to reveal compromise status
        if "query_host" in available:
            unqueried = [h for h in obs.hosts if not getattr(h, "is_queried", False)]
            if unqueried:
                # Prioritize servers and domain controllers
                priority_roles = {"domain_controller", "server", "database"}
                priority_hosts = [h for h in unqueried if h.role in priority_roles]
                target_host = priority_hosts[0] if priority_hosts else unqueried[0]
                return Action(
                    action_type="query_host",
                    target=target_host.host_id,
                    parameters={},
                    justification=None,
                )

        # Priority 2: Triage critical/high severity alerts first
        if "triage_alert" in available:
            untriaged = [a for a in obs.alerts if not getattr(a, "agent_classification", None)]
            critical = [a for a in untriaged if a.severity in ("critical", "high")]
            if critical:
                alert = critical[0]
                # Heuristic: high confidence + MITRE mapping = true positive
                classification = (
                    "true_positive"
                    if (getattr(alert, "confidence", 0) > 0.6
                        or getattr(alert, "mitre_technique", None) is not None)
                    else "false_positive"
                )
                return Action(
                    action_type="triage_alert",
                    target=alert.alert_id,
                    parameters={"classification": classification},
                    justification=None,
                )

        # Priority 3: Isolate hosts that show compromise signs
        if "isolate_host" in available:
            compromised = [
                h for h in obs.hosts
                if (getattr(h, "is_queried", False)
                    and (getattr(h, "c2_active", False)
                         or bool(getattr(h, "persistence", []))))
                and h.host_id not in (
                    {h2.host_id for h2 in obs.hosts if getattr(h2, "status", "") == "isolated"}
                )
            ]
            if compromised:
                # Isolate highest criticality first
                criticality_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                compromised.sort(
                    key=lambda h: criticality_order.get(getattr(h, "criticality", "low"), 3)
                )
                return Action(
                    action_type="isolate_host",
                    target=compromised[0].host_id,
                    parameters={},
                    justification=None,
                )

        # Priority 4: Remove persistence mechanisms
        if "remove_persistence" in available:
            for h in obs.hosts:
                if getattr(h, "is_queried", False) and getattr(h, "persistence", []):
                    return Action(
                        action_type="remove_persistence",
                        target=h.persistence[0],
                        parameters={},
                        justification=None,
                    )

        # Priority 5: Restore isolated hosts (after eradication)
        if "restore_host" in available:
            isolated = [
                h for h in obs.hosts
                if getattr(h, "status", "") == "isolated"
                and not getattr(h, "persistence", [])
                and not getattr(h, "vulnerabilities", [])
            ]
            if isolated:
                return Action(
                    action_type="restore_host",
                    target=isolated[0].host_id,
                    parameters={},
                    justification=None,
                )

        # Priority 6: Triage remaining medium/low alerts
        if "triage_alert" in available:
            untriaged = [a for a in obs.alerts if not getattr(a, "agent_classification", None)]
            if untriaged:
                alert = untriaged[0]
                classification = (
                    "true_positive"
                    if getattr(alert, "confidence", 0) > 0.5
                    else "false_positive"
                )
                return Action(
                    action_type="triage_alert",
                    target=alert.alert_id,
                    parameters={"classification": classification},
                    justification=None,
                )

        # Fallback: monitor
        return Action(
            action_type="monitor",
            target="",
            parameters={},
            justification=None,
        )


# ═══════════════════════════════════════════════════════════════════
# Evaluation Runner
# ═══════════════════════════════════════════════════════════════════

def run_rule_based_baseline(
    task_id: int = 1,
    seed: int = 42,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run rule-based agent for one episode and return performance metrics.

    This is the existing endpoint called by POST /baseline.
    Kept for backward compatibility.

    Args:
        task_id: Task to evaluate (1, 2, or 3)
        seed: Random seed for reproducibility
        env: AnomalyGuardEnvironment instance to use

    Returns:
        Dict with scores and episode summary
    """
    if env is None:
        from .environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()

    agent = RuleBasedAgent()

    # Run one episode
    obs, info = env.reset(task_id=task_id, seed=seed)

    total_reward = 0.0
    steps = 0
    rewards = []

    done = False
    truncated = False

    while not (done or truncated):
        action = agent.choose_action(obs)
        obs, reward, done, truncated, step_info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        steps += 1

    # Get final grade
    from .grader import grade_episode
    grade = grade_episode(env._state, task_id)

    return {
        "agent":           "rule_based",
        "task_id":         task_id,
        "seed":            seed,
        "steps":           steps,
        "total_reward":    round(total_reward, 4),
        "avg_reward":      round(total_reward / max(steps, 1), 4),
        "final_score":     grade.final_score,
        "termination":     step_info.get("termination_reason", "unknown"),
        "grade_details": {
            "action_correctness":  grade.action_correctness,
            "explanation_quality": grade.explanation_quality,
            "threats_detected":    grade.threats_detected,
            "threats_missed":      grade.threats_missed,
            "containment_rate":    grade.containment_rate,
        },
    }
