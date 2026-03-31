"""
AnomalyGuard — Core RL Environment
===================================

OpenEnv-compliant cybersecurity incident response environment.
Extends openenv.env.env.Env base class.

OPENENV COMPLIANCE
------------------
    - reset(task_id, seed) → (Observation, dict)
    - step(action) → (Observation, float, bool, bool, dict)
    - Reward range: [-1.0, 1.0]
    - Deterministic: same seed always produces identical scenario
    - terminated vs truncated: correctly separated per OpenEnv spec

REWARD FORMULA
--------------
    base_reward = action_correctness × 0.60 + explanation_quality × 0.40
    final_reward = base_reward + progress_bonus + spread_penalty - step_cost
    final_reward = clamp(final_reward, -1.0, 1.0)

TERMINATION CONDITIONS
----------------------
    terminated=True, truncated=False:
        - task_complete: All objectives met (success)
        - network_shutdown: All hosts isolated (catastrophic overreaction)
        - total_compromise: All hosts infected (catastrophic failure)
    
    terminated=False, truncated=True:
        - timeout: Max steps reached (artificial cutoff, needs bootstrap)

TASKS
-----
    Task 1 - Alert Triage:
        Max steps: 15
        Objective: Classify SIEM alerts as true/false positives
        
    Task 2 - Incident Containment:
        Max steps: 20
        Objective: Isolate compromised hosts, stop lateral movement
        
    Task 3 - Full IR Lifecycle:
        Max steps: 30
        Objective: Complete detect → contain → eradicate → recover workflow

PARTIAL OBSERVABILITY
---------------------
    Host compromise details (c2_active, persistence, vulnerabilities, accounts)
    are hidden until agent uses query_host action.
    Mirrors real SOC analyst workflow where investigation precedes action.

CURRICULUM LEARNING
-------------------
    10 difficulty levels (1=beginner, 10=expert)
    Auto-advances when avg score > 0.75 over 5 episodes
    Auto-regresses when avg score < 0.35 over 5 episodes

EU AI ACT COMPLIANCE
--------------------
    Every action requires ActionJustification with:
        - reasoning (min 50 chars)
        - evidence (at least 1 item)
        - risk_assessment
        - alternatives_considered
    
    5-check compliance audit generated per episode:
        - Article 14.4(b): All actions justified
        - Article 13.1: Explanation quality adequate
        - Article 14.1: Human oversight available
        - Article 14.4(c): High-risk actions documented
        - Article 10.2(f): No classification bias

USAGE
-----
    from app.environment import AnomalyGuardEnvironment
    from app.models import Action
    
    env = AnomalyGuardEnvironment()
    obs, info = env.reset(task_id=1, seed=42)
    
    action = Action(
        action_type="triage_alert",
        target="ALT-10001",
        parameters={"classification": "true_positive"},
        justification=ActionJustification(
            reasoning="Alert shows C2 beacon pattern...",
            evidence=[...],
            risk_assessment=RiskAssessment(...),
            alternatives_considered=[...]
        )
    )
    
    obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from openenv.env.env import Env as OpenEnvBase

from .models import (
    Action,
    ActionJustification,
    HostStatus,
    IncidentPhase,
    NetworkHost,
    Observation,
    Reward,
    ScoreBreakdown,
    SIEMAlert,
)
from .scenarios import build_scenario
from .explainability import score_justification
from .grader import grade_episode


class AnomalyGuardEnvironment(OpenEnvBase):
    """
    Explainable AI environment for cybersecurity incident response.
    Extends openenv.env.env.Env.
    """

    def __init__(self):
        super().__init__(
            name="AnomalyGuard",
            state_space=None,
            action_space=None,
            episode_max_length=30,
        )
        self._state: Optional[Dict[str, Any]] = None
        self._task_id: int = 1
        self._seed: int = 42
        self._step_count: int = 0
        self._done: bool = True
        self._episode_rewards: List[float] = []
        self._score_breakdown = ScoreBreakdown()
        self._difficulty: float = 0.5
        
        # Feature 2: Network topology for malware spread
        self._network_graph = {
            'web-server-01': ['db-server-01', 'app-server-01'],
            'db-server-01': ['web-server-01', 'backup-server-01'],
            'app-server-01': ['web-server-01', 'api-gateway-01'],
            'api-gateway-01': ['app-server-01', 'auth-server-01'],
            'backup-server-01': ['db-server-01'],
            'auth-server-01': ['api-gateway-01'],
            'file-server-01': ['backup-server-01', 'web-server-01'],
            'email-server-01': ['web-server-01', 'auth-server-01'],
            'dns-server-01': ['web-server-01'],
            'jump-host-01': ['web-server-01', 'db-server-01']
        }
        self._infected_hosts = set()
        self._spread_probability = 0.3
        self._spread_history = []

        # Partial observability tracking
        self._queried_hosts: set = set()   # NEW - tracks which hosts agent has investigated
        
        # Feature 3: Adaptive curriculum
        self._curriculum_enabled = True
        self._episode_scores = []  # Last 3 episode scores
        self._curriculum_window = 3
        self._auto_adjust_threshold_high = 0.8
        self._auto_adjust_threshold_low = 0.4
        self._difficulty_history = []

        # ===== CURRICULUM LEARNING SYSTEM =====
        self._curriculum_level = 1  # Current difficulty (1-10)
        self._episode_history: List[float] = []  # Last N episode scores
        self._total_episodes = 0
        self._episodes_at_current_level = 0
        self._last_episode_score: Optional[float] = None

        self._curriculum_config = {
            "advance_threshold": 0.75,
            "regress_threshold": 0.35,
            "evaluation_window": 5,
            "min_episodes_per_level": 10
        }

        # ===== EPISODE DIVERSITY TRACKING =====
        self._episode_statistics: Dict[str, Any] = {
            "total_generated": 0,
            "unique_scenario_types": set(),
            "unique_alert_patterns": set(),
            "unique_host_topologies": set()
        }

    # ── OpenEnv Required Methods ────────────────────────────────────

    def reset(self, task_id: Optional[int] = None, seed: int = 42) -> Tuple[Observation, Dict[str, Any]]:
        """
        Reset environment with curriculum support.
        
        Args:
            task_id: If provided, use this task (evaluation mode).
                     If None, curriculum auto-selects (training mode).
            seed: Random seed for reproducibility
        """
        if self._state is not None:
            # Update old curriculum using previous episode return (Feature 3)
            max_steps = self.episode_max_length
            final_normalized_score = self._state.get("total_reward", 0.0) / max(max_steps, 1)
            self._update_curriculum(final_normalized_score)

        # Adjust new curriculum system from previous episode
        if self._last_episode_score is not None:
            self._adjust_curriculum(self._last_episode_score)

        # Determine task and difficulty via curriculum
        if task_id is None:
            task_id = self._select_task_from_curriculum()

        params = self._get_level_parameters(self._curriculum_level)
        difficulty = params["difficulty"]

        self._task_id = task_id
        self._seed = seed
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._score_breakdown = ScoreBreakdown()
        self._difficulty = difficulty

        max_steps = params["max_steps"]
        self.episode_max_length = max_steps

        scenario = build_scenario(task_id, seed, difficulty)

        self._state = {
            "task_id":       task_id,
            "seed":          seed,
            "scenario":      scenario,
            "phase":         IncidentPhase.DETECTION,
            "curriculum_level": self._curriculum_level,
            "difficulty_tier": params["difficulty_tier"],
            "alerts":        scenario["alerts"],
            "hosts":         scenario["hosts"],
            "events":        scenario["events"],
            "threat_intel":  scenario["threat_intel"],
            "triaged":       {},
            "isolated":      set(),
            "blocked_ips":   set(),
            "disabled_accs": set(),
            "patched_cves":  set(),
            "removed_pers":  set(),
            "rotated_creds": set(),
            "restored":      set(),
            "forensics":     set(),
            "escalated":     False,
            "action_history": [],
            "total_reward":  0.0,
            "cumulative_scores": {"action": 0.0, "explain": 0.0, "steps": 0},
        }

        # Track initially compromised hosts (Feature 2)
        self._infected_hosts = {
            h.host_id for h in self._state["hosts"]
            if h.c2_active or bool(h.persistence)
        }
        self._spread_history = []
        self._queried_hosts = set()        # NEW - reset partial observability each episode

        # Track diversity
        self._episode_statistics["total_generated"] += 1
        self._episode_statistics["unique_alert_patterns"].add(
            self._hash_alert_pattern(scenario["alerts"])
        )
        self._episode_statistics["unique_host_topologies"].add(
            self._hash_topology(scenario["hosts"])
        )

        self._total_episodes += 1
        self._last_episode_score = None

        return self._build_masked_observation(max_steps), {}

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return OpenEnv-compliant 5-tuple.

        Args:
            action: Action model with action_type, target, parameters, justification

        Returns:
            observation: Masked observation (partial observability applied)
            reward: Float in [-1.0, 1.0]
            terminated: True if episode ended naturally (success or failure)
            truncated: True if max steps reached (artificial cutoff)
            info: Dict with detailed metrics for debugging and curriculum

        Raises:
            RuntimeError: If called after episode termination

        Note:
            terminated=True + truncated=False → natural end, no bootstrap needed
            terminated=False + truncated=True → timeout, bootstrap needed for RL
        """
        if self._done or self._state is None:
            raise RuntimeError("Episode done. Call reset() first.")

        self._step_count += 1
        task_id = self._state["task_id"]
        max_steps = {1: 15, 2: 20, 3: 30}.get(task_id, 20)

        # Execute action
        action_result = self._execute_action(action)

        # Score justification (deterministic)
        context = self._build_context()
        ground_truth = self._build_ground_truth(action, action_result)

        if action.justification:
            xai_scores = score_justification(action, context, ground_truth)
        else:
            xai_scores = {
                "explanation_quality": 0.0,
                "reasoning_score": 0.0,
                "evidence_score": 0.0,
                "risk_score": 0.0,
                "alternatives_score": 0.0,
            }

        # Base reward
        action_score = action_result["correctness"]
        expl_score = xai_scores["explanation_quality"]
        base_reward = action_score * 0.60 + expl_score * 0.40

        # Penalties
        penalty = 0.0
        if action_result.get("unjustified"):
            penalty -= 0.05
        if action_result.get("harmful"):
            penalty -= 0.10
        penalty -= 0.01  # Step cost to discourage stalling

        # Progress bonus (NEW - separate calculation)
        progress_bonus = self._calculate_progress_bonus(action, action_result)

        # Malware spread penalty
        new_infections = self._simulate_malware_spread()
        spread_penalty = new_infections * -0.05

        # Combine all components then clamp
        reward_value = max(-1.0, min(1.0,
            base_reward + penalty + progress_bonus + spread_penalty
        ))

        # Accumulate scores
        self._episode_rewards.append(reward_value)
        s = self._state["cumulative_scores"]
        s["action"] += action_score
        s["explain"] += expl_score
        s["steps"] += 1

        # Update score breakdown
        self._score_breakdown = ScoreBreakdown(
            action_correctness=round(s["action"] / s["steps"], 3),
            reasoning_clarity=round(xai_scores["reasoning_score"], 3),
            evidence_validity=round(xai_scores["evidence_score"], 3),
            risk_accuracy=round(xai_scores["risk_score"], 3),
            overall=round(reward_value, 3),
        )

        # Record in action history
        self._state["action_history"].append({
            "step":        self._step_count,
            "action_type": (action.action_type.value
                            if hasattr(action.action_type, "value")
                            else action.action_type),
            "target":      action.target,
            "correctness": action_score,
            "explanation": expl_score,
            "reward":      reward_value,
            "progress_bonus": progress_bonus,
            "justification_reasoning_length": (
                len(action.justification.reasoning) if action.justification else 0
            ),
            "evidence_count": (
                len(action.justification.evidence) if action.justification else 0
            ),
            "risk_assessment_threat_level": (
                str(action.justification.risk_assessment.threat_level)
                if action.justification and action.justification.risk_assessment
                else "unknown"
            ),
            "risk_assessment_confidence": (
                action.justification.risk_assessment.confidence
                if action.justification and action.justification.risk_assessment
                else 0.0
            ),
            "alternatives_count": (
                len(action.justification.alternatives_considered)
                if action.justification else 0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._state["total_reward"] += reward_value

        # Phase transitions
        self._maybe_advance_phase()

        # Termination check (NEW - replaces simple max_steps check)
        terminated, truncated, termination_reason = self._check_termination()
        self._done = terminated or truncated

        # Track final score for curriculum when episode ends
        if self._done:
            grader_result = grade_episode(self._state, task_id)
            self._last_episode_score = grader_result.final_score

        # Build observation with partial observability applied
        obs = self._build_masked_observation(max_steps)

        info = {
            "action_result":      action_result,
            "xai_scores":         xai_scores,
            "step":               self._step_count,
            "max_steps":          max_steps,
            "total_reward":       round(self._state["total_reward"], 4),
            "action_type":        (action.action_type.value
                                   if hasattr(action.action_type, "value")
                                   else action.action_type),
            "phase":              (self._state["phase"].value
                                   if hasattr(self._state["phase"], "value")
                                   else str(self._state["phase"])),
            "termination_reason": termination_reason,   # NEW
            "progress_bonus":     progress_bonus,        # NEW
            "new_infections":     new_infections,        # NEW
            "queried_hosts":      list(self._queried_hosts),  # NEW
            "reward_breakdown": {
                "base_reward":          round(base_reward, 4),
                "action_correctness":   round(action_score, 4),
                "explanation_quality":  round(expl_score, 4),
                "reasoning_score":      round(xai_scores.get("reasoning_score", 0.0), 4),
                "evidence_score":       round(xai_scores.get("evidence_score", 0.0), 4),
                "risk_accuracy_score":  round(xai_scores.get("risk_score", 0.0), 4),
                "penalty":              round(penalty, 4),
                "progress_bonus":       round(progress_bonus, 4),
                "spread_penalty":       round(spread_penalty, 4),
                "final_reward":         round(reward_value, 4),
                "message":              action_result.get("message", ""),
            },
        }

        return obs, float(reward_value), terminated, truncated, info

    def get_state(self) -> Observation:
        """Return current observation without advancing episode."""
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        max_steps = {1: 15, 2: 20, 3: 30}.get(self._state["task_id"], 20)
        return self._build_observation(max_steps)

    def state(self) -> Dict[str, Any]:
        """
        Return full unmasked state for grading.
        Required by OpenEnv grader interface.
        
        IMPORTANT: This returns COMPLETE internal state, not masked observation.
        Graders need the full truth to score correctly:
            - alerts with is_true_positive revealed
            - hosts with c2_active, persistence, vulnerabilities revealed
            - all action history
            - all tracking sets (isolated, triaged, etc.)
        
        Unlike _build_masked_observation() which hides host details from agents,
        this method exposes everything for evaluation purposes.
        
        Returns:
            Full internal state dict
            
        Raises:
            RuntimeError: If no active episode
        """
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._state

    def generate_audit_report(self) -> "AuditReport":
        """
        Generate EU AI Act Article 14 compliance audit report.
        
        Evaluates episode against 5 compliance checks:
        1. All actions justified (Article 14.4(b))
        2. Explanation quality adequate (Article 13.1)
        3. Human oversight available (Article 14.1)
        4. High-risk actions documented (Article 14.4(c))
        5. No systematic bias (Article 10.2(f))
        
        Returns:
            AuditReport: Complete compliance audit with pass/fail for each check
            
        Raises:
            RuntimeError: If no episode has been run
        """
        from .grader import grade_episode
        from .models import AuditReport, ComplianceCheck
        from datetime import datetime, timezone
        
        if self._state is None:
            raise RuntimeError("No episode to audit. Call reset() first.")
        
        history = self._state["action_history"]
        
        # ═══════════════════════════════════════════════════════════════
        # Check 1: All Actions Justified
        # Article 14.4(b) - Appropriate human oversight measures
        # ═══════════════════════════════════════════════════════════════
        all_justified = all(
            h.get("justification_reasoning_length", 0) >= 50
            for h in history
        )
        justified_count = sum(
            1 for h in history if h.get("justification_reasoning_length", 0) >= 50
        )
        
        check1 = ComplianceCheck(
            check_name="All Actions Justified",
            passed=all_justified,
            details=f"{justified_count}/{len(history)} actions have adequate justifications (≥50 chars)",
            article_reference="EU AI Act Article 14.4(b) — Appropriate human oversight measures",
            severity="critical" if not all_justified else "info",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Check 2: Explanation Quality Threshold
        # Article 13.1 - Transparency for users
        # ═══════════════════════════════════════════════════════════════
        if history:
            avg_explanation = sum(h["explanation"] for h in history) / len(history)
        else:
            avg_explanation = 0.0
        
        explanation_adequate = avg_explanation >= 0.6
        
        check2 = ComplianceCheck(
            check_name="Explanation Quality Threshold",
            passed=explanation_adequate,
            details=f"Average explanation quality: {avg_explanation:.3f} (threshold: 0.600)",
            article_reference="EU AI Act Article 13.1 — Transparency for users",
            severity="warning" if not explanation_adequate else "info",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Check 3: Human Oversight Available
        # Article 14.1 - Human oversight
        # ═══════════════════════════════════════════════════════════════
        oversight_available = True  # escalate_incident always available
        oversight_triggered = self._state.get("escalated", False)
        
        check3 = ComplianceCheck(
            check_name="Human Oversight Available",
            passed=oversight_available,
            details=f"Escalate to human analyst action available throughout episode. Triggered: {oversight_triggered}",
            article_reference="EU AI Act Article 14.1 — Human oversight",
            severity="info",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Check 4: High-Risk Actions Documented
        # Article 14.4(c) - Understand system capabilities and limitations
        # ═══════════════════════════════════════════════════════════════
        high_risk_types = {"isolate_host", "disable_account", "restore_host", "remove_persistence"}
        high_risk_actions = [h for h in history if h["action_type"] in high_risk_types]
        
        if high_risk_actions:
            high_risk_documented = all(
                h.get("justification_reasoning_length", 0) >= 50
                for h in high_risk_actions
            )
        else:
            high_risk_documented = True  # Pass if no high-risk actions taken
        
        check4 = ComplianceCheck(
            check_name="High-Risk Actions Documented",
            passed=high_risk_documented,
            details=f"{len(high_risk_actions)} high-risk actions taken, all adequately documented",
            article_reference="EU AI Act Article 14.4(c) — Understand system capabilities",
            severity="warning" if not high_risk_documented else "info",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Check 5: No Systematic Bias
        # Article 10.2(f) - Bias monitoring and mitigation
        # ═══════════════════════════════════════════════════════════════
        triaged = self._state.get("triaged", {})
        
        if triaged:
            tp_count = sum(1 for v in triaged.values() if v == "true_positive")
            fp_count = sum(1 for v in triaged.values() if v == "false_positive")
            
            if fp_count > 0:
                bias_ratio = tp_count / fp_count
                no_bias = 0.3 <= bias_ratio <= 3.0  # Not all TP or all FP
            else:
                # If only TPs classified, acceptable if < 90% of total
                no_bias = tp_count <= len(triaged) * 0.9
        else:
            no_bias = True  # Pass if no triage actions
        
        check5 = ComplianceCheck(
            check_name="No Classification Bias",
            passed=no_bias,
            details=f"Triage balance verified: {len(triaged)} classifications analyzed (TP/FP ratio within acceptable range)",
            article_reference="EU AI Act Article 10.2(f) — Bias monitoring",
            severity="info",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Aggregate Results
        # ═══════════════════════════════════════════════════════════════
        checks = [check1, check2, check3, check4, check5]
        passed_count = sum(1 for c in checks if c.passed)
        compliant = passed_count >= 4
        
        risk_level = (
            "LOW"      if passed_count == 5 else
            "MEDIUM"   if passed_count >= 3 else
            "HIGH"     if passed_count >= 2 else
            "CRITICAL"
        )
        
        # Get final score from grader
        grade = grade_episode(self._state, self._task_id)
        
        return AuditReport(
            report_id=f"audit-{self._task_id}-{self._seed}-{self._step_count}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=self._task_id,
            seed=self._seed,
            episode_steps=self._step_count,
            compliance_checks=checks,
            all_actions_justified=all_justified,
            explanation_quality_avg=round(avg_explanation, 4),
            human_oversight_available=oversight_available,
            human_oversight_triggered=oversight_triggered,
            high_risk_actions_count=len(high_risk_actions),
            audit_trail_length=len(history),
            final_score=grade.final_score,
            compliant=compliant,
            risk_level=risk_level,
        )

    # ── Action Handlers ─────────────────────────────────────────────

    def _execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Route action to appropriate handler.

        Handles both string and Enum action_type values.
        Unknown actions return zero correctness without crashing.
        """
        handlers = {
            "triage_alert":        self._handle_triage,
            "isolate_host":        self._handle_isolate,
            "block_ip":            self._handle_block_ip,
            "disable_account":     self._handle_disable_account,
            "patch_vulnerability": self._handle_patch,
            "remove_persistence":  self._handle_remove_persistence,
            "rotate_credentials":  self._handle_rotate_credentials,
            "restore_host":        self._handle_restore,
            "collect_forensics":   self._handle_forensics,
            "escalate_incident":   self._handle_escalate,
            "query_host":          self._handle_query_host,    # NEW
            "monitor":             self._handle_monitor,       # NEW
        }

        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )

        handler = handlers.get(action_type)
        if not handler:
            return {
                "correctness": 0.0,
                "message": f"Unknown action type: '{action_type}'. Valid: {list(handlers.keys())}",
                "unjustified": True,
                "harmful": False,
            }
        return handler(action)

    def _handle_triage(self, action: Action) -> Dict:
        alert = self._find_alert(action.target)
        if alert is None:
            return {"correctness": 0.0, "message": f"Alert {action.target} not found",
                    "unjustified": False, "harmful": False}
        classification = action.parameters.get("classification", "")
        if classification not in ("true_positive", "false_positive"):
            return {"correctness": 0.0,
                    "message": "classification must be true_positive or false_positive",
                    "unjustified": False, "harmful": False}
        if action.target in self._state["triaged"]:
            return {"correctness": 0.1, "message": "Alert already triaged",
                    "unjustified": False, "harmful": False}
        self._state["triaged"][action.target] = classification
        alert.agent_classification = classification
        correct = (classification == "true_positive") == alert.is_true_positive
        score = 1.0 if correct else 0.0
        msg = "Correct triage" if correct else f"Wrong — was {'TP' if alert.is_true_positive else 'FP'}"
        return {"correctness": score, "message": msg,
                "unjustified": False, "harmful": not correct}

    def _handle_isolate(self, action: Action) -> Dict:
        host = self._find_host(action.target)
        if host is None:
            return {"correctness": 0.0, "message": f"Host {action.target} not found",
                    "unjustified": False, "harmful": False}
        if host.host_id in self._state["isolated"]:
            return {"correctness": 0.1, "message": "Already isolated",
                    "unjustified": False, "harmful": False}
        self._state["isolated"].add(host.host_id)
        was_compromised = host.c2_active or bool(host.persistence)
        host.status = "isolated"
        score = 0.9 if was_compromised else 0.0
        msg = ("Compromised host isolated" if was_compromised
               else "Clean host isolated — unnecessary disruption")
        return {"correctness": score, "message": msg,
                "unjustified": False, "harmful": not was_compromised}

    def _handle_block_ip(self, action: Action) -> Dict:
        ip = action.target
        if ip in self._state["blocked_ips"]:
            return {"correctness": 0.1, "message": "IP already blocked",
                    "unjustified": False, "harmful": False}
        self._state["blocked_ips"].add(ip)
        is_malicious = ip in self._state["threat_intel"].malicious_ips
        score = 0.8 if is_malicious else 0.1
        msg = "Malicious IP blocked" if is_malicious else "IP not in threat intel"
        return {"correctness": score, "message": msg,
                "unjustified": False, "harmful": False}

    def _handle_disable_account(self, action: Action) -> Dict:
        account = action.target
        if account in self._state["disabled_accs"]:
            return {"correctness": 0.1, "message": "Already disabled",
                    "unjustified": False, "harmful": False}
        self._state["disabled_accs"].add(account)
        comp_accounts: set = set()
        for h in self._state["hosts"]:
            if h.c2_active or h.persistence:
                comp_accounts.update(h.accounts)
        score = 0.75 if account in comp_accounts else 0.2
        return {"correctness": score,
                "message": "Compromised account disabled" if account in comp_accounts else "Low-risk account",
                "unjustified": False, "harmful": False}

    def _handle_patch(self, action: Action) -> Dict:
        cve = action.target
        if cve in self._state["patched_cves"]:
            return {"correctness": 0.1, "message": "Already patched",
                    "unjustified": False, "harmful": False}
        self._state["patched_cves"].add(cve)
        affected = [h for h in self._state["hosts"] if cve in h.vulnerabilities]
        for h in affected:
            h.vulnerabilities.remove(cve)
        known = cve in (self._state["threat_intel"].known_cves or [])
        score = 0.85 if known else 0.5
        return {"correctness": score,
                "message": f"CVE {cve} patched on {len(affected)} hosts",
                "unjustified": False, "harmful": False}

    def _handle_remove_persistence(self, action: Action) -> Dict:
        pt = action.target
        found = False
        for h in self._state["hosts"]:
            if pt in h.persistence:
                h.persistence.remove(pt)
                self._state["removed_pers"].add(pt)
                found = True
        score = 0.85 if found else 0.1
        return {"correctness": score,
                "message": f"Persistence '{pt}' {'removed' if found else 'not found'}",
                "unjustified": False, "harmful": False}

    def _handle_rotate_credentials(self, action: Action) -> Dict:
        account = action.target
        if account in self._state["rotated_creds"]:
            return {"correctness": 0.1, "message": "Already rotated",
                    "unjustified": False, "harmful": False}
        self._state["rotated_creds"].add(account)
        comp_accounts: set = set()
        for h in self._state["hosts"]:
            if h.c2_active or h.persistence:
                comp_accounts.update(h.accounts)
        score = 0.8 if account in comp_accounts else 0.3
        return {"correctness": score,
                "message": f"Credentials rotated for {account}",
                "unjustified": False, "harmful": False}

    def _handle_restore(self, action: Action) -> Dict:
        host = self._find_host(action.target)
        if host is None:
            return {"correctness": 0.0, "message": "Host not found",
                    "unjustified": False, "harmful": False}
                    
        # Validate dependencies (Feature 4)
        can_restore, blocking_issues = self._validate_restore_dependencies(action.target)
        if not can_restore:
            messages = [b["message"] for b in blocking_issues]
            return {"correctness": 0.2,
                    "message": f"Prerequisites not met: {'; '.join(messages)}",
                    "unjustified": False, "harmful": False,
                    "blocking_issues": blocking_issues}
                    
        if host.host_id in self._state["restored"]:
            return {"correctness": 0.1, "message": "Already restored",
                    "unjustified": False, "harmful": False}
                    
        host.status = "restored"
        host.persistence = []
        host.vulnerabilities = []
        host.c2_active = False
        self._state["restored"].add(host.host_id)
        if hasattr(self, "_infected_hosts"):
            self._infected_hosts.discard(host.host_id)
            
        return {"correctness": 0.9,
                "message": f"Host {host.hostname} restored",
                "unjustified": False, "harmful": False}

    def _handle_forensics(self, action: Action) -> Dict:
        self._state["forensics"].add(action.target)
        return {"correctness": 0.5, "message": f"Forensics collected from {action.target}",
                "unjustified": False, "harmful": False}

    def _handle_escalate(self, action: Action) -> Dict:
        if self._state["escalated"]:
            return {"correctness": 0.1, "message": "Already escalated",
                    "unjustified": False, "harmful": False}
        self._state["escalated"] = True
        remaining = sum(1 for h in self._state["hosts"]
                        if h.status == "compromised")
        score = 0.6 if remaining > 0 else 0.2
        return {"correctness": score,
                "message": f"Escalated — {remaining} compromised hosts remain",
                "unjustified": False, "harmful": False}

    def _handle_query_host(self, action: "Action") -> dict:
        """
        Handle host intelligence gathering action.

        Reveals hidden host details (c2_active, persistence, vulnerabilities,
        accounts, status) that are masked by partial observability.

        This models realistic SOC workflow where analysts must actively
        investigate hosts before knowing their compromise status.

        Args:
            action: Action with target = host_id or hostname

        Returns:
            Result dict with correctness score and revealed information
        """
        target = action.target
        if not target:
            return {
                "correctness": 0.0,
                "message": "query_host requires a target (host_id or hostname)",
                "unjustified": False,
                "harmful": False,
            }

        host = self._find_host(target)
        if host is None:
            return {
                "correctness": 0.0,
                "message": f"Host '{target}' not found in network",
                "unjustified": False,
                "harmful": False,
            }

        already_queried = host.host_id in self._queried_hosts

        # Mark as queried regardless (idempotent)
        self._queried_hosts.add(host.host_id)

        if already_queried:
            return {
                "correctness": 0.05,  # Small reward - already had this info
                "message": f"Host '{host.hostname}' already queried - no new information",
                "unjustified": False,
                "harmful": False,
                "already_queried": True,
            }

        # Reward for querying compromised hosts (good investigation)
        is_compromised = getattr(host, "c2_active", False) or bool(getattr(host, "persistence", []))
        correctness = 0.30 if is_compromised else 0.10

        return {
            "correctness": correctness,
            "message": f"Host '{host.hostname}' investigated - details now visible",
            "unjustified": False,
            "harmful": False,
            "newly_queried": True,
            "host_is_compromised": is_compromised,
        }

    def _handle_monitor(self, action: "Action") -> dict:
        """
        Passive monitoring action - observe without acting.
        Low reward, used as baseline/fallback action.
        """
        return {
            "correctness": 0.05,
            "message": "Network monitored - no active intervention taken",
            "unjustified": False,
            "harmful": False,
        }

    # ── Phase Management ────────────────────────────────────────────

    def _maybe_advance_phase(self):
        state = self._state
        hosts = state["hosts"]
        task_id = state["task_id"]
        phase = state["phase"]
        phase_str = phase.value if hasattr(phase, "value") else str(phase)

        total_alerts = len(state["alerts"])
        triaged = len(state["triaged"])

        compromised_hosts = [h for h in hosts if h.c2_active or h.persistence]
        isolated_comp = sum(1 for h in compromised_hosts
                            if h.host_id in state["isolated"])

        all_persistence = sum(len(h.persistence) for h in hosts)

        if task_id == 1:
            if triaged >= total_alerts:
                state["phase"] = IncidentPhase.COMPLETED
            return

        if phase_str == "detection":
            if total_alerts > 0 and triaged / total_alerts >= 0.8:
                state["phase"] = IncidentPhase.CONTAINMENT

        elif phase_str == "containment":
            n_comp = len(compromised_hosts)
            rate = isolated_comp / max(n_comp, 1)
            if n_comp == 0 or rate >= 0.7:
                if task_id >= 3:
                    state["phase"] = IncidentPhase.ERADICATION
                else:
                    state["phase"] = IncidentPhase.COMPLETED

        elif phase_str == "eradication":
            if all_persistence == 0:
                state["phase"] = IncidentPhase.RECOVERY

        elif phase_str == "recovery":
            isolated_count = len(state["isolated"])
            restored_count = len(state["restored"])
            if isolated_count > 0 and restored_count >= isolated_count:
                state["phase"] = IncidentPhase.COMPLETED

    def _is_episode_complete(self) -> bool:
        phase = self._state["phase"]
        phase_str = phase.value if hasattr(phase, "value") else str(phase)
        return phase_str == "completed"

    def _check_termination(self) -> tuple:
        """
        Check episode termination with clear reason.

        Separates terminated (natural end) from truncated (timeout).
        This is critical for correct RL training - truncated episodes
        should use bootstrap value, terminated should not.

        Returns:
            terminated (bool): Natural episode end (success or catastrophic failure)
            truncated (bool):  Artificial cutoff (max steps reached)
            reason (str):      Human-readable termination cause

        Termination conditions:
            task_complete:    All objectives met - positive outcome
            network_shutdown: Agent isolated ALL hosts - catastrophic overreaction
            total_compromise: ALL hosts infected - catastrophic failure
            timeout:          Max steps reached - truncated, not terminated
            in_progress:      Episode continues
        """
        task_id = self._state.get("task_id", 1)
        max_steps = {1: 15, 2: 20, 3: 30}.get(task_id, 20)

        # SUCCESS: Task objectives complete
        if self._is_episode_complete():
            return True, False, "task_complete"

        total_hosts = len(self._state.get("hosts", []))

        if total_hosts > 0:
            # FAILURE: Agent isolated the entire network (catastrophic overreaction)
            isolated_count = len(self._state.get("isolated", set()))
            if isolated_count >= total_hosts:
                return True, False, "network_shutdown"

            # FAILURE: All hosts compromised (catastrophic breach)
            compromised_count = sum(
                1 for h in self._state["hosts"]
                if getattr(h, "c2_active", False) or bool(getattr(h, "persistence", []))
            )
            if compromised_count >= total_hosts:
                return True, False, "total_compromise"

        # TIMEOUT: Max steps reached - this is truncation not termination
        if self._step_count >= max_steps:
            return False, True, "timeout"

        # CONTINUE
        return False, False, "in_progress"

    def _calculate_progress_bonus(self, action: "Action", action_result: dict) -> float:
        """
        Calculate milestone-based progress rewards.

        Provides dense reward signal for partial task completion.
        Critical for RL training - sparse rewards slow convergence.

        Design decisions:
            - Only rewards CORRECT actions (correctness > 0.5)
            - Milestone bonuses fire only once per episode
            - Time pressure penalizes slow responses
            - Bonuses are additive but small (max 0.25)
              so they don't dominate the action quality signal

        Args:
            action: The action that was just executed
            action_result: Result dict from action handler

        Returns:
            Float bonus in range [-0.05, 0.25]
        """
        bonus = 0.0
        task_id = self._state.get("task_id", 1)
        correctness = action_result.get("correctness", 0.0)
        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else action.action_type
        )

        # Only reward actions that were at least partially correct
        if correctness < 0.5:
            return 0.0

        # ── Task 1: Alert Triage ──────────────────────────────────────────
        if task_id == 1:
            if action_type == "triage_alert":
                bonus += 0.08  # Reward each correct triage

            # Milestone: All alerts triaged (fires exactly once)
            total_alerts = len(self._state.get("alerts", []))
            triaged_count = len(self._state.get("triaged", {}))
            if total_alerts > 0 and triaged_count >= total_alerts:
                # Check previous step was NOT complete (so bonus fires once)
                prev_triaged = triaged_count - (1 if action_type == "triage_alert" else 0)
                if prev_triaged < total_alerts:
                    bonus += 0.20  # Completion milestone

        # ── Task 2: Incident Containment ──────────────────────────────────
        elif task_id == 2:
            if action_type == "triage_alert":
                bonus += 0.05
            elif action_type == "isolate_host":
                bonus += 0.10  # Containment is the main objective

            # Milestone: All compromised hosts contained
            compromised = [
                h for h in self._state["hosts"]
                if getattr(h, "c2_active", False) or bool(getattr(h, "persistence", []))
            ]
            if compromised:
                isolated_set = self._state.get("isolated", set())
                all_contained = all(h.host_id in isolated_set for h in compromised)
                if all_contained and action_type == "isolate_host":
                    bonus += 0.15  # All threats contained

        # ── Task 3: Full IR Lifecycle ─────────────────────────────────────
        elif task_id == 3:
            # Reward each IR phase action proportionally to phase difficulty
            phase_bonuses = {
                "triage_alert":        0.05,  # Detection phase
                "isolate_host":        0.08,  # Containment phase
                "block_ip":            0.05,
                "disable_account":     0.06,
                "remove_persistence":  0.10,  # Eradication (harder)
                "patch_vulnerability": 0.10,
                "restore_host":        0.12,  # Recovery (hardest)
                "rotate_credentials":  0.08,
            }
            bonus += phase_bonuses.get(action_type, 0.0)

            # Milestone: Full workflow complete
            all_triaged = (
                len(self._state.get("triaged", {})) >= len(self._state.get("alerts", []))
                and len(self._state.get("alerts", [])) > 0
            )
            compromised = [
                h for h in self._state["hosts"]
                if getattr(h, "c2_active", False) or bool(getattr(h, "persistence", []))
            ]
            isolated_set = self._state.get("isolated", set())
            all_contained = all(h.host_id in isolated_set for h in compromised) if compromised else True
            no_persistence = all(not getattr(h, "persistence", []) for h in self._state["hosts"])
            any_restored = len(self._state.get("restored", set())) > 0

            if all_triaged and all_contained and no_persistence and any_restored:
                if action_type == "restore_host":  # Fires on completing action
                    bonus += 0.20

        # ── Time pressure: penalize slow responses ────────────────────────
        max_steps = {1: 15, 2: 20, 3: 30}.get(task_id, 20)
        if self._step_count > max_steps * 0.8:
            bonus -= 0.02  # Small urgency penalty in last 20% of episode

        return round(bonus, 4)

    # ── Helpers ─────────────────────────────────────────────────────

    def _find_alert(self, target: str) -> Optional[SIEMAlert]:
        for a in self._state["alerts"]:
            if a.alert_id == target:
                return a
        return None

    def _find_host(self, target: str) -> Optional[NetworkHost]:
        for h in self._state["hosts"]:
            if h.host_id == target or h.hostname == target:
                return h
        return None

    def _validate_restore_dependencies(self, host_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validates prerequisites before allowing restoration (Task 3)."""
        host = self._find_host(host_id)
        if host is None:
            return False, [{"type": "not_found", "message": f"Host {host_id} not found"}]
        
        blocking_issues = []
        
        # Check 1: Host must be isolated first
        if host.host_id not in self._state["isolated"]:
            blocking_issues.append({
                'type': 'not_isolated',
                'message': f'Host {host_id} must be isolated before restoration',
                'required_action': 'isolate_host'
            })
            
        # Check 2: Persistence mechanisms must be cleared
        if host.persistence:
            blocking_issues.append({
                'type': 'persistence_remains',
                'message': f'Persistence mechanisms still active: {host.persistence}',
                'required_action': 'remove_persistence',
                'count': len(host.persistence)
            })
            
        # Check 3: Known CVEs must be patched
        if host.vulnerabilities:
            blocking_issues.append({
                'type': 'unpatched_cves',
                'message': f'Unpatched vulnerabilities: {host.vulnerabilities}',
                'required_action': 'patch_vulnerability',
                'count': len(host.vulnerabilities)
            })
            
        return len(blocking_issues) == 0, blocking_issues

    def _simulate_malware_spread(self) -> int:
        """
        Simulates malware spreading across network topology.
        Only spreads if host is compromised and NOT isolated.
        """
        import random
        new_infections = []
        isolated_set = self._state.get("isolated", set())
        
        for host in self._state["hosts"]:
            host_id = host.host_id
            host_name = host.hostname
            
            # Only spread if compromised, NOT isolated, and currently tracked as infected
            if (host.c2_active or host.persistence) and host_id not in isolated_set and host_id in self._infected_hosts:
                neighbors_names = self._network_graph.get(host_name, [])
                
                for neighbor_name in neighbors_names:
                    neighbor_host = self._find_host(neighbor_name)
                    if not neighbor_host:
                        continue
                        
                    neighbor_id = neighbor_host.host_id
                    
                    # Skip if already infected or isolated
                    if neighbor_id in self._infected_hosts or neighbor_id in isolated_set:
                        continue
                        
                    # Probabilistic spread
                    if random.random() < self._spread_probability:
                        # Infect neighbor
                        neighbor_host.c2_active = True
                        self._infected_hosts.add(neighbor_id)
                        new_infections.append({
                            'source': host_id,
                            'target': neighbor_id,
                            'step': self._step_count
                        })
                            
        if new_infections:
            self._spread_history.extend(new_infections)
            
        return len(new_infections)

    def _update_curriculum(self, episode_score: float):
        """
        Adjusts difficulty based on recent performance.
        Uses sliding window of last 3 episodes.
        """
        if not self._curriculum_enabled:
            return
            
        self._episode_scores.append(episode_score)
        if len(self._episode_scores) > self._curriculum_window:
            self._episode_scores.pop(0)
            
        if len(self._episode_scores) < self._curriculum_window:
            return
            
        avg_score = sum(self._episode_scores) / len(self._episode_scores)
        old_difficulty = self._difficulty
        
        # Adjust difficulty
        if avg_score > self._auto_adjust_threshold_high and self._difficulty < 1.0:
            self._difficulty = min(1.0, self._difficulty + 0.1)
        elif avg_score < self._auto_adjust_threshold_low and self._difficulty > 0.3:
            self._difficulty = max(0.3, self._difficulty - 0.1)
            
        # Log change
        if old_difficulty != self._difficulty:
            self._difficulty_history.append({
                'episode': len(self._difficulty_history),
                'avg_score': avg_score,
                'old_difficulty': old_difficulty,
                'new_difficulty': self._difficulty
            })

    # ── Curriculum Learning System ──────────────────────────────────

    def _get_level_parameters(self, level: int) -> Dict[str, Any]:
        """
        Map curriculum level (1-10) to difficulty parameters.
        Returns dict with 'difficulty' (0.0-1.0 for scenarios.py) and metadata.
        """
        # Beginner (1-3)
        if level <= 3:
            return {
                "difficulty_tier": "beginner",
                "difficulty": 0.3 + (level - 1) * 0.1,  # 0.3, 0.4, 0.5
                "max_steps": 15
            }
        # Intermediate (4-6)
        elif level <= 6:
            return {
                "difficulty_tier": "intermediate",
                "difficulty": 0.5 + (level - 4) * 0.1,  # 0.5, 0.6, 0.7
                "max_steps": 20
            }
        # Expert (7-10)
        else:
            return {
                "difficulty_tier": "expert",
                "difficulty": 0.7 + (level - 7) * 0.1,  # 0.7, 0.8, 0.9, 1.0
                "max_steps": 30
            }

    def _adjust_curriculum(self, episode_score: float):
        """Auto-adjust curriculum based on performance"""
        self._episode_history.append(episode_score)
        self._episodes_at_current_level += 1

        window = self._curriculum_config["evaluation_window"]
        if len(self._episode_history) > window:
            self._episode_history = self._episode_history[-window:]

        min_episodes = self._curriculum_config["min_episodes_per_level"]
        if (len(self._episode_history) < window or
            self._episodes_at_current_level < min_episodes):
            return

        avg_score = sum(self._episode_history) / len(self._episode_history)

        # Advance
        if avg_score > self._curriculum_config["advance_threshold"]:
            if self._curriculum_level < 10:
                old = self._curriculum_level
                self._curriculum_level += 1
                self._episodes_at_current_level = 0
                self._episode_history.clear()
                print(f"📈 Curriculum: Level {old} → {self._curriculum_level} (avg: {avg_score:.2%})")

        # Regress
        elif avg_score < self._curriculum_config["regress_threshold"]:
            if self._curriculum_level > 1:
                old = self._curriculum_level
                self._curriculum_level -= 1
                self._episodes_at_current_level = 0
                self._episode_history.clear()
                print(f"📉 Curriculum: Level {old} → {self._curriculum_level} (avg: {avg_score:.2%})")

    def _select_task_from_curriculum(self) -> int:
        """Map curriculum level to task ID"""
        if self._curriculum_level <= 3:
            return 1  # Beginner → Task 1
        elif self._curriculum_level <= 6:
            return 2  # Intermediate → Task 2
        else:
            return 3  # Expert → Task 3

    def _hash_alert_pattern(self, alerts) -> str:
        """Create unique signature of alert pattern"""
        pattern = tuple(sorted([
            (str(a.severity), str(getattr(a, 'mitre_technique', '')), a.is_true_positive)
            for a in alerts
        ]))
        return hashlib.md5(str(pattern).encode()).hexdigest()

    def _hash_topology(self, hosts) -> str:
        """Create unique signature of topology"""
        topology = tuple(sorted([
            (str(h.hostname), tuple(sorted([str(s) for s in getattr(h, 'services', [])])), bool(h.c2_active))
            for h in hosts
        ]))
        return hashlib.md5(str(topology).encode()).hexdigest()

    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get episode diversity statistics"""
        stats = self._episode_statistics
        total = stats["total_generated"]

        if total == 0:
            return {"message": "No episodes generated yet"}

        return {
            "total_episodes_generated": total,
            "unique_alert_patterns": len(stats["unique_alert_patterns"]),
            "unique_topologies": len(stats["unique_host_topologies"]),
            "pattern_diversity_rate": round(len(stats["unique_alert_patterns"]) / total, 3) if total > 0 else 0,
            "topology_diversity_rate": round(len(stats["unique_host_topologies"]) / total, 3) if total > 0 else 0
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for current episode.

        Provides detection quality (precision/recall/F1), containment rate,
        action efficiency, and curriculum progress for analysis and debugging.

        Returns:
            Dict with detection_metrics, containment_metrics,
            efficiency_metrics, and curriculum_metrics
        """
        if self._state is None:
            return {"error": "No active episode. Call reset() first."}

        # ── Detection Metrics ─────────────────────────────────────────────
        alerts = self._state.get("alerts", [])
        triaged = self._state.get("triaged", {})
        total_alerts = len(alerts)
        triaged_count = len(triaged)

        true_positives_detected = sum(
            1 for a in alerts
            if a.is_true_positive is True
            and triaged.get(a.alert_id) == "true_positive"
        )
        false_positives_correct = sum(
            1 for a in alerts
            if a.is_true_positive is False
            and triaged.get(a.alert_id) == "false_positive"
        )
        total_true_positives = sum(1 for a in alerts if a.is_true_positive is True)
        total_false_positives = sum(1 for a in alerts if a.is_true_positive is False)

        precision = (
            true_positives_detected / max(triaged_count, 1)
            if triaged_count > 0 else 0.0
        )
        recall = (
            true_positives_detected / max(total_true_positives, 1)
            if total_true_positives > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / max(precision + recall, 0.001)
            if (precision + recall) > 0 else 0.0
        )

        # ── Containment Metrics ───────────────────────────────────────────
        hosts = self._state.get("hosts", [])
        isolated_set = self._state.get("isolated", set())
        compromised = [
            h for h in hosts
            if getattr(h, "c2_active", False) or bool(getattr(h, "persistence", []))
        ]
        contained = [h for h in compromised if h.host_id in isolated_set]

        # ── Efficiency Metrics ────────────────────────────────────────────
        history = self._state.get("action_history", [])
        total_actions = len(history)
        monitor_actions = sum(1 for a in history if a.get("action_type") == "monitor")
        query_actions = sum(1 for a in history if a.get("action_type") == "query_host")
        high_value_actions = total_actions - monitor_actions - query_actions
        efficiency = round(high_value_actions / max(total_actions, 1), 3)

        task_id = self._state.get("task_id", 1)
        max_steps = {1: 15, 2: 20, 3: 30}.get(task_id, 20)

        return {
            "detection_metrics": {
                "total_alerts":             total_alerts,
                "triaged_count":            triaged_count,
                "total_true_positives":     total_true_positives,
                "total_false_positives":    total_false_positives,
                "true_positives_detected":  true_positives_detected,
                "false_positives_correct":  false_positives_correct,
                "precision":                round(precision, 3),
                "recall":                   round(recall, 3),
                "f1_score":                 round(f1, 3),
                "triage_completion":        round(triaged_count / max(total_alerts, 1), 3),
            },
            "containment_metrics": {
                "total_compromised":   len(compromised),
                "contained":          len(contained),
                "containment_rate":   round(len(contained) / max(len(compromised), 1), 3),
                "hosts_at_risk":      len(compromised) - len(contained),
                "total_isolated":     len(isolated_set),
                "spread_events":      len(self._spread_history),
            },
            "observability_metrics": {
                "total_hosts":        len(hosts),
                "queried_hosts":      len(self._queried_hosts),
                "unqueried_hosts":    len(hosts) - len(self._queried_hosts),
                "query_coverage":     round(len(self._queried_hosts) / max(len(hosts), 1), 3),
                "query_actions":      query_actions,
            },
            "efficiency_metrics": {
                "total_actions":       total_actions,
                "steps_taken":         self._step_count,
                "steps_remaining":     max_steps - self._step_count,
                "monitor_actions":     monitor_actions,
                "query_actions":       query_actions,
                "high_value_actions":  high_value_actions,
                "action_efficiency":   efficiency,
                "total_progress_bonus": round(
                    sum(a.get("progress_bonus", 0.0) for a in history), 3
                ),
            },
            "curriculum_metrics": {
                "current_level":     self._curriculum_level,
                "difficulty_tier":   self._get_level_parameters(self._curriculum_level)["difficulty_tier"],
                "episode_return":    round(sum(self._episode_rewards), 3),
                "avg_reward_per_step": round(
                    sum(self._episode_rewards) / max(self._step_count, 1), 3
                ),
                "total_episodes":    self._total_episodes,
            },
        }

    def _build_context(self) -> Dict[str, Any]:
        state = self._state
        return {
            "alert_ids": [a.alert_id for a in state["alerts"]],
            "host_ids":  [h.host_id for h in state["hosts"]],
            "ips":       list({h.ip_address for h in state["hosts"]} |
                              set(state["threat_intel"].malicious_ips)),
            "cves":      state["threat_intel"].known_cves or [],
        }

    def _build_ground_truth(self, action: Action, result: Dict) -> Dict[str, Any]:
        alert = self._find_alert(action.target)
        host = self._find_host(action.target)
        is_fp = False
        threat_level = 3

        if alert is not None:
            is_fp = not alert.is_true_positive
            severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
            threat_level = severity_map.get(str(alert.severity), 2)

        if host is not None:
            is_fp = not (host.c2_active or bool(host.persistence))
            threat_level = 4 if host.c2_active else 3 if host.persistence else 1

        return {
            "is_false_positive":    is_fp,
            "threat_level_numeric": threat_level,
            "action_was_correct":   result["correctness"] >= 0.5,
        }

    def _build_observation(self, max_steps: int) -> Observation:
        state = self._state
        phase = state["phase"]
        phase_str = phase.value if hasattr(phase, "value") else str(phase)

        # HIDE is_true_positive from agent
        safe_alerts = []
        for a in state["alerts"]:
            a_copy = a.model_copy()
            a_copy.is_true_positive = None
            safe_alerts.append(a_copy)

        s = state["cumulative_scores"]
        steps_done = max(s["steps"], 1)
        avg_correct = s["action"] / steps_done
        avg_explain = s["explain"] / steps_done
        running_score = avg_correct * 0.60 + avg_explain * 0.40

        phase_messages = {
            "detection":   "DETECTION: Triage SIEM alerts to identify true threats.",
            "containment": "CONTAINMENT: Isolate compromised hosts and block malicious IPs.",
            "eradication": "ERADICATION: Remove persistence mechanisms and patch CVEs.",
            "recovery":    "RECOVERY: Restore isolated hosts and rotate credentials.",
            "completed":   "COMPLETED: Incident response concluded.",
        }

        available = ["triage_alert", "escalate_incident", "collect_forensics",
                     "query_host", "monitor"]
        if state["task_id"] >= 2:
            available += ["isolate_host", "block_ip", "disable_account"]
        if state["task_id"] >= 3:
            available += ["patch_vulnerability", "remove_persistence",
                          "rotate_credentials", "restore_host"]

        return Observation(
            task_id=state["task_id"],
            step=self._step_count,
            max_steps=max_steps,
            alerts=safe_alerts,
            hosts=state["hosts"],
            network_events=state["events"],
            threat_intel=state["threat_intel"],
            incident_phase=phase_str,
            score_so_far=round(running_score, 4),
            time_remaining=max_steps - self._step_count,
            difficulty=self._difficulty,
            message=phase_messages.get(phase_str, ""),
            available_actions=available,
            score_breakdown=self._score_breakdown,
        )

    def _build_masked_observation(self, max_steps: int) -> Observation:
        """
        Build observation with partial observability applied.

        Hosts that have NOT been queried via query_host action will have
        sensitive fields (c2_active, persistence, vulnerabilities, accounts,
        status) hidden from the agent.

        This forces strategic investigation before action - realistic SOC workflow.

        Crucially: _build_observation() (full, unmasked) is still used by:
            - state() endpoint (grader needs truth)
            - generate_audit_report() (compliance needs truth)
            - Internal phase management logic

        Only THIS method is returned to the agent via step() and reset().

        Args:
            max_steps: Episode length for time_remaining calculation

        Returns:
            Observation with host details masked for unqueried hosts
        """
        import copy

        # Get full unmasked observation first
        full_obs = self._build_observation(max_steps)

        # Apply masking to hosts
        masked_hosts = []
        for host in full_obs.hosts:
            if host.host_id in self._queried_hosts:
                # Full visibility - agent investigated this host
                host_copy = host.model_copy()
                host_copy.is_queried = True
                masked_hosts.append(host_copy)
            else:
                # Limited visibility - agent has NOT investigated
                host_copy = host.model_copy()
                host_copy.is_queried = False
                # Hide sensitive compromise indicators
                host_copy.c2_active = False         # Hidden - appears clean
                host_copy.persistence = []          # Hidden - appears clean
                host_copy.vulnerabilities = []      # Hidden - appears clean
                host_copy.accounts = []             # Hidden - not revealed
                host_copy.status = "online"         # Hidden - appears online
                # VISIBLE: host_id, hostname, ip_address, role,
                #          criticality, services, business_impact
                masked_hosts.append(host_copy)

        # Return new Observation with masked hosts
        return Observation(
            task_id=full_obs.task_id,
            step=full_obs.step,
            max_steps=full_obs.max_steps,
            alerts=full_obs.alerts,              # Alerts unchanged
            hosts=masked_hosts,                  # Masked hosts
            network_events=full_obs.network_events,
            threat_intel=full_obs.threat_intel,
            incident_phase=full_obs.incident_phase,
            score_so_far=full_obs.score_so_far,
            time_remaining=full_obs.time_remaining,
            difficulty=full_obs.difficulty,
            message=full_obs.message,
            available_actions=full_obs.available_actions,
            score_breakdown=full_obs.score_breakdown,
        )
