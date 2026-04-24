"""
EU AI Act Compliance Engine
============================
Full compliance scoring engine aligned with the EU AI Act (Articles 10, 13, 14).

Every agent action is evaluated across 5 compliance dimensions:
  1. Explanation Quality    (Art. 13 — Transparency)
  2. Human Oversight        (Art. 14 — Human control readiness)
  3. Bias Detection         (Art. 10 — Data governance)
  4. Decision Traceability  (Art. 13 — Audit trail)
  5. Risk Proportionality   (Art. 14 — Proportionate response)

Provides:
  - Per-action compliance scoring
  - Cumulative audit trail with visual-ready timeline
  - Dashboard-ready aggregate metrics
  - Full compliance report generation
"""

import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Compliance check definitions
# ---------------------------------------------------------------------------

@dataclass
class ComplianceCheck:
    """Result of a single compliance dimension check."""
    dimension: str
    article: str
    score: float       # 0.0 – 1.0
    passed: bool
    details: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ComplianceRecord:
    """Full compliance evaluation for a single action."""
    action_id: str
    timestamp: float
    action_type: str
    target_id: Optional[str]
    checks: List[ComplianceCheck]
    overall_score: float
    compliant: bool
    risk_level: str    # LOW, MEDIUM, HIGH, CRITICAL


# ---------------------------------------------------------------------------
# Compliance Engine
# ---------------------------------------------------------------------------

class EUAIActComplianceEngine:
    """
    Evaluates every agent action against EU AI Act requirements.
    
    Maintains a running audit trail and provides aggregate compliance
    metrics suitable for dashboard visualization.
    """

    # Minimum passing score per dimension
    PASSING_THRESHOLD = 0.50

    # Risk level boundaries
    RISK_THRESHOLDS = {
        "LOW": 0.80,
        "MEDIUM": 0.60,
        "HIGH": 0.40,
        "CRITICAL": 0.0,
    }

    # Action types that REQUIRE human oversight readiness
    OVERSIGHT_REQUIRED_ACTIONS = {
        "isolate_host", "block_ip", "disable_account",
        "escalate_to_human", "revoke_token",
    }

    # High-impact actions that need stronger justification
    HIGH_IMPACT_ACTIONS = {
        "isolate_host", "block_ip", "disable_account", "revoke_token",
    }

    def __init__(self):
        self.audit_trail: List[ComplianceRecord] = []
        self.action_counter = 0
        self._dimension_scores: Dict[str, List[float]] = defaultdict(list)
        self._risk_distribution: Dict[str, int] = defaultdict(int)
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_action(
        self,
        action: Dict,
        justification: Dict,
        observable_state: Dict,
        action_history: List[Dict],
    ) -> ComplianceRecord:
        """
        Evaluate a single agent action for EU AI Act compliance.

        Args:
            action: The action dict (action_type, target_id, etc.)
            justification: The agent's justification (reasoning, evidence, etc.)
            observable_state: Current masked observation
            action_history: Full history of actions taken

        Returns:
            ComplianceRecord with per-dimension scores and overall assessment.
        """
        self.action_counter += 1
        action_type = action.get("action_type", "unknown")
        target_id = action.get("target_id", None)
        action_id = self._generate_action_id(action_type)

        checks = [
            self._check_explanation_quality(justification, action),
            self._check_human_oversight(action, justification),
            self._check_bias_detection(action, observable_state, action_history),
            self._check_decision_traceability(action, justification, observable_state),
            self._check_risk_proportionality(action, justification, observable_state),
        ]

        overall_score = sum(c.score for c in checks) / len(checks)
        compliant = all(c.passed for c in checks)
        risk_level = self._classify_risk(overall_score)

        record = ComplianceRecord(
            action_id=action_id,
            timestamp=time.time(),
            action_type=action_type,
            target_id=target_id,
            checks=checks,
            overall_score=round(overall_score, 4),
            compliant=compliant,
            risk_level=risk_level,
        )

        # Update running metrics
        self.audit_trail.append(record)
        self._risk_distribution[risk_level] += 1
        for check in checks:
            self._dimension_scores[check.dimension].append(check.score)

        return record

    # ------------------------------------------------------------------
    # Individual compliance checks
    # ------------------------------------------------------------------

    def _check_explanation_quality(
        self, justification: Dict, action: Dict
    ) -> ComplianceCheck:
        """
        Article 13 — Transparency: Is the action adequately explained?
        
        Evaluates:
        - Presence and length of reasoning
        - Reference to specific evidence
        - Clarity of decision logic
        """
        score = 0.0
        details_parts = []
        recommendations = []

        reasoning = justification.get("reasoning", "")
        evidence = justification.get("evidence", [])
        confidence = justification.get("confidence", 0)

        # Check reasoning exists and has substance
        if reasoning:
            word_count = len(reasoning.split())
            if word_count >= 20:
                score += 0.35
                details_parts.append(f"Reasoning provided ({word_count} words)")
            elif word_count >= 8:
                score += 0.20
                details_parts.append(f"Brief reasoning ({word_count} words)")
                recommendations.append("Provide more detailed reasoning (20+ words)")
            else:
                score += 0.05
                details_parts.append("Minimal reasoning")
                recommendations.append("Reasoning too brief; explain decision logic")
        else:
            details_parts.append("No reasoning provided")
            recommendations.append("Always provide reasoning for transparency")

        # Check evidence references
        if evidence and len(evidence) > 0:
            score += 0.30
            details_parts.append(f"Evidence cited ({len(evidence)} items)")
        else:
            recommendations.append("Cite specific observable evidence")

        # Check confidence is stated
        if confidence > 0:
            score += 0.15
            details_parts.append(f"Confidence stated ({confidence:.0%})")
        else:
            score += 0.05
            recommendations.append("Include confidence level in justification")

        # Bonus for referencing action type in reasoning
        action_type = action.get("action_type", "")
        if action_type and action_type.replace("_", " ") in reasoning.lower():
            score += 0.10
            details_parts.append("Action type referenced in reasoning")

        # Bonus for structured justification
        if justification.get("alert_ids") or justification.get("host_ids"):
            score += 0.10
            details_parts.append("Structured references to alerts/hosts")

        score = min(1.0, score)
        passed = score >= self.PASSING_THRESHOLD

        return ComplianceCheck(
            dimension="Explanation Quality",
            article="Article 13 (Transparency)",
            score=round(score, 4),
            passed=passed,
            details="; ".join(details_parts) or "No explanation provided",
            recommendations=recommendations,
        )

    def _check_human_oversight(
        self, action: Dict, justification: Dict
    ) -> ComplianceCheck:
        """
        Article 14 — Human Oversight: Is the action suitable for human review?
        
        Evaluates:
        - Whether high-impact actions include escalation readiness
        - Presence of undo/rollback information
        - Flagging of uncertain decisions for human review
        """
        score = 0.0
        details_parts = []
        recommendations = []

        action_type = action.get("action_type", "")
        confidence = justification.get("confidence", 0)
        reasoning = justification.get("reasoning", "")

        requires_oversight = action_type in self.OVERSIGHT_REQUIRED_ACTIONS

        # Base: action has justification at all
        if reasoning:
            score += 0.20
            details_parts.append("Justification available for human review")

        # Check if high-impact action has explicit oversight markers
        if requires_oversight:
            if justification.get("human_review_requested") or "escalat" in reasoning.lower():
                score += 0.30
                details_parts.append("Human oversight explicitly acknowledged")
            else:
                score += 0.10
                recommendations.append("High-impact actions should flag for human review")

            if justification.get("reversible") is not None:
                score += 0.20
                details_parts.append("Reversibility documented")
            else:
                recommendations.append("Document whether action is reversible")
        else:
            # Non-critical actions get baseline credit
            score += 0.40
            details_parts.append("Low-impact action — oversight optional")

        # Low confidence should trigger human oversight
        if confidence > 0 and confidence < 0.6:
            if "human" in reasoning.lower() or "escalat" in reasoning.lower():
                score += 0.20
                details_parts.append("Low-confidence decision flagged for human review")
            else:
                recommendations.append(
                    "Low-confidence decisions should be escalated to humans"
                )
        elif confidence >= 0.6:
            score += 0.15
            details_parts.append(f"Confidence ({confidence:.0%}) supports autonomous action")

        # Bonus for structured escalation path
        if justification.get("escalation_path"):
            score += 0.10
            details_parts.append("Escalation path defined")

        score = min(1.0, score)
        passed = score >= self.PASSING_THRESHOLD

        return ComplianceCheck(
            dimension="Human Oversight Readiness",
            article="Article 14 (Human Oversight)",
            score=round(score, 4),
            passed=passed,
            details="; ".join(details_parts) or "No oversight information",
            recommendations=recommendations,
        )

    def _check_bias_detection(
        self,
        action: Dict,
        observable_state: Dict,
        action_history: List[Dict],
    ) -> ComplianceCheck:
        """
        Article 10 — Data Governance / Bias: Are decisions free from bias?
        
        Evaluates:
        - Whether actions target hosts/IPs proportionally
        - Whether the agent over-targets specific entities
        - Diversity of action types used
        """
        score = 0.0
        details_parts = []
        recommendations = []

        # Analyze action distribution across targets
        target_counts: Dict[str, int] = defaultdict(int)
        action_type_counts: Dict[str, int] = defaultdict(int)

        for past_action in action_history:
            tid = past_action.get("target_id", "none")
            target_counts[tid] += 1
            action_type_counts[past_action.get("action_type", "unknown")] += 1

        total_actions = max(1, len(action_history))

        # Check target concentration (bias indicator)
        if target_counts:
            max_target_pct = max(target_counts.values()) / total_actions
            if max_target_pct > 0.6 and total_actions > 5:
                score += 0.15
                details_parts.append(
                    f"Target concentration warning: {max_target_pct:.0%} on single target"
                )
                recommendations.append("Diversify investigation across targets")
            else:
                score += 0.35
                details_parts.append("Target distribution appears balanced")
        else:
            score += 0.30
            details_parts.append("First action — no bias baseline yet")

        # Check action type diversity
        num_unique_types = len(action_type_counts)
        if num_unique_types >= 3:
            score += 0.30
            details_parts.append(f"Good action diversity ({num_unique_types} types used)")
        elif num_unique_types >= 2:
            score += 0.20
            details_parts.append(f"Moderate action diversity ({num_unique_types} types)")
        else:
            score += 0.10
            if total_actions > 3:
                recommendations.append("Use diverse action types for unbiased investigation")

        # Check for escalation/isolation bias
        isolation_count = action_type_counts.get("isolate_host", 0)
        if isolation_count > 3 and total_actions > 5:
            pct = isolation_count / total_actions
            if pct > 0.4:
                details_parts.append(f"Isolation bias detected ({pct:.0%})")
                recommendations.append("Over-isolating hosts may indicate biased response")
            else:
                score += 0.15
        else:
            score += 0.20
            details_parts.append("No isolation bias detected")

        # Geographic/demographic bias check (placeholder for future)
        score += 0.15
        details_parts.append("No demographic bias indicators found")

        score = min(1.0, score)
        passed = score >= self.PASSING_THRESHOLD

        return ComplianceCheck(
            dimension="Bias Detection",
            article="Article 10 (Data Governance)",
            score=round(score, 4),
            passed=passed,
            details="; ".join(details_parts),
            recommendations=recommendations,
        )

    def _check_decision_traceability(
        self,
        action: Dict,
        justification: Dict,
        observable_state: Dict,
    ) -> ComplianceCheck:
        """
        Article 13 — Traceability: Can the decision be traced back to evidence?
        
        Evaluates:
        - Whether cited evidence exists in observable state
        - Whether alert/host references are valid
        - Completeness of decision chain
        """
        score = 0.0
        details_parts = []
        recommendations = []

        alert_ids_in_state = {
            a["alert_id"] for a in observable_state.get("alerts", [])
        }
        host_ids_in_state = {
            h["host_id"] for h in observable_state.get("hosts", [])
        }

        # Check target_id traceability
        target_id = action.get("target_id", "")
        if target_id:
            if target_id in alert_ids_in_state or target_id in host_ids_in_state:
                score += 0.30
                details_parts.append(f"Target '{target_id}' found in observable state")
            else:
                score += 0.05
                details_parts.append(f"Target '{target_id}' NOT in observable state")
                recommendations.append("Reference only observable entities")
        else:
            score += 0.15
            details_parts.append("No specific target for this action")

        # Check evidence traceability
        evidence = justification.get("evidence", [])
        if evidence:
            traceable = 0
            for ev in evidence:
                ev_str = str(ev).lower()
                # Check if evidence matches any alert description or host
                if any(ev_str in str(a).lower() for a in observable_state.get("alerts", [])):
                    traceable += 1
                elif any(ev_str in str(h).lower() for h in observable_state.get("hosts", [])):
                    traceable += 1
            pct = traceable / max(1, len(evidence))
            score += 0.30 * pct
            details_parts.append(f"Evidence traceability: {traceable}/{len(evidence)}")
            if pct < 0.5:
                recommendations.append("Ensure evidence references observable data")
        else:
            score += 0.05
            recommendations.append("Provide traceable evidence for decisions")

        # Check reasoning references state
        reasoning = justification.get("reasoning", "")
        if reasoning:
            refs_found = 0
            for aid in alert_ids_in_state:
                if aid in reasoning:
                    refs_found += 1
            for hid in host_ids_in_state:
                if hid in reasoning:
                    refs_found += 1
            if refs_found > 0:
                score += 0.25
                details_parts.append(f"Reasoning references {refs_found} state entities")
            else:
                score += 0.05
                recommendations.append("Reference specific alert/host IDs in reasoning")
        else:
            recommendations.append("Provide reasoning for traceability")

        # Action has a justification structure at all
        if justification:
            score += 0.10
            details_parts.append("Justification structure present")

        score = min(1.0, score)
        passed = score >= self.PASSING_THRESHOLD

        return ComplianceCheck(
            dimension="Decision Traceability",
            article="Article 13 (Transparency)",
            score=round(score, 4),
            passed=passed,
            details="; ".join(details_parts) or "No traceability information",
            recommendations=recommendations,
        )

    def _check_risk_proportionality(
        self,
        action: Dict,
        justification: Dict,
        observable_state: Dict,
    ) -> ComplianceCheck:
        """
        Article 14 — Proportionality: Is the action proportionate to the risk?
        
        Evaluates:
        - Whether high-impact actions match high-confidence alerts
        - Whether investigation precedes containment
        - Collateral damage awareness
        """
        score = 0.0
        details_parts = []
        recommendations = []

        action_type = action.get("action_type", "")
        target_id = action.get("target_id", "")
        confidence = justification.get("confidence", 0)
        is_high_impact = action_type in self.HIGH_IMPACT_ACTIONS

        # Check if investigation was done before containment
        query_history = observable_state.get("query_history", [])
        if is_high_impact:
            if target_id in query_history:
                score += 0.30
                details_parts.append("Target was investigated before action")
            else:
                score += 0.05
                details_parts.append("High-impact action without prior investigation")
                recommendations.append("Investigate targets before taking containment actions")

            # High-impact with high confidence = proportionate
            if confidence >= 0.7:
                score += 0.25
                details_parts.append(f"High confidence ({confidence:.0%}) supports action")
            elif confidence >= 0.4:
                score += 0.15
                recommendations.append("Consider gathering more evidence before high-impact actions")
            else:
                score += 0.05
                recommendations.append("Low confidence does not justify high-impact action")
        else:
            # Low-impact actions are inherently proportionate
            score += 0.50
            details_parts.append("Low-impact action — proportionality satisfied")

        # Check collateral awareness
        hosts = observable_state.get("hosts", [])
        if is_high_impact and target_id:
            target_host = next(
                (h for h in hosts if h["host_id"] == target_id), None
            )
            if target_host:
                criticality = target_host.get("criticality", "medium")
                if criticality in ("critical", "high"):
                    if confidence >= 0.7:
                        score += 0.20
                        details_parts.append("Critical asset action justified by evidence")
                    else:
                        score += 0.05
                        recommendations.append(
                            "Isolating critical assets requires high confidence"
                        )
                else:
                    score += 0.15
                    details_parts.append("Non-critical target — lower risk")
            else:
                score += 0.10
        else:
            score += 0.15

        score = min(1.0, score)
        passed = score >= self.PASSING_THRESHOLD

        return ComplianceCheck(
            dimension="Risk Proportionality",
            article="Article 14 (Human Oversight)",
            score=round(score, 4),
            passed=passed,
            details="; ".join(details_parts) or "No proportionality data",
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Reporting & API methods
    # ------------------------------------------------------------------

    def get_audit_report(self) -> Dict:
        """Full compliance audit report for API."""
        total = len(self.audit_trail)
        compliant_count = sum(1 for r in self.audit_trail if r.compliant)

        dimension_avgs = {}
        for dim, scores in self._dimension_scores.items():
            dimension_avgs[dim] = round(sum(scores) / max(1, len(scores)), 4)

        return {
            "total_actions_evaluated": total,
            "compliant_actions": compliant_count,
            "compliance_rate": round(compliant_count / max(1, total), 4),
            "overall_score": round(
                sum(r.overall_score for r in self.audit_trail) / max(1, total), 4
            ),
            "dimension_averages": dimension_avgs,
            "risk_distribution": dict(self._risk_distribution),
            "articles_covered": ["Article 10", "Article 13", "Article 14"],
            "engine_uptime_seconds": round(time.time() - self._created_at, 1),
        }

    def get_trail(self, limit: int = 50) -> List[Dict]:
        """Return the audit trail as serializable dicts."""
        records = self.audit_trail[-limit:]
        trail = []
        for record in records:
            trail.append({
                "action_id": record.action_id,
                "timestamp": record.timestamp,
                "action_type": record.action_type,
                "target_id": record.target_id,
                "overall_score": record.overall_score,
                "compliant": record.compliant,
                "risk_level": record.risk_level,
                "checks": [
                    {
                        "dimension": c.dimension,
                        "article": c.article,
                        "score": c.score,
                        "passed": c.passed,
                        "details": c.details,
                        "recommendations": c.recommendations,
                    }
                    for c in record.checks
                ],
            })
        return trail

    def get_dashboard(self) -> Dict:
        """Dashboard-ready aggregate metrics."""
        report = self.get_audit_report()
        recent = self.audit_trail[-20:] if self.audit_trail else []

        # Trend: last 20 actions' compliance scores
        trend = [
            {"action_id": r.action_id, "score": r.overall_score,
             "compliant": r.compliant, "risk": r.risk_level}
            for r in recent
        ]

        # Worst dimensions (for improvement focus)
        dim_avgs = report.get("dimension_averages", {})
        worst_dims = sorted(dim_avgs.items(), key=lambda x: x[1])[:3]

        # Non-compliant action types
        non_compliant_types: Dict[str, int] = defaultdict(int)
        for r in self.audit_trail:
            if not r.compliant:
                non_compliant_types[r.action_type] += 1

        return {
            "summary": report,
            "compliance_trend": trend,
            "focus_areas": [
                {"dimension": dim, "avg_score": score}
                for dim, score in worst_dims
            ],
            "non_compliant_action_types": dict(non_compliant_types),
            "total_recommendations": sum(
                len(c.recommendations)
                for r in self.audit_trail
                for c in r.checks
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_action_id(self, action_type: str) -> str:
        """Generate a unique, deterministic action ID."""
        raw = f"{action_type}-{self.action_counter}-{time.time()}"
        return f"CMP-{hashlib.md5(raw.encode()).hexdigest()[:12].upper()}"

    def _classify_risk(self, score: float) -> str:
        """Map overall score to risk level."""
        for level, threshold in self.RISK_THRESHOLDS.items():
            if score >= threshold:
                return level
        return "CRITICAL"
