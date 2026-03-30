"""
AnomalyGuard — Explanation Quality Scorer
All functions are pure — no side effects, no global state.
Regex patterns compiled at module level for performance.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import Action, ActionJustification, EvidenceItem, RiskAssessment


# ── Compiled regex patterns ────────────────────────────────────────

_RE_HOST_ID = re.compile(r"HOST-\d{3}", re.IGNORECASE)
_RE_ALERT_ID = re.compile(r"ALT-\d{5}", re.IGNORECASE)
_RE_IP_ADDR = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_RE_CVE_ID = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
_RE_MITRE_ID = re.compile(r"T\d{4}(?:\.\d{3})?")
_RE_HOSTNAME = re.compile(r"(?:corp-ws|srv-|db-|dc-)\S+", re.IGNORECASE)

_CAUSAL_WORDS = [
    "because", "therefore", "since", "as a result", "indicates",
    "shows", "suggests", "confirms", "implies", "demonstrates",
    "evidenced by", "due to", "caused by", "correlates with",
]


# ── Public API ──────────────────────────────────────────────────────

def score_justification(
    action: Action,
    context: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """
    Score the quality of an action's justification.
    Returns a dict with individual scores and composite explanation_quality.
    All returned values are in [0.0, 1.0].
    """
    justification = action.justification

    if justification is None:
        return {
            "explanation_quality": 0.0,
            "reasoning_score": 0.0,
            "evidence_score": 0.0,
            "risk_score": 0.0,
            "alternatives_score": 0.0,
        }

    reasoning = score_reasoning(justification.reasoning)
    evidence = score_evidence(justification.evidence, context)
    risk = score_risk_assessment(
        justification.risk_assessment, ground_truth
    )
    alternatives = score_alternatives(justification.alternatives_considered)

    # Weighted composite
    explanation_quality = (
        reasoning * 0.30
        + evidence * 0.30
        + risk * 0.25
        + alternatives * 0.15
    )

    return {
        "explanation_quality": _clamp(round(explanation_quality, 4)),
        "reasoning_score": _clamp(round(reasoning, 4)),
        "evidence_score": _clamp(round(evidence, 4)),
        "risk_score": _clamp(round(risk, 4)),
        "alternatives_score": _clamp(round(alternatives, 4)),
    }


# ── Reasoning Scorer ───────────────────────────────────────────────

def score_reasoning(reasoning: str) -> float:
    """Score textual reasoning quality. Pure function."""
    if not reasoning:
        return 0.0

    score = 0.0

    # Length quality (min 50 chars required by model)
    length = len(reasoning)
    if length >= 100:
        score += 0.30
    elif length >= 50:
        score += 0.20
    elif length >= 20:
        score += 0.05
    # else 0

    # Specificity — mentions concrete entities
    specificity_hits = 0
    if _RE_HOST_ID.search(reasoning):
        specificity_hits += 1
    if _RE_ALERT_ID.search(reasoning):
        specificity_hits += 1
    if _RE_IP_ADDR.search(reasoning):
        specificity_hits += 1
    if _RE_CVE_ID.search(reasoning):
        specificity_hits += 1
    if _RE_MITRE_ID.search(reasoning):
        specificity_hits += 1
    if _RE_HOSTNAME.search(reasoning):
        specificity_hits += 1

    score += min(specificity_hits / 3.0, 1.0) * 0.35

    # Causal reasoning words
    reasoning_lower = reasoning.lower()
    causal_count = sum(1 for w in _CAUSAL_WORDS if w in reasoning_lower)
    if causal_count >= 2:
        score += 0.25
    elif causal_count >= 1:
        score += 0.15

    # Technical depth — mentions techniques/tools
    tech_terms = [
        "mimikatz", "psexec", "lateral", "credential", "persistence",
        "c2", "beacon", "exfiltration", "ransomware", "exploit",
        "smb", "lsass", "powershell", "kerberos", "ntlm",
        "firewall", "edr", "siem", "ioc", "ttp",
    ]
    tech_count = sum(1 for t in tech_terms if t in reasoning_lower)
    score += min(tech_count / 3.0, 1.0) * 0.10

    return _clamp(score)


# ── Evidence Scorer ────────────────────────────────────────────────

def score_evidence(
    evidence_list: List[EvidenceItem],
    context: Dict[str, Any],
) -> float:
    """Score evidence quality and validity. Pure function."""
    if not evidence_list:
        return 0.0

    score = 0.0

    # Quantity bonus (1-3 items is ideal)
    num = len(evidence_list)
    if num >= 3:
        score += 0.25
    elif num >= 2:
        score += 0.20
    elif num >= 1:
        score += 0.10

    # Validity — does the evidence source exist in known context?
    known_ids = set()
    for key in ("alert_ids", "host_ids", "ips", "cves"):
        if key in context:
            known_ids.update(str(x) for x in context[key])

    # Generic valid sources
    generic_sources = {
        "siem", "system", "network", "logs", "firewall_logs",
        "edr_logs", "security_log", "threat_intel", "playbook",
    }

    valid_count = 0
    for ev in evidence_list:
        src = ev.source.lower()
        if ev.source in known_ids:
            valid_count += 1
        elif any(g in src for g in generic_sources):
            valid_count += 1

    validity_ratio = valid_count / max(len(evidence_list), 1)
    score += validity_ratio * 0.40

    # Average relevance scores
    avg_relevance = sum(ev.relevance_score for ev in evidence_list) / len(evidence_list)
    score += avg_relevance * 0.20

    # Content quality — evidence content is substantive
    substantive = sum(1 for ev in evidence_list if len(ev.content) >= 20)
    content_ratio = substantive / max(len(evidence_list), 1)
    score += content_ratio * 0.15

    return _clamp(score)


# ── Risk Assessment Scorer ─────────────────────────────────────────

def score_risk_assessment(
    risk: Optional[RiskAssessment],
    ground_truth: Dict[str, Any],
) -> float:
    """Score risk assessment accuracy against ground truth. Pure function."""
    if risk is None:
        return 0.0

    score = 0.0

    # Threat level accuracy
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    predicted = risk.threat_level.upper()
    true_level = ground_truth.get("threat_level_numeric", 2)

    # Convert numeric to string for comparison
    true_str = levels[min(true_level, 3)]

    try:
        predicted_idx = levels.index(predicted)
    except ValueError:
        predicted_idx = 1  # Default MEDIUM

    true_idx = min(true_level, 3)
    level_diff = abs(predicted_idx - true_idx)

    if level_diff == 0:
        score += 0.45
    elif level_diff == 1:
        score += 0.25
    elif level_diff == 2:
        score += 0.10
    # else 0

    # Confidence calibration
    action_correct = ground_truth.get("action_was_correct", False)
    if action_correct and risk.confidence >= 0.7:
        score += 0.25  # Correct and confident — good
    elif not action_correct and risk.confidence >= 0.8:
        score += 0.0   # Wrong but very confident — bad calibration
    elif action_correct:
        score += 0.15
    else:
        score += 0.10

    # Impact description quality
    impact_len = len(risk.potential_impact or "")
    if impact_len >= 30:
        score += 0.15
    elif impact_len >= 15:
        score += 0.10
    elif impact_len >= 5:
        score += 0.05

    # Disruption estimate quality
    disruption_len = len(risk.business_disruption_estimate or "")
    if disruption_len >= 20:
        score += 0.15
    elif disruption_len >= 10:
        score += 0.10
    elif disruption_len >= 5:
        score += 0.05

    return _clamp(score)


# ── Alternatives Scorer ────────────────────────────────────────────

def score_alternatives(
    alternatives: List,
) -> float:
    """Score quality of alternatives considered. Pure function."""
    if not alternatives:
        return 0.15  # Some credit for not listing garbage

    score = 0.0

    # Having alternatives is good
    num = len(alternatives)
    if num >= 2:
        score += 0.40
    elif num >= 1:
        score += 0.25

    # Quality of rejection reasoning
    total_quality = 0.0
    for alt in alternatives:
        reason = ""
        if hasattr(alt, "rejected_because"):
            reason = alt.rejected_because
        elif isinstance(alt, dict):
            reason = alt.get("rejected_because", "")

        if len(reason) >= 30:
            total_quality += 1.0
        elif len(reason) >= 15:
            total_quality += 0.6
        elif len(reason) >= 5:
            total_quality += 0.3

    avg_quality = total_quality / max(num, 1)
    score += avg_quality * 0.60

    return _clamp(score)


# ── Utility ─────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
