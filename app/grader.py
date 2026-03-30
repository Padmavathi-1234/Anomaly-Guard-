"""
AnomalyGuard — Deterministic Grader
100% deterministic. No random. No time-based logic.
Same env_state dict always produces identical TaskGraderResult.

Scoring (UPDATED FOR SAFETY):
  Task 1: final = triage_accuracy * 0.70 + avg_explanation * 0.30
  Task 2: final = (triage*0.5 + containment*0.5) * 0.80 + avg_explanation * 0.20
  Task 3: final = (triage*0.20 + containment*0.30 + eradication*0.25 + recovery*0.25) * 0.75
           + avg_explanation * 0.25

All clamped to [0.0, 1.0], rounded to 4 decimal places.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .models import TaskGraderResult


def grade_episode(env_state: Dict[str, Any], task_id: int) -> TaskGraderResult:
    """
    Grade a completed episode deterministically.
    env_state is the internal state dict from the environment.
    """
    if task_id == 1:
        return _grade_task_1(env_state)
    elif task_id == 2:
        return _grade_task_2(env_state)
    elif task_id == 3:
        return _grade_task_3(env_state)
    else:
        return TaskGraderResult(
            final_score=0.0,
            action_correctness=0.0,
            explanation_quality=0.0,
            feedback=[f"Unknown task_id: {task_id}"],
        )


# ── Task 1: Alert Triage ───────────────────────────────────────────

def _grade_task_1(state: Dict) -> TaskGraderResult:
    alerts = state.get("alerts", [])
    triaged = state.get("triaged", {})
    history = state.get("action_history", [])
    cumulative = state.get("cumulative_scores", {})

    # Count detection accuracy
    total_tp = sum(1 for a in alerts if a.is_true_positive)
    total_fp = sum(1 for a in alerts if not a.is_true_positive)

    correct_tp = 0
    correct_fp = 0
    missed_tp = 0
    false_alarms = 0

    for a in alerts:
        aid = a.alert_id
        if aid in triaged:
            classification = triaged[aid]
            if a.is_true_positive:
                if classification == "true_positive":
                    correct_tp += 1
                else:
                    missed_tp += 1
            else:
                if classification == "false_positive":
                    correct_fp += 1
                else:
                    false_alarms += 1
        else:
            if a.is_true_positive:
                missed_tp += 1

    total_correct = correct_tp + correct_fp
    total_triaged = len(triaged)
    triage_accuracy = total_correct / max(len(alerts), 1)

    # Explanation quality from cumulative scores
    steps = max(cumulative.get("steps", 1), 1)
    avg_explain = cumulative.get("explain", 0.0) / steps

    # Task 1 formula
    final = triage_accuracy * 0.70 + avg_explain * 0.30

    feedback: List[str] = []
    if triage_accuracy >= 0.8:
        feedback.append("✅ Excellent triage accuracy")
    elif triage_accuracy >= 0.5:
        feedback.append("⚠️ Moderate triage accuracy — review missed alerts")
    else:
        feedback.append("❌ Low triage accuracy — many threats missed")

    if avg_explain >= 0.6:
        feedback.append("✅ Good explanation quality")
    else:
        feedback.append("⚠️ Explanations need more evidence and reasoning")

    threats_detected = correct_tp
    threats_missed_count = missed_tp

    return TaskGraderResult(
        final_score=_clamp4(final),
        action_correctness=_clamp4(triage_accuracy),
        explanation_quality=_clamp4(avg_explain),
        threats_detected=threats_detected,
        threats_missed=threats_missed_count,
        containment_rate=0.0,
        eradication_rate=0.0,
        recovery_rate=0.0,
        steps_taken=steps,
        feedback=feedback,
        details={
            "total_alerts": len(alerts),
            "triaged": total_triaged,
            "correct_tp": correct_tp,
            "correct_fp": correct_fp,
            "missed_tp": missed_tp,
            "false_alarms": false_alarms,
            "triage_accuracy": _clamp4(triage_accuracy),
            "avg_explanation": _clamp4(avg_explain),
        },
    )


# ── Task 2: Incident Containment ──────────────────────────────────

def _grade_task_2(state: Dict) -> TaskGraderResult:
    alerts = state.get("alerts", [])
    triaged = state.get("triaged", {})
    hosts = state.get("hosts", [])
    isolated = state.get("isolated", set())
    blocked = state.get("blocked_ips", set())
    cumulative = state.get("cumulative_scores", {})
    history = state.get("action_history", [])

    # Triage accuracy
    total_tp = sum(1 for a in alerts if a.is_true_positive)
    correct_tp = 0
    missed_tp = 0
    for a in alerts:
        if a.is_true_positive:
            if a.alert_id in triaged and triaged[a.alert_id] == "true_positive":
                correct_tp += 1
            else:
                missed_tp += 1

    triage_score = correct_tp / max(total_tp, 1)

    # Containment rate
    compromised = [h for h in hosts if h.c2_active or h.persistence]
    contained = sum(1 for h in compromised if h.host_id in isolated)
    containment_rate = contained / max(len(compromised), 1)

    # Explanation quality
    steps = max(cumulative.get("steps", 1), 1)
    avg_explain = cumulative.get("explain", 0.0) / steps

    # Task 2 formula
    action_comp = triage_score * 0.5 + containment_rate * 0.5
    final = action_comp * 0.80 + avg_explain * 0.20

    feedback: List[str] = []
    if containment_rate >= 0.8:
        feedback.append("✅ Strong containment of compromised hosts")
    elif containment_rate >= 0.5:
        feedback.append("⚠️ Partial containment — some threats remain active")
    else:
        feedback.append("❌ Poor containment — active threats spreading")

    if triage_score >= 0.7:
        feedback.append("✅ Good alert triage")
    else:
        feedback.append("⚠️ Missed some true positive alerts")

    return TaskGraderResult(
        final_score=_clamp4(final),
        action_correctness=_clamp4(action_comp),
        explanation_quality=_clamp4(avg_explain),
        threats_detected=correct_tp,
        threats_missed=missed_tp,
        containment_rate=_clamp4(containment_rate),
        eradication_rate=0.0,
        recovery_rate=0.0,
        steps_taken=steps,
        feedback=feedback,
        details={
            "triage_score": _clamp4(triage_score),
            "containment_rate": _clamp4(containment_rate),
            "compromised_hosts": len(compromised),
            "contained_hosts": contained,
            "blocked_ips": len(blocked),
            "avg_explanation": _clamp4(avg_explain),
        },
    )


# ── Task 3: Full Incident Response ─────────────────────────────────

def _grade_task_3(state: Dict) -> TaskGraderResult:
    alerts = state.get("alerts", [])
    triaged = state.get("triaged", {})
    hosts = state.get("hosts", [])
    isolated = state.get("isolated", set())
    patched = state.get("patched_cves", set())
    removed_pers = state.get("removed_pers", set())
    rotated = state.get("rotated_creds", set())
    restored = state.get("restored", set())
    cumulative = state.get("cumulative_scores", {})
    threat_intel = state.get("threat_intel")

    # Triage
    total_tp = sum(1 for a in alerts if a.is_true_positive)
    correct_tp = sum(
        1 for a in alerts
        if a.is_true_positive
        and a.alert_id in triaged
        and triaged[a.alert_id] == "true_positive"
    )
    missed_tp = total_tp - correct_tp
    triage_score = correct_tp / max(total_tp, 1)

    # Containment
    compromised = [h for h in hosts if h.c2_active or h.persistence]
    contained = sum(1 for h in compromised if h.host_id in isolated)
    containment_rate = contained / max(len(compromised), 1)

    # Eradication
    all_persistence = set()
    for h in hosts:
        all_persistence.update(h.persistence)
    # Also count already-removed ones
    total_pers = len(all_persistence) + len(removed_pers)
    eradication_rate = len(removed_pers) / max(total_pers, 1)

    known_cves = set()
    if threat_intel and hasattr(threat_intel, "known_cves"):
        known_cves = set(threat_intel.known_cves)
    patch_rate = len(patched & known_cves) / max(len(known_cves), 1) if known_cves else 1.0

    eradication_combined = eradication_rate * 0.5 + patch_rate * 0.5

    # Recovery
    recovery_rate = len(restored) / max(len(isolated), 1) if isolated else 0.0
    cred_accounts = set()
    for h in hosts:
        if h.c2_active or h.persistence:
            cred_accounts.update(h.accounts)
    cred_rate = len(rotated & cred_accounts) / max(len(cred_accounts), 1) if cred_accounts else 1.0
    recovery_combined = recovery_rate * 0.5 + cred_rate * 0.5

    # Explanation quality
    steps = max(cumulative.get("steps", 1), 1)
    avg_explain = cumulative.get("explain", 0.0) / steps

    # Task 3 formula
    action_comp = (
        triage_score * 0.20
        + containment_rate * 0.30
        + eradication_combined * 0.25
        + recovery_combined * 0.25
    )
    final = action_comp * 0.75 + avg_explain * 0.25

    feedback: List[str] = []
    if triage_score >= 0.7:
        feedback.append("✅ Good threat detection")
    else:
        feedback.append("⚠️ Some threats were missed in triage")

    if containment_rate >= 0.7:
        feedback.append("✅ Effective containment")
    else:
        feedback.append("⚠️ Containment incomplete")

    if eradication_combined >= 0.6:
        feedback.append("✅ Eradication progressing well")
    else:
        feedback.append("⚠️ Persistence mechanisms or CVEs remain")

    if recovery_combined >= 0.5:
        feedback.append("✅ Recovery underway")
    else:
        feedback.append("⚠️ Hosts not restored / credentials not rotated")

    return TaskGraderResult(
        final_score=_clamp4(final),
        action_correctness=_clamp4(action_comp),
        explanation_quality=_clamp4(avg_explain),
        threats_detected=correct_tp,
        threats_missed=missed_tp,
        containment_rate=_clamp4(containment_rate),
        eradication_rate=_clamp4(eradication_combined),
        recovery_rate=_clamp4(recovery_combined),
        steps_taken=steps,
        feedback=feedback,
        details={
            "triage_score": _clamp4(triage_score),
            "containment_rate": _clamp4(containment_rate),
            "eradication_rate": _clamp4(eradication_combined),
            "recovery_rate": _clamp4(recovery_combined),
            "patch_rate": _clamp4(patch_rate),
            "persistence_removed": len(removed_pers),
            "credentials_rotated": len(rotated),
            "hosts_restored": len(restored),
            "avg_explanation": _clamp4(avg_explain),
        },
    )


# ── Utility ─────────────────────────────────────────────────────────

def _clamp4(v: float) -> float:
    """Clamp to [0.0, 1.0] and round to 4 decimals."""
    return round(max(0.0, min(1.0, v)), 4)
