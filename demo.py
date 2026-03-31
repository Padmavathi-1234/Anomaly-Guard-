"""
AnomalyGuard — Self-Running Demo
Demonstrates the complete environment in action using rule-based agent.

Run with:
    python demo.py

Requirements:
    pip install requests
    Environment must be running (local or HF Spaces)
"""
from __future__ import annotations

import json
import os
import sys
import time
import requests

BASE_URL = os.getenv(
    "ENV_URL",
    "https://padmavathi-123-anomalyguard.hf.space"
)

TASK_NAMES = {
    1: "Alert Triage",
    2: "Incident Containment",
    3: "Full Incident Response"
}

MAX_STEPS = {1: 15, 2: 20, 3: 30}


def separator(title: str = "") -> None:
    print(f"\n{'═'*60}")
    if title:
        print(f"  {title}")
        print(f"{'═'*60}")


def run_demo() -> None:
    separator("AnomalyGuard — Live Environment Demo")
    print(f"  Environment: {BASE_URL}")
    print(f"  Demonstrating all 3 tasks with rule-based agent")
    print(f"  No LLM required — runs fully offline")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Health check
    print(f"\n  Checking environment health...")
    for attempt in range(10):
        try:
            r = session.get(f"{BASE_URL}/health", timeout=15)
            if r.status_code == 200:
                print(f"  Health: {r.json().get('status', 'ok').upper()} ✓")
                break
        except Exception:
            pass
        print(f"  Waiting... attempt {attempt + 1}/10")
        time.sleep(3)
    else:
        print("  ERROR: Environment not reachable")
        print(f"  Make sure it is running at: {BASE_URL}")
        sys.exit(1)

    results = []

    for task_id in [1, 2, 3]:
        separator(f"Task {task_id}: {TASK_NAMES[task_id]}")

        # Reset environment
        r = session.post(
            f"{BASE_URL}/reset",
            params={"task_id": task_id, "seed": 42},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()

        # Handle both response formats
        if "observation" in data:
            obs = data["observation"]
            info = data.get("info", {})
        else:
            obs = data
            info = {}

        print(f"  Curriculum Level: {info.get('curriculum_level', 1)}")
        print(f"  Difficulty:       {info.get('difficulty_tier', 'beginner')}")
        print(f"  Alerts:           {len(obs.get('alerts', []))}")
        print(f"  Hosts:            {len(obs.get('hosts', []))}")
        print(f"  Max Steps:        {obs.get('max_steps', MAX_STEPS[task_id])}")
        print()

        step = 0
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated) and step < MAX_STEPS[task_id]:
            step += 1

            # Select action using rule-based strategy
            action = _select_action(obs)

            try:
                r = session.post(f"{BASE_URL}/step", json=action, timeout=30)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                print(f"  Step error: {e}")
                break

            obs = result.get("observation", obs)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            truncated = bool(result.get("truncated", False))
            info_step = result.get("info", {})
            total_reward += reward

            rb = info_step.get("reward_breakdown", {})
            action_correct = rb.get("action_correctness", 0.0)
            status_icon = "✓" if action_correct > 0.5 else "✗"
            action_type = action.get("action_type", "unknown")
            target = str(action.get("target", ""))[:15]
            msg = info_step.get("action_result", {}).get("message", "")
            progress = info_step.get("progress_bonus", 0.0)

            print(f"  Step {step:2d}: {action_type:<20} "
                  f"target={target:<15} reward={reward:+.3f} {status_icon}")

            if msg:
                print(f"         → {msg}")
            if progress > 0:
                print(f"         🏆 Milestone bonus: +{progress:.3f}")
            if done or truncated:
                reason = info_step.get("termination_reason", "unknown")
                print(f"\n  Episode ended: {reason}")

        # Grade
        r = session.post(f"{BASE_URL}/grader", params={"task_id": task_id}, timeout=15)
        grade = r.json()

        # Metrics
        try:
            r = session.get(f"{BASE_URL}/metrics/detailed", timeout=10)
            metrics = r.json()
        except:
            metrics = {}

        # EU AI Act compliance
        try:
            r = session.get(f"{BASE_URL}/compliance/audit", timeout=10)
            audit = r.json()
        except:
            audit = {}

        compliant = audit.get("compliant", False)
        risk_level = audit.get("risk_level", "UNKNOWN")
        checks_passed = sum(1 for c in audit.get("compliance_checks", []) if c.get("passed", False))

        print(f"\n  {'─'*50}")
        print(f"  Final Score:      {grade.get('final_score', 0):.4f}")
        print(f"  Action Quality:   {grade.get('action_correctness', 0):.4f}")
        print(f"  Explanation:      {grade.get('explanation_quality', 0):.4f}")
        print(f"  Threats Found:    {grade.get('threats_detected', 0)}")
        print(f"  Threats Missed:   {grade.get('threats_missed', 0)}")
        print(f"  Containment:      {grade.get('containment_rate', 0):.2f}")

        if "detection_metrics" in metrics:
            dm = metrics["detection_metrics"]
            print(f"  Precision:        {dm.get('precision', 0):.3f}")
            print(f"  Recall:           {dm.get('recall', 0):.3f}")
            print(f"  F1 Score:         {dm.get('f1_score', 0):.3f}")

        eu_status = "COMPLIANT ✓" if compliant else "NON-COMPLIANT ✗"
        print(f"  EU AI Act:        {eu_status} ({checks_passed}/5 checks) Risk: {risk_level}")

        for fb in grade.get("feedback", []):
            print(f"  {fb}")

        results.append({
            "task": TASK_NAMES[task_id],
            "score": grade.get("final_score", 0.0),
            "steps": step,
            "compliant": compliant
        })

        time.sleep(1)

    # Final summary
    separator("DEMO RESULTS SUMMARY")
    print(f"  {'Task':<25} {'Score':>8} {'Steps':>6} {'EU AI Act':>14}")
    print(f"  {'─'*55}")
    for res in results:
        eu = "COMPLIANT ✓" if res["compliant"] else "FAILED ✗"
        print(f"  {res['task']:<25} {res['score']:>8.4f} {res['steps']:>6} {eu:>14}")

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"  {'─'*55}")
        print(f"  {'AVERAGE':<25} {avg:>8.4f}")
        print()
        print(f"  Baseline Comparison:")
        print(f"    Random agent expected:    0.05 - 0.10")
        print(f"    Rule-based (this demo):   {avg:.4f}")
        print(f"    RL agent target:          0.75+")
        print(f"    LLM agent (GPT-4):        0.70 - 0.85")

    separator("HOW TO USE")
    print(f"  Interactive API:  {BASE_URL}/docs")
    print(f"  Run LLM Agent:    python inference.py")
    print(f"  Run Tests:        pytest tests/ -v")
    separator()


def _select_action(obs: dict) -> dict:
    """Rule-based action selection for demonstration."""
    available = set(obs.get("available_actions", []))
    hosts = obs.get("hosts", [])
    alerts = obs.get("alerts", [])

    # Priority 1: Query unqueried hosts
    if "query_host" in available:
        unqueried = [h for h in hosts if not h.get("is_queried", False)]
        if unqueried:
            host = unqueried[0]
            return _make_action(
                "query_host", host["host_id"],
                f"Investigating host {host['host_id']} ({host.get('hostname', '')}) "
                f"role={host.get('role', '')} criticality={host.get('criticality', '')}. "
                f"Must query before determining compromise status.",
                host["host_id"], f"Host unqueried — status unknown", 0.7,
                "MEDIUM", 0.6, "Unknown compromise status", "Non-disruptive",
                "isolate_host", "Cannot isolate without confirming compromise"
            )

    # Priority 2: Triage alerts
    if "triage_alert" in available:
        untriaged = [a for a in alerts if not a.get("agent_classification")]
        critical = [a for a in untriaged if a.get("severity") in ("critical", "high")]
        target = critical[0] if critical else (untriaged[0] if untriaged else None)
        if target:
            has_ioc = bool(target.get("ioc_matches"))
            has_mitre = bool(target.get("mitre_technique"))
            high_sev = target.get("severity") in ("critical", "high")
            is_tp = has_ioc or has_mitre or high_sev
            return _make_action(
                "triage_alert", target["alert_id"],
                f"Triaging alert {target['alert_id']} severity={target.get('severity')} "
                f"confidence={target.get('confidence', 0):.2f}. IOCs={target.get('ioc_matches', [])}. "
                f"Classification based on evidence weight.",
                target["alert_id"], f"severity={target.get('severity')} confidence={target.get('confidence', 0):.2f}", 0.85,
                "HIGH" if high_sev else "MEDIUM", 0.75, "Active threat if TP missed", "Non-disruptive",
                "collect_forensics", "Triage must precede forensics",
                {"classification": "true_positive" if is_tp else "false_positive"}
            )

    # Priority 3: Isolate compromised hosts
    if "isolate_host" in available:
        isolated_ids = {h["host_id"] for h in hosts if h.get("status") == "isolated"}
        compromised = [h for h in hosts if h.get("is_queried") and h.get("c2_active") and h["host_id"] not in isolated_ids]
        if compromised:
            host = compromised[0]
            return _make_action(
                "isolate_host", host["host_id"],
                f"Host {host['host_id']} confirmed compromised — c2_active=True. "
                f"Isolation prevents lateral movement.",
                host["host_id"], "c2_active=True confirmed", 0.95,
                "CRITICAL", 0.92, "Lateral movement if not isolated", f"Host {host.get('hostname', '')} offline",
                "block_ip", "IP blocking insufficient for active C2"
            )

    # Fallback: monitor
    return _make_action(
        "monitor", "",
        "No high-priority actions identified. Monitoring for additional indicators.",
        "system", "No untriaged alerts or compromised hosts", 0.5,
        "LOW", 0.5, "Minimal", "Non-disruptive",
        "escalate_incident", "No active threats requiring escalation"
    )


def _make_action(action_type, target, reasoning, ev_source, ev_content, relevance,
                 threat, confidence, impact, disruption, alt_action, alt_reason, params=None):
    return {
        "action_type": action_type,
        "target": target,
        "parameters": params or {},
        "justification": {
            "reasoning": reasoning,
            "evidence": [{"source": ev_source, "content": ev_content, "relevance_score": relevance}],
            "risk_assessment": {
                "threat_level": threat,
                "confidence": confidence,
                "potential_impact": impact,
                "business_disruption_estimate": disruption
            },
            "alternatives_considered": [{"action": alt_action, "rejected_because": alt_reason}]
        }
    }


if __name__ == "__main__":
    run_demo()
