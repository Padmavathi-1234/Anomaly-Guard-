"""
AnomalyGuard — Rule-Based Baseline Agent
Every action includes COMPLETE ActionJustification — no shortcuts.

Priority order:
1. Triage untriaged alerts (TP if: has_ioc OR (high/critical severity AND has_mitre))
2. Isolate compromised hosts (status==compromised OR c2_active)
3. Block first 2 malicious IPs from threat_intel
4. Patch known CVEs from threat_intel (task 3 only)
5. Remove persistence mechanisms (task 3 only)
6. Rotate credentials for compromised accounts (task 3 only)
7. Restore isolated hosts with empty persistence (task 3 only)
"""
from __future__ import annotations

from typing import Any, Dict

from .models import (
    Action, ActionJustification, AlternativeConsidered,
    EvidenceItem, RiskAssessment,
)
from .grader import grade_episode


def run_rule_based_baseline(
    task_id: int, seed: int, env: Any
) -> Dict[str, Any]:
    """Run a complete baseline episode and return results."""
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    max_steps = obs_dict["max_steps"]
    done = False
    step_num = 0
    rewards = []

    while not done and step_num < max_steps:
        step_num += 1
        action = _pick_action(obs_dict, task_id)
        try:
            result = env.step(action)
            result_dict = result.model_dump()
            obs_dict = result_dict["observation"]
            done = result_dict["done"]
            rewards.append(result_dict["reward"]["value"])
        except Exception as e:
            break

    # Grade
    grade = grade_episode(env._state, task_id)

    return {
        "task_id":      task_id,
        "seed":         seed,
        "steps":        step_num,
        "total_reward": round(sum(rewards), 4),
        "final_score":  grade.final_score,
        "grade":        grade.model_dump(),
    }


def _pick_action(obs: Dict, task_id: int) -> Action:
    """Select the best action based on priority rules."""
    alerts = obs.get("alerts", [])
    hosts = obs.get("hosts", [])
    threat_intel = obs.get("threat_intel", {})
    phase = obs.get("incident_phase", "detection")

    # 1. Triage untriaged alerts
    for a in alerts:
        if a.get("agent_classification") is None:
            has_ioc = bool(a.get("ioc_matches"))
            severity = a.get("severity", "low")
            has_mitre = a.get("mitre_technique") is not None
            high_sev = severity in ("critical", "high")

            is_tp = has_ioc or (high_sev and has_mitre)
            classification = "true_positive" if is_tp else "false_positive"

            return Action(
                action_type="triage_alert",
                target=a["alert_id"],
                parameters={"classification": classification},
                justification=ActionJustification(
                    reasoning=(
                        f"Triaging alert {a['alert_id']} with severity={severity}, "
                        f"alert_type={a.get('alert_type', 'unknown')}. "
                        f"IOC matches: {a.get('ioc_matches', [])}. "
                        f"MITRE technique: {a.get('mitre_technique', {}).get('technique_id', 'none') if isinstance(a.get('mitre_technique'), dict) else 'none'}. "
                        f"Classifying as {classification} because "
                        + ("IOC matches present and/or high severity with MITRE mapping indicates genuine threat activity."
                           if is_tp else
                           "low severity with no IOC matches and no MITRE technique mapping suggests benign activity.")
                    ),
                    evidence=[EvidenceItem(
                        source=a["alert_id"],
                        content=f"severity={severity}, iocs={a.get('ioc_matches', [])}, host={a.get('source_host', 'unknown')}",
                        relevance_score=0.80 if is_tp else 0.50,
                    )],
                    risk_assessment=RiskAssessment(
                        threat_level="HIGH" if is_tp else "LOW",
                        confidence=0.75 if has_ioc else 0.60,
                        potential_impact="Active threat if true positive is missed — potential data breach or lateral movement",
                        business_disruption_estimate="Triage is non-disruptive — monitoring and classification only",
                    ),
                    alternatives_considered=[AlternativeConsidered(
                        action="collect_forensics",
                        rejected_because="Triage must precede forensics in standard IR workflow to prioritize alerts",
                    )],
                ),
            )

    # 2. Isolate compromised hosts
    if task_id >= 2:
        for h in hosts:
            status = h.get("status", "online")
            c2 = h.get("c2_active", False)
            if (status == "compromised" or c2) and status != "isolated":
                return Action(
                    action_type="isolate_host",
                    target=h["host_id"],
                    parameters={},
                    justification=ActionJustification(
                        reasoning=(
                            f"Host {h['host_id']} ({h['hostname']}) has status={status}, "
                            f"c2_active={c2}, persistence={h.get('persistence', [])}. "
                            f"Isolating to prevent lateral movement and contain the threat. "
                            f"Host criticality is {h.get('criticality', 'unknown')} with "
                            f"business impact score {h.get('business_impact', 0)}."
                        ),
                        evidence=[EvidenceItem(
                            source=h["host_id"],
                            content=f"status={status}, c2_active={c2}, ip={h.get('ip_address', '')}",
                            relevance_score=0.90,
                        )],
                        risk_assessment=RiskAssessment(
                            threat_level="CRITICAL" if c2 else "HIGH",
                            confidence=0.85,
                            potential_impact="Lateral movement to other hosts, data exfiltration, or ransomware deployment",
                            business_disruption_estimate=f"Host {h['hostname']} will be offline — {h.get('role', 'unknown')} impact",
                        ),
                        alternatives_considered=[AlternativeConsidered(
                            action="collect_forensics",
                            rejected_because="Containment takes priority over evidence collection to stop active threat spread",
                        )],
                    ),
                )

    # 3. Block malicious IPs (first 2)
    if task_id >= 2:
        mal_ips = threat_intel.get("malicious_ips", [])
        for ip in mal_ips[:2]:
            return Action(
                action_type="block_ip",
                target=ip,
                parameters={},
                justification=ActionJustification(
                    reasoning=(
                        f"Blocking IP {ip} identified in threat intelligence as malicious. "
                        f"Associated with campaign: {threat_intel.get('attack_campaign', 'unknown')}. "
                        f"Threat actor: {threat_intel.get('threat_actor', 'unknown')}. "
                        f"Blocking known C2 infrastructure prevents command and control communication."
                    ),
                    evidence=[EvidenceItem(
                        source="threat_intel",
                        content=f"IP {ip} in malicious_ips list for campaign {threat_intel.get('attack_campaign', '')}",
                        relevance_score=0.85,
                    )],
                    risk_assessment=RiskAssessment(
                        threat_level="HIGH",
                        confidence=0.80,
                        potential_impact="Continued C2 communication enables attacker persistence and data theft",
                        business_disruption_estimate="Firewall rule addition — minimal disruption to legitimate traffic",
                    ),
                    alternatives_considered=[AlternativeConsidered(
                        action="collect_forensics",
                        rejected_because="Blocking C2 IPs is more urgent than evidence gathering to cut attacker access",
                    )],
                ),
            )

    # 4. Patch known CVEs (task 3)
    if task_id >= 3:
        known_cves = threat_intel.get("known_cves", [])
        for cve in known_cves:
            return Action(
                action_type="patch_vulnerability",
                target=cve,
                parameters={},
                justification=ActionJustification(
                    reasoning=(
                        f"Patching {cve} identified in threat intelligence as actively exploited. "
                        f"This vulnerability is part of the {threat_intel.get('attack_campaign', 'unknown')} campaign. "
                        f"Patching eliminates the attack vector used for initial compromise or privilege escalation."
                    ),
                    evidence=[EvidenceItem(
                        source="threat_intel",
                        content=f"{cve} in known_cves for active campaign",
                        relevance_score=0.85,
                    )],
                    risk_assessment=RiskAssessment(
                        threat_level="HIGH",
                        confidence=0.80,
                        potential_impact="Unpatched vulnerability allows re-exploitation after remediation",
                        business_disruption_estimate="Patch deployment may require service restart — brief downtime expected",
                    ),
                    alternatives_considered=[AlternativeConsidered(
                        action="isolate_host",
                        rejected_because="Hosts already isolated — patching addresses root cause vulnerability",
                    )],
                ),
            )

    # 5. Remove persistence (task 3)
    if task_id >= 3:
        for h in hosts:
            for p in h.get("persistence", []):
                return Action(
                    action_type="remove_persistence",
                    target=p,
                    parameters={},
                    justification=ActionJustification(
                        reasoning=(
                            f"Removing persistence mechanism '{p}' found on host {h['host_id']} ({h['hostname']}). "
                            f"Persistence mechanisms allow attackers to maintain access after reboot. "
                            f"Eradication of all persistence is required before recovery."
                        ),
                        evidence=[EvidenceItem(
                            source=h["host_id"],
                            content=f"persistence={h.get('persistence', [])}, host={h['hostname']}",
                            relevance_score=0.90,
                        )],
                        risk_assessment=RiskAssessment(
                            threat_level="HIGH",
                            confidence=0.85,
                            potential_impact="Attacker regains access after system reboot if persistence remains",
                            business_disruption_estimate="Removing scheduled tasks or registry keys — no service impact",
                        ),
                        alternatives_considered=[AlternativeConsidered(
                            action="restore_host",
                            rejected_because="Must remove persistence before restoring to prevent re-infection",
                        )],
                    ),
                )

    # 6. Rotate credentials (task 3)
    if task_id >= 3:
        for h in hosts:
            if h.get("c2_active") or h.get("persistence"):
                for acc in h.get("accounts", []):
                    return Action(
                        action_type="rotate_credentials",
                        target=acc,
                        parameters={},
                        justification=ActionJustification(
                            reasoning=(
                                f"Rotating credentials for account '{acc}' on compromised host "
                                f"{h['host_id']} ({h['hostname']}). Compromised credentials must "
                                f"be rotated to prevent attacker re-access with stolen credentials. "
                                f"Host had c2_active={h.get('c2_active')}."
                            ),
                            evidence=[EvidenceItem(
                                source=h["host_id"],
                                content=f"account={acc} on compromised host with c2={h.get('c2_active')}",
                                relevance_score=0.80,
                            )],
                            risk_assessment=RiskAssessment(
                                threat_level="HIGH",
                                confidence=0.75,
                                potential_impact="Attacker uses stolen credentials for lateral movement or persistence",
                                business_disruption_estimate="Credential rotation requires user password reset — moderate impact",
                            ),
                            alternatives_considered=[AlternativeConsidered(
                                action="disable_account",
                                rejected_because="Rotation preferred over disabling to maintain business continuity",
                            )],
                        ),
                    )

    # 7. Restore isolated hosts (task 3)
    if task_id >= 3:
        for h in hosts:
            if h.get("status") == "isolated" and not h.get("persistence"):
                return Action(
                    action_type="restore_host",
                    target=h["host_id"],
                    parameters={},
                    justification=ActionJustification(
                        reasoning=(
                            f"Restoring host {h['host_id']} ({h['hostname']}) — currently isolated "
                            f"with no remaining persistence mechanisms. Host has been cleaned and "
                            f"is safe to return to production. All threats eradicated."
                        ),
                        evidence=[EvidenceItem(
                            source=h["host_id"],
                            content=f"status=isolated, persistence=[], c2_active={h.get('c2_active', False)}",
                            relevance_score=0.85,
                        )],
                        risk_assessment=RiskAssessment(
                            threat_level="LOW",
                            confidence=0.80,
                            potential_impact="Extended downtime if host is not restored promptly",
                            business_disruption_estimate="Restoring host to production — positive business impact",
                        ),
                        alternatives_considered=[AlternativeConsidered(
                            action="collect_forensics",
                            rejected_because="Forensics already collected during containment phase — restoration is priority",
                        )],
                    ),
                )

    # Fallback: escalate
    return Action(
        action_type="escalate_incident",
        target="tier2",
        parameters={},
        justification=ActionJustification(
            reasoning=(
                "No remaining actionable items identified in current observation. "
                "Escalating to Tier-2 SOC analysts for manual review of any residual "
                "threats. All automated triage, containment, and remediation steps "
                "have been completed within the current action space."
            ),
            evidence=[EvidenceItem(
                source="system",
                content="All identified alerts triaged, threats contained, no pending actions",
                relevance_score=0.50,
            )],
            risk_assessment=RiskAssessment(
                threat_level="MEDIUM",
                confidence=0.55,
                potential_impact="Delayed response if undetected threats remain in the environment",
                business_disruption_estimate="Escalation is non-disruptive — handoff to senior analysts",
            ),
            alternatives_considered=[AlternativeConsidered(
                action="collect_forensics",
                rejected_because="No specific unanalyzed targets identified for forensic collection at this point",
            )],
        ),
    )
