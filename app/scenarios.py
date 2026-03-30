"""
AnomalyGuard — Reproducible Scenario Generator
Uses random.Random(seed) for 100% deterministic scenarios.
Same seed + same task_id ALWAYS produces identical scenario.
"""
from __future__ import annotations

import random as _random_mod
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .models import (
    MitreMapping, NetworkEvent, NetworkHost, SIEMAlert, ThreatIntel,
)
from .real_data import (
    ALERT_TEMPLATES_FP, ALERT_TEMPLATES_TP, ALL_CVE_IDS,
    ATTACK_PATTERNS, CVE_DATABASE, MITRE_TECHNIQUES,
    PERSISTENCE_MECHANISMS, get_all_malicious_domains,
    get_all_malicious_hashes, get_all_malicious_ips,
)


# ── Host pools ──────────────────────────────────────────────────────

_HOSTNAMES_WS = [
    "corp-ws-01", "corp-ws-02", "corp-ws-03", "corp-ws-04",
    "corp-ws-05", "corp-ws-06", "corp-ws-07", "corp-ws-08",
]
_HOSTNAMES_SRV = [
    "web-server-01", "app-server-01", "db-server-01",
    "backup-server-01", "api-gateway-01", "auth-server-01",
    "file-server-01", "email-server-01", "dns-server-01",
    "jump-host-01"
]
_HOSTNAMES_DB = ["db-prod-01", "db-analytics-01", "db-backup-01"]
_HOSTNAMES_DC = ["dc-primary-01", "dc-secondary-01"]

_ACCOUNTS = [
    "admin", "svc_backup", "jsmith", "mwilson", "djones",
    "svc_sql", "svc_web", "developer1", "developer2", "helpdesk",
]

_BASE_IP = "10.0.{subnet}.{host}"


def build_scenario(task_id: int, seed: int, difficulty: float) -> Dict[str, Any]:
    """Build a completely deterministic scenario."""
    rng = _random_mod.Random(seed)

    if task_id == 1:
        return _build_alert_triage(rng, difficulty)
    elif task_id == 2:
        return _build_incident_containment(rng, difficulty)
    elif task_id == 3:
        return _build_full_ir(rng, difficulty)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# ── Task 1: Alert Triage ───────────────────────────────────────────

def _build_alert_triage(rng: _random_mod.Random, difficulty: float) -> Dict:
    num_tp = int(3 + difficulty * 4)  # 3-7
    num_fp = int(3 + difficulty * 3)  # 3-6
    total = num_tp + num_fp

    hosts = _generate_hosts(rng, count=5, difficulty=difficulty)
    host_names = [h.hostname for h in hosts]

    alerts: List[SIEMAlert] = []

    # True positive alerts
    tp_templates = _pick_from_list(rng, ALERT_TEMPLATES_TP, num_tp)
    for i, tpl in enumerate(tp_templates):
        host = rng.choice(host_names)
        host_obj = next((h for h in hosts if h.hostname == host), hosts[0])
        mitre_id = tpl.get("mitre")
        mitre_map = None
        if mitre_id and mitre_id in MITRE_TECHNIQUES:
            tech = MITRE_TECHNIQUES[mitre_id]
            mitre_map = MitreMapping(
                technique_id=mitre_id,
                technique_name=tech["name"],
                tactic=tech["tactic"],
            )

        # IOC matches for TP alerts
        ioc_matches = []
        for ap in ATTACK_PATTERNS:
            for at in ap.alert_templates:
                if at["rule"] == tpl["rule"]:
                    ioc_matches = ap.ioc_hashes[:2] + ap.ioc_ips[:1]
                    break

        alerts.append(SIEMAlert(
            alert_id=f"ALT-{10001 + i:05d}",
            timestamp=_make_timestamp(rng),
            severity=tpl["severity"],
            alert_type=tpl["alert_type"],
            description=tpl["desc"].format(host=host),
            source_host=host,
            source_ip=host_obj.ip_address,
            mitre_technique=mitre_map,
            ioc_matches=ioc_matches,
            confidence=round(rng.uniform(0.70, 0.95), 2),
            is_true_positive=True,
        ))

    # False positive alerts
    fp_templates = _pick_from_list(rng, ALERT_TEMPLATES_FP, num_fp)
    for i, tpl in enumerate(fp_templates):
        host = rng.choice(host_names)
        host_obj = next((h for h in hosts if h.hostname == host), hosts[0])
        alerts.append(SIEMAlert(
            alert_id=f"ALT-{10001 + num_tp + i:05d}",
            timestamp=_make_timestamp(rng),
            severity=tpl["severity"],
            alert_type=tpl["alert_type"],
            description=tpl["desc"].format(host=host),
            source_host=host,
            source_ip=host_obj.ip_address,
            mitre_technique=None,
            ioc_matches=[],
            confidence=round(rng.uniform(0.25, 0.60), 2),
            is_true_positive=False,
        ))

    rng.shuffle(alerts)

    threat_intel = _build_threat_intel(rng, difficulty)
    events = _generate_events(rng, hosts, count=3)

    return {
        "alerts": alerts,
        "hosts": hosts,
        "events": events,
        "threat_intel": threat_intel,
    }


# ── Task 2: Incident Containment ──────────────────────────────────

def _build_incident_containment(rng: _random_mod.Random, difficulty: float) -> Dict:
    num_hosts = int(10 + difficulty * 5)  # 10-15
    hosts = _generate_hosts(rng, count=num_hosts, difficulty=difficulty)

    # Mark some hosts as compromised
    num_compromised = int(2 + difficulty * 3)  # 2-5
    eligible = [h for h in hosts if h.criticality != "critical"]
    compromised_hosts = _pick_from_list(rng, eligible, min(num_compromised, len(eligible)))

    mal_ips = get_all_malicious_ips()
    for h in compromised_hosts:
        h.status = "compromised"
        h.c2_active = True

    # Generate alerts for compromised hosts
    alerts: List[SIEMAlert] = []
    for i, h in enumerate(compromised_hosts):
        tpl = rng.choice(ALERT_TEMPLATES_TP)
        mitre_id = tpl.get("mitre")
        mitre_map = None
        if mitre_id and mitre_id in MITRE_TECHNIQUES:
            tech = MITRE_TECHNIQUES[mitre_id]
            mitre_map = MitreMapping(
                technique_id=mitre_id,
                technique_name=tech["name"],
                tactic=tech["tactic"],
            )

        ioc_matches = []
        for ap in ATTACK_PATTERNS:
            for at in ap.alert_templates:
                if at["rule"] == tpl["rule"]:
                    ioc_matches = ap.ioc_hashes[:1] + ap.ioc_ips[:1]
                    break

        alerts.append(SIEMAlert(
            alert_id=f"ALT-{20001 + i:05d}",
            timestamp=_make_timestamp(rng),
            severity=tpl["severity"],
            alert_type=tpl["alert_type"],
            description=tpl["desc"].format(host=h.hostname),
            source_host=h.hostname,
            source_ip=h.ip_address,
            mitre_technique=mitre_map,
            ioc_matches=ioc_matches,
            confidence=round(rng.uniform(0.75, 0.95), 2),
            is_true_positive=True,
        ))

    # Add some FP alerts
    num_fp = int(2 + difficulty * 2)
    host_names = [h.hostname for h in hosts]
    for i in range(num_fp):
        tpl = rng.choice(ALERT_TEMPLATES_FP)
        host = rng.choice(host_names)
        host_obj = next((h for h in hosts if h.hostname == host), hosts[0])
        alerts.append(SIEMAlert(
            alert_id=f"ALT-{20001 + len(compromised_hosts) + i:05d}",
            timestamp=_make_timestamp(rng),
            severity=tpl["severity"],
            alert_type=tpl["alert_type"],
            description=tpl["desc"].format(host=host),
            source_host=host,
            source_ip=host_obj.ip_address,
            mitre_technique=None,
            ioc_matches=[],
            confidence=round(rng.uniform(0.20, 0.55), 2),
            is_true_positive=False,
        ))

    rng.shuffle(alerts)
    threat_intel = _build_threat_intel(rng, difficulty)
    events = _generate_events(rng, hosts, count=5)

    return {
        "alerts": alerts,
        "hosts": hosts,
        "events": events,
        "threat_intel": threat_intel,
    }


# ── Task 3: Full Incident Response ─────────────────────────────────

def _build_full_ir(rng: _random_mod.Random, difficulty: float) -> Dict:
    # Start from containment scenario
    scenario = _build_incident_containment(rng, difficulty)
    hosts = scenario["hosts"]

    # Add persistence to compromised hosts
    compromised = [h for h in hosts if h.c2_active]
    num_persist = int(1 + difficulty * 3)  # 1-4
    persist_pool = list(PERSISTENCE_MECHANISMS)
    rng.shuffle(persist_pool)
    persist_selected = persist_pool[:min(num_persist, len(persist_pool))]

    for h in compromised:
        num_h_persist = rng.randint(1, min(2, len(persist_selected)))
        h.persistence = rng.sample(persist_selected, num_h_persist)

    # Add vulnerabilities
    num_vulns = int(2 + difficulty * 3)  # 2-5
    vuln_pool = list(ALL_CVE_IDS)
    rng.shuffle(vuln_pool)
    vulns_selected = vuln_pool[:min(num_vulns, len(vuln_pool))]

    # Distribute vulns to hosts
    for v in vulns_selected:
        target = rng.choice(hosts)
        if v not in target.vulnerabilities:
            target.vulnerabilities.append(v)

    # Add accounts to hosts
    accts = list(_ACCOUNTS)
    rng.shuffle(accts)
    for h in hosts:
        num_accts = rng.randint(1, 3)
        h.accounts = rng.sample(accts, min(num_accts, len(accts)))

    # Enrich threat_intel with CVEs
    threat_intel = scenario["threat_intel"]
    threat_intel.known_cves = vulns_selected

    events = _generate_events(rng, hosts, count=8)

    return {
        "alerts": scenario["alerts"],
        "hosts": hosts,
        "events": events,
        "threat_intel": threat_intel,
    }


# ── Helpers ─────────────────────────────────────────────────────────

def _generate_hosts(
    rng: _random_mod.Random, count: int, difficulty: float
) -> List[NetworkHost]:
    """Generate a network of hosts."""
    all_names = list(_HOSTNAMES_WS + _HOSTNAMES_SRV + _HOSTNAMES_DB + _HOSTNAMES_DC)
    rng.shuffle(all_names)
    selected = all_names[:min(count, len(all_names))]

    # If we need more, generate extras
    while len(selected) < count:
        selected.append(f"host-extra-{len(selected):02d}")

    hosts: List[NetworkHost] = []
    for i, name in enumerate(selected):
        if "ws" in name:
            role, crit = "workstation", "low"
        elif "srv" in name or "extra" in name:
            role, crit = "server", "medium"
        elif "db" in name:
            role, crit = "database", "high"
        elif "dc" in name:
            role, crit = "domain_controller", "critical"
        else:
            role, crit = "server", "medium"

        subnet = 1 if role in ("workstation",) else 2 if role == "server" else 3
        hosts.append(NetworkHost(
            host_id=f"HOST-{i + 1:03d}",
            hostname=name,
            ip_address=_BASE_IP.format(subnet=subnet, host=10 + i),
            role=role,
            criticality=crit,
            status="online",
            c2_active=False,
            persistence=[],
            vulnerabilities=[],
            accounts=[],
            business_impact={"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}[crit],
        ))

    return hosts


def _build_threat_intel(rng: _random_mod.Random, difficulty: float) -> ThreatIntel:
    """Build threat intelligence context."""
    all_ips = get_all_malicious_ips()
    all_hashes = get_all_malicious_hashes()
    all_domains = get_all_malicious_domains()

    # Select subset based on difficulty
    num_ips = int(2 + difficulty * 4)
    num_hashes = int(1 + difficulty * 3)
    num_domains = int(1 + difficulty * 2)

    rng.shuffle(all_ips)
    rng.shuffle(all_hashes)
    rng.shuffle(all_domains)

    campaign = rng.choice(ATTACK_PATTERNS)

    return ThreatIntel(
        attack_campaign=campaign.name,
        malicious_ips=all_ips[:num_ips],
        malicious_domains=all_domains[:num_domains],
        malicious_hashes=all_hashes[:num_hashes],
        known_cves=[],
        threat_actor=campaign.threat_actor,
    )


def _generate_events(
    rng: _random_mod.Random, hosts: List[NetworkHost], count: int
) -> List[NetworkEvent]:
    """Generate network events."""
    event_types = [
        "connection", "dns_query", "file_access", "process_start",
        "auth_attempt", "firewall_block",
    ]
    events: List[NetworkEvent] = []
    for _ in range(count):
        src = rng.choice(hosts)
        dst = rng.choice(hosts)
        events.append(NetworkEvent(
            timestamp=_make_timestamp(rng),
            event_type=rng.choice(event_types),
            source=src.hostname,
            destination=dst.hostname if dst != src else None,
            details=f"Network event from {src.hostname}",
        ))
    return events


def _make_timestamp(rng: _random_mod.Random) -> str:
    """Generate a realistic timestamp within the last 2 hours."""
    base = datetime(2026, 3, 29, 10, 0, 0)
    offset = timedelta(minutes=rng.randint(0, 120))
    return (base + offset).isoformat()


def _pick_from_list(rng: _random_mod.Random, pool: list, count: int) -> list:
    """Pick `count` items from pool with replacement if needed."""
    if count <= len(pool):
        return rng.sample(pool, count)
    result = list(pool)
    while len(result) < count:
        result.append(rng.choice(pool))
    rng.shuffle(result)
    return result[:count]
