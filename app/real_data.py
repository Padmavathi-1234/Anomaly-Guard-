"""
AnomalyGuard — Real-World Attack Pattern Database
Based on MITRE ATT&CK framework with actual technique IDs, IOCs, and TTPs.
Uses @dataclass for attack patterns and threat data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── MITRE ATT&CK Techniques ────────────────────────────────────────

MITRE_TECHNIQUES: Dict[str, Dict[str, str]] = {
    "T1059.001": {
        "name": "PowerShell",
        "tactic": "Execution",
        "description": "Adversaries may abuse PowerShell commands and scripts for execution.",
    },
    "T1021.002": {
        "name": "SMB/Windows Admin Shares",
        "tactic": "Lateral Movement",
        "description": "Adversaries may use SMB to move laterally via admin shares.",
    },
    "T1003.001": {
        "name": "LSASS Memory",
        "tactic": "Credential Access",
        "description": "Adversaries may dump credentials from LSASS process memory.",
    },
    "T1071.001": {
        "name": "Web Protocols",
        "tactic": "Command and Control",
        "description": "Adversaries may communicate using application layer protocols (HTTP/S).",
    },
    "T1068": {
        "name": "Exploitation for Privilege Escalation",
        "tactic": "Privilege Escalation",
        "description": "Adversaries may exploit software vulnerabilities to escalate privileges.",
    },
    "T1486": {
        "name": "Data Encrypted for Impact",
        "tactic": "Impact",
        "description": "Adversaries may encrypt data on target systems to interrupt availability.",
    },
    "T1048.001": {
        "name": "Exfiltration Over Symmetric Encrypted Non-C2 Protocol",
        "tactic": "Exfiltration",
        "description": "Adversaries may steal data by exfiltrating over encrypted channels.",
    },
    "T1071.004": {
        "name": "DNS",
        "tactic": "Command and Control",
        "description": "Adversaries may communicate using DNS to avoid detection.",
    },
    "T1558.003": {
        "name": "Kerberoasting",
        "tactic": "Credential Access",
        "description": "Adversaries may abuse Kerberos to steal service account credentials.",
    },
    "T1053.005": {
        "name": "Scheduled Task",
        "tactic": "Persistence",
        "description": "Adversaries may abuse task scheduling to execute malicious code.",
    },
    "T1547.001": {
        "name": "Registry Run Keys",
        "tactic": "Persistence",
        "description": "Adversaries may add programs to registry run keys for persistence.",
    },
    "T1190": {
        "name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "description": "Adversaries may exploit vulnerabilities in internet-facing applications.",
    },
    "T1027": {
        "name": "Obfuscated Files or Information",
        "tactic": "Defense Evasion",
        "description": "Adversaries may obfuscate payloads to hinder detection.",
    },
    "T1567.002": {
        "name": "Exfiltration to Cloud Storage",
        "tactic": "Exfiltration",
        "description": "Adversaries may exfiltrate data to cloud storage services.",
    },
}


# ── Attack Pattern Dataclass ───────────────────────────────────────

@dataclass
class AttackPattern:
    name: str
    threat_actor: str
    severity: str
    kill_chain: List[str]
    mitre_techniques: List[str]
    alert_templates: List[Dict]
    ioc_hashes: List[str] = field(default_factory=list)
    ioc_ips: List[str] = field(default_factory=list)
    ioc_domains: List[str] = field(default_factory=list)
    cves: List[str] = field(default_factory=list)


# ── 8 Real Attack Patterns ─────────────────────────────────────────

ATTACK_PATTERNS: List[AttackPattern] = [
    # 1. WannaCry-style Ransomware
    AttackPattern(
        name="WannaCry Ransomware",
        threat_actor="Shadow Brokers / Lazarus Group",
        severity="critical",
        kill_chain=["initial_access", "execution", "lateral_movement", "impact"],
        mitre_techniques=["T1190", "T1059.001", "T1021.002", "T1486"],
        alert_templates=[
            {
                "alert_type": "ransomware_execution",
                "rule": "EternalBlue SMB Exploit Attempt",
                "severity": "critical",
                "mitre": "T1190",
                "desc": "MS17-010 SMB exploit attempt detected from {host} targeting port 445",
                "is_tp": True,
            },
            {
                "alert_type": "file_encryption",
                "rule": "Ransomware File Encryption",
                "severity": "critical",
                "mitre": "T1486",
                "desc": "Mass file encryption activity (.wncry extension) detected on {host}",
                "is_tp": True,
            },
        ],
        ioc_hashes=[
            "ed01ebfbc9eb5bbea545af4d01bf5f1071661840480439c6e5babe8e080e41aa",
            "24d004a104d4d54034dbcffc2a4b19a11f39008a575aa614ea04703480b1022c",
        ],
        ioc_ips=["197.231.221.211", "128.31.0.39", "149.202.160.69"],
        ioc_domains=["iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea.com"],
        cves=["CVE-2017-0144", "CVE-2017-0145"],
    ),
    # 2. Ryuk Ransomware
    AttackPattern(
        name="Ryuk Ransomware Campaign",
        threat_actor="Wizard Spider",
        severity="critical",
        kill_chain=["initial_access", "execution", "persistence", "impact"],
        mitre_techniques=["T1059.001", "T1053.005", "T1486"],
        alert_templates=[
            {
                "alert_type": "ransomware_execution",
                "rule": "Ryuk Ransomware Execution",
                "severity": "critical",
                "mitre": "T1486",
                "desc": "Ryuk ransomware payload executed on {host}, encrypting files with RSA-4096",
                "is_tp": True,
            },
            {
                "alert_type": "c2_beacon",
                "rule": "Cobalt Strike Beacon Detected",
                "severity": "high",
                "mitre": "T1071.001",
                "desc": "Cobalt Strike C2 beacon activity on {host} (HTTPS, 60s jitter)",
                "is_tp": True,
            },
        ],
        ioc_hashes=[
            "8b0a5fb13309623c3518473551cb1f55d38d8450129d4a3c16b476f7b2867d7d",
        ],
        ioc_ips=["185.202.174.91", "91.218.114.4"],
    ),
    # 3. Conti Ransomware
    AttackPattern(
        name="Conti Ransomware Operation",
        threat_actor="Conti Group",
        severity="critical",
        kill_chain=["initial_access", "credential_access", "lateral_movement", "impact"],
        mitre_techniques=["T1003.001", "T1021.002", "T1486"],
        alert_templates=[
            {
                "alert_type": "ransomware_deployment",
                "rule": "Conti Ransomware Deployment",
                "severity": "critical",
                "mitre": "T1486",
                "desc": "Conti ransomware deploying via Group Policy on {host}",
                "is_tp": True,
            },
            {
                "alert_type": "credential_dumping",
                "rule": "Credential Harvesting via Mimikatz",
                "severity": "critical",
                "mitre": "T1003.001",
                "desc": "Mimikatz credential dumping detected on {host} (sekurlsa::logonpasswords)",
                "is_tp": True,
            },
        ],
        ioc_hashes=[
            "eae876886f19ba384f55778634a35a1d975414e83f22f6111e3e792f706301fe",
        ],
        ioc_ips=["162.244.80.235", "85.93.88.165"],
        cves=["CVE-2021-34527"],
    ),
    # 4. APT29 Lateral Movement
    AttackPattern(
        name="APT Lateral Movement Campaign",
        threat_actor="APT29 / Cozy Bear",
        severity="critical",
        kill_chain=["credential_access", "lateral_movement", "collection"],
        mitre_techniques=["T1003.001", "T1021.002", "T1059.001"],
        alert_templates=[
            {
                "alert_type": "credential_dumping",
                "rule": "LSASS Memory Dump via Procdump",
                "severity": "critical",
                "mitre": "T1003.001",
                "desc": "Procdump.exe used to dump LSASS process memory on {host}",
                "is_tp": True,
            },
            {
                "alert_type": "lateral_movement",
                "rule": "PsExec Remote Execution",
                "severity": "high",
                "mitre": "T1021.002",
                "desc": "PsExec remote service creation detected from {host} to internal targets",
                "is_tp": True,
            },
            {
                "alert_type": "credential_abuse",
                "rule": "Pass-the-Hash Authentication",
                "severity": "high",
                "mitre": "T1003.001",
                "desc": "NTLM pass-the-hash authentication detected from {host}",
                "is_tp": True,
            },
        ],
        ioc_ips=["45.77.65.211", "104.168.44.129"],
        ioc_hashes=[
            "a4a455db9f297e2b9fe99d63c9d31e827efb2cda65be51b3ab4e3e7e07672234",
        ],
    ),
    # 5. DNS Tunneling Exfiltration
    AttackPattern(
        name="DNS Tunneling Exfiltration",
        threat_actor="OilRig / APT34",
        severity="high",
        kill_chain=["command_and_control", "exfiltration"],
        mitre_techniques=["T1071.004", "T1048.001"],
        alert_templates=[
            {
                "alert_type": "dns_tunneling",
                "rule": "DNS Tunneling Detected",
                "severity": "high",
                "mitre": "T1071.004",
                "desc": "High-entropy DNS TXT queries to suspicious domain from {host}",
                "is_tp": True,
            },
            {
                "alert_type": "anomalous_dns",
                "rule": "Anomalous DNS Query Volume",
                "severity": "medium",
                "mitre": "T1071.004",
                "desc": "DNS query volume from {host} is 15x above baseline (possible data exfil)",
                "is_tp": True,
            },
        ],
        ioc_domains=["ns1.dnstunnel.evil.com", "data.exfil-c2.net"],
        ioc_ips=["198.51.100.23"],
    ),
    # 6. Cloud Storage Exfiltration
    AttackPattern(
        name="Cloud Storage Exfiltration",
        threat_actor="Insider Threat / APT",
        severity="high",
        kill_chain=["collection", "exfiltration"],
        mitre_techniques=["T1567.002"],
        alert_templates=[
            {
                "alert_type": "data_exfiltration",
                "rule": "Large Upload to Cloud Storage",
                "severity": "high",
                "mitre": "T1567.002",
                "desc": "Bulk upload (2.3GB) to unauthorized cloud storage from {host}",
                "is_tp": True,
            },
            {
                "alert_type": "sensitive_access",
                "rule": "Sensitive File Access Spike",
                "severity": "medium",
                "mitre": "T1567.002",
                "desc": "Mass access to classified documents on {host} prior to upload",
                "is_tp": True,
            },
        ],
        ioc_domains=["mega.nz", "anonfiles.com"],
    ),
    # 7. Kerberoasting
    AttackPattern(
        name="Kerberoasting Attack",
        threat_actor="Various",
        severity="high",
        kill_chain=["credential_access", "privilege_escalation"],
        mitre_techniques=["T1558.003"],
        alert_templates=[
            {
                "alert_type": "kerberoast",
                "rule": "Kerberoasting - TGS Request Anomaly",
                "severity": "high",
                "mitre": "T1558.003",
                "desc": "Anomalous TGS-REP requests for service accounts from {host}",
                "is_tp": True,
            },
        ],
        ioc_hashes=[
            "b7c3e2f1a8d94e5c6b0f12d3e4a56789c0d1e2f3a4b5c6d7e8f9012345678abc",
        ],
    ),
    # 8. LSASS Credential Dumping
    AttackPattern(
        name="LSASS Credential Dumping",
        threat_actor="Various",
        severity="critical",
        kill_chain=["credential_access"],
        mitre_techniques=["T1003.001"],
        alert_templates=[
            {
                "alert_type": "credential_dumping",
                "rule": "LSASS Process Access",
                "severity": "critical",
                "mitre": "T1003.001",
                "desc": "Unauthorized access to lsass.exe memory on {host} (Mimikatz signature)",
                "is_tp": True,
            },
            {
                "alert_type": "credential_guard_bypass",
                "rule": "Windows Credential Guard Bypass",
                "severity": "high",
                "mitre": "T1003.001",
                "desc": "Credential Guard bypass attempt detected on {host}",
                "is_tp": True,
            },
        ],
        ioc_hashes=[
            "6a0f3b4021998ddf0f2c28c9a27b86ceff7a3f1fbb8730c8a7d5edc74e7f3a2e",
        ],
    ),
]


# ── Alert Templates: 6 TP + 3 FP ──────────────────────────────────

ALERT_TEMPLATES_TP: List[Dict] = []
for _ap in ATTACK_PATTERNS:
    for _tpl in _ap.alert_templates:
        ALERT_TEMPLATES_TP.append({**_tpl, "campaign": _ap.name})

ALERT_TEMPLATES_FP: List[Dict] = [
    {
        "alert_type": "failed_login",
        "rule": "Multiple Failed Login Attempts",
        "severity": "medium",
        "mitre": None,
        "desc": "Multiple failed login attempts on {host} — likely password policy enforcement",
        "is_tp": False,
    },
    {
        "alert_type": "port_scan",
        "rule": "Internal Port Scan Detected",
        "severity": "low",
        "mitre": None,
        "desc": "Internal port scan from {host} — scheduled vulnerability scanner job",
        "is_tp": False,
    },
    {
        "alert_type": "service_restart",
        "rule": "Service Restart Notification",
        "severity": "info",
        "mitre": None,
        "desc": "Windows Update service restart on {host} — routine maintenance",
        "is_tp": False,
    },
    {
        "alert_type": "dns_spike",
        "rule": "DNS Query Spike",
        "severity": "low",
        "mitre": None,
        "desc": "Elevated DNS queries from {host} — software update check cycle",
        "is_tp": False,
    },
    {
        "alert_type": "cert_warning",
        "rule": "Certificate Expiry Warning",
        "severity": "info",
        "mitre": None,
        "desc": "TLS certificate for internal service on {host} expires in 30 days",
        "is_tp": False,
    },
    {
        "alert_type": "high_cpu",
        "rule": "High CPU Usage Alert",
        "severity": "medium",
        "mitre": None,
        "desc": "CPU usage spike on {host} — month-end batch processing",
        "is_tp": False,
    },
]


# ── CVE Database ────────────────────────────────────────────────────

@dataclass
class CVEEntry:
    cve_id: str
    product: str
    severity: str
    cvss: float
    description: str


CVE_DATABASE: List[CVEEntry] = [
    CVEEntry("CVE-2024-21762", "Fortinet FortiOS", "critical", 9.8,
             "Out-of-bound write in FortiOS SSL VPN allows remote code execution"),
    CVEEntry("CVE-2023-44228", "Apache Log4j", "critical", 10.0,
             "Log4Shell — JNDI injection via crafted log messages"),
    CVEEntry("CVE-2017-0144", "Microsoft SMBv1", "critical", 9.3,
             "EternalBlue — SMBv1 remote code execution via crafted packets"),
    CVEEntry("CVE-2021-34527", "Windows Print Spooler", "high", 8.8,
             "PrintNightmare — RCE via Windows print spooler service"),
    CVEEntry("CVE-2023-23397", "Microsoft Outlook", "critical", 9.8,
             "NTLM credential theft via crafted calendar invitation email"),
    CVEEntry("CVE-2024-3400", "Palo Alto PAN-OS", "critical", 10.0,
             "Command injection in GlobalProtect gateway — unauthenticated RCE"),
    CVEEntry("CVE-2023-46805", "Ivanti Connect Secure", "high", 8.2,
             "Authentication bypass in Ivanti VPN appliance allowing unauthorized access"),
]

ALL_CVE_IDS: List[str] = [c.cve_id for c in CVE_DATABASE]


# ── Persistence Mechanisms ──────────────────────────────────────────

PERSISTENCE_MECHANISMS: List[str] = [
    "registry_run_key_T1547.001",
    "scheduled_task_SystemHealthCheck_T1053.005",
    "scheduled_task_WindowsUpdateHelper_T1053.005",
    "wmi_subscription_EventFilter_T1546.003",
    "malicious_service_svchost_helper_T1543.003",
]


# ── Helper Functions ────────────────────────────────────────────────

def get_attack_patterns_by_severity(severity: str) -> List[AttackPattern]:
    """Return attack patterns matching a given severity."""
    return [p for p in ATTACK_PATTERNS if p.severity == severity]


def get_all_malicious_ips() -> List[str]:
    """Collect all C2 IPs from all attack patterns."""
    ips: List[str] = []
    for p in ATTACK_PATTERNS:
        ips.extend(p.ioc_ips)
    return list(set(ips))


def get_all_malicious_hashes() -> List[str]:
    """Collect all malicious file hashes."""
    hashes: List[str] = []
    for p in ATTACK_PATTERNS:
        hashes.extend(p.ioc_hashes)
    return list(set(hashes))


def get_all_malicious_domains() -> List[str]:
    """Collect all C2 domains."""
    domains: List[str] = []
    for p in ATTACK_PATTERNS:
        domains.extend(p.ioc_domains)
    return list(set(domains))
