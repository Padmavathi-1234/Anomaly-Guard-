"""
Dynamic Realistic Attack Scenario Generator
============================================
Generates fully randomized, multi-type cyber attack scenarios with:
- 7 distinct attack archetypes (Supply Chain, Ransomware, Zero-Day, etc.)
- Randomized step counts, timelines, observables, alerts, and prevention windows
- Seed-based reproducibility for consistent RL training
- Curriculum-level integration (higher levels = longer, harder attacks)
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttackTimeline:
    step: int
    phase: str
    technique: str          # MITRE ATT&CK ID
    description: str
    observables: List[str]
    alert_generated: Optional[Dict]
    prevention_opportunity: Optional[Dict]


@dataclass
class RealWorldAttack:
    name: str
    difficulty: float
    cve_ids: List[str]
    mitre_techniques: List[str]
    timeline: List[AttackTimeline]
    iocs: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Attack archetype definitions
# ---------------------------------------------------------------------------

ATTACK_ARCHETYPES = {
    "supply_chain": {
        "name_templates": [
            "{vendor} Supply Chain Compromise",
            "{vendor} Dependency Hijack",
            "{vendor} Build Pipeline Attack",
        ],
        "vendors": ["NPM", "PyPI", "Docker Hub", "GitHub Actions", "Vercel",
                     "CircleCI", "Jenkins", "GitLab CI"],
        "phases": [
            ("initial_compromise", "T1195.001", "Compromised {component} dependency injected",
             ["Unexpected package update", "Hash mismatch in build artifact"]),
            ("persistence", "T1543.003", "Malicious build hook installed in CI pipeline",
             ["Build hook modified", "New systemd service created"]),
            ("credential_theft", "T1552.001", "Pipeline secrets and tokens exfiltrated",
             ["Tokens accessed outside build window", "Secrets manager anomalous read"]),
            ("discovery", "T1087", "Internal service enumeration via stolen credentials",
             ["Unusual API discovery calls", "Service account lateral probe"]),
            ("collection", "T1530", "Sensitive data harvested from cloud storage",
             ["Bulk S3/GCS read", "Database export triggered"]),
            ("exfiltration", "T1567.002", "Data exfiltrated to external service",
             ["Large outbound transfer", "DNS tunneling detected"]),
        ],
        "cve_pool": ["CVE-2024-3094", "CVE-2024-21626", "CVE-2023-44487",
                      "CVE-2024-29190", "CVE-2024-27304"],
    },
    "ransomware": {
        "name_templates": [
            "{group} Ransomware Campaign",
            "{group} Double-Extortion Attack",
            "{group} Crypto-Locker Deployment",
        ],
        "vendors": ["LockBit 4.0", "BlackCat", "Cl0p", "Royal", "Akira",
                     "Play", "Medusa", "8Base"],
        "phases": [
            ("initial_access", "T1566.001", "Phishing email with malicious attachment delivered",
             ["Suspicious email attachment", "Macro execution detected"]),
            ("execution", "T1059.001", "PowerShell payload executed on endpoint",
             ["PowerShell encoded command", "AMSI bypass attempt"]),
            ("privilege_escalation", "T1548.002", "UAC bypass to gain admin privileges",
             ["Registry modification", "Elevated process spawned"]),
            ("defense_evasion", "T1562.001", "Security tools disabled on host",
             ["AV service stopped", "EDR tamper alert"]),
            ("lateral_movement", "T1021.001", "RDP lateral movement to file servers",
             ["Unusual RDP session", "Admin share access"]),
            ("impact", "T1486", "File encryption initiated across network shares",
             ["Mass file rename detected", "Ransom note dropped", "Volume shadow copy deleted"]),
        ],
        "cve_pool": ["CVE-2024-1709", "CVE-2023-46805", "CVE-2024-21887",
                      "CVE-2023-4966", "CVE-2024-27198"],
    },
    "zero_day": {
        "name_templates": [
            "{target} Zero-Day Exploitation",
            "{target} Unknown Vulnerability Attack",
            "{target} 0-Day Remote Code Execution",
        ],
        "vendors": ["Exchange Server", "Confluence", "Fortinet VPN",
                     "Ivanti Connect", "Citrix NetScaler", "PAN-OS"],
        "phases": [
            ("initial_access", "T1190", "Zero-day exploit against {target} service",
             ["Anomalous HTTP request pattern", "Unexpected process spawn on server"]),
            ("execution", "T1059.003", "Web shell deployed via exploit",
             ["Web shell file created", "Suspicious outbound connection from web server"]),
            ("persistence", "T1505.003", "Backdoor implanted in server component",
             ["Modified server binary", "Unexpected scheduled task"]),
            ("credential_access", "T1003.001", "LSASS memory dumped for credentials",
             ["LSASS access from unusual process", "Credential dump tool detected"]),
            ("collection", "T1005", "Sensitive files collected from local systems",
             ["Bulk file access", "Archive creation in temp directory"]),
            ("exfiltration", "T1041", "Staged data exfiltrated over C2 channel",
             ["Encrypted outbound traffic spike", "Beaconing pattern detected"]),
        ],
        "cve_pool": ["CVE-2024-3400", "CVE-2024-21762", "CVE-2024-20353",
                      "CVE-2024-1708", "CVE-2024-27199"],
    },
    "apt_espionage": {
        "name_templates": [
            "APT {group} Cyber Espionage Campaign",
            "{group} State-Sponsored Intrusion",
            "{group} Advanced Persistent Threat",
        ],
        "vendors": ["Cozy Bear", "Lazarus Group", "APT29", "Volt Typhoon",
                     "Sandworm", "Charming Kitten"],
        "phases": [
            ("reconnaissance", "T1595.002", "Active scanning of external infrastructure",
             ["Port scan from known APT infrastructure", "DNS enumeration detected"]),
            ("initial_access", "T1078", "Valid credentials used from compromised partner",
             ["Login from unusual geolocation", "Off-hours VPN authentication"]),
            ("persistence", "T1547.001", "Registry run key persistence established",
             ["New registry autostart entry", "Startup folder modification"]),
            ("defense_evasion", "T1027", "Obfuscated payload to evade detection",
             ["Packed binary detected", "Entropy anomaly in executable"]),
            ("discovery", "T1083", "File and directory discovery across network",
             ["Recursive directory listing", "Sensitive file path enumeration"]),
            ("collection", "T1560.001", "Data archived and staged for exfiltration",
             ["7z/RAR archive created in staging directory", "Large archive in temp"]),
            ("exfiltration", "T1048.002", "Data exfiltrated over encrypted channel",
             ["Unusual DNS TXT queries", "Steganography indicators"]),
        ],
        "cve_pool": ["CVE-2023-42793", "CVE-2024-23222", "CVE-2024-20356",
                      "CVE-2023-38831", "CVE-2024-3273"],
    },
    "insider_threat": {
        "name_templates": [
            "Insider Data Theft by {role}",
            "Malicious {role} Exfiltration",
            "Privileged {role} Abuse",
        ],
        "vendors": ["Database Admin", "DevOps Engineer", "IT Admin",
                     "Security Analyst", "Cloud Architect"],
        "phases": [
            ("abuse_of_access", "T1078.002", "Legitimate credentials used outside normal patterns",
             ["Off-hours database access", "Unusual query volume"]),
            ("collection", "T1213", "Sensitive data collected from internal systems",
             ["SharePoint bulk download", "Wiki/Confluence mass export"]),
            ("staging", "T1074.001", "Data staged to local removable media",
             ["USB device connected", "Local archive creation"]),
            ("exfiltration", "T1048.003", "Data exfiltrated via personal cloud storage",
             ["Upload to personal Google Drive", "Dropbox sync from corp device"]),
        ],
        "cve_pool": [],
    },
    "cloud_attack": {
        "name_templates": [
            "{provider} Cloud Infrastructure Attack",
            "{provider} IAM Privilege Escalation",
            "{provider} Cloud Resource Hijack",
        ],
        "vendors": ["AWS", "Azure", "GCP", "Oracle Cloud", "DigitalOcean"],
        "phases": [
            ("initial_access", "T1078.004", "Stolen cloud API keys used for access",
             ["API call from unknown IP", "New region activity detected"]),
            ("privilege_escalation", "T1548", "IAM role assumption chain exploited",
             ["Cross-account role assumption", "Policy modification detected"]),
            ("discovery", "T1580", "Cloud infrastructure enumeration performed",
             ["EC2/VM listing API calls", "S3/Blob storage enumeration"]),
            ("persistence", "T1098", "New IAM user and access key created",
             ["New IAM user created", "Access key generated for service account"]),
            ("impact", "T1496", "Crypto-mining instances launched across regions",
             ["GPU instance launch spike", "Unusual compute spend alert"]),
        ],
        "cve_pool": ["CVE-2024-21626", "CVE-2024-29990", "CVE-2024-2961"],
    },
    "iot_botnet": {
        "name_templates": [
            "{family} IoT Botnet Propagation",
            "{family} Device Mesh Attack",
            "{family} Smart Device Compromise",
        ],
        "vendors": ["Mirai-NG", "Mozi", "BotenaGo", "Zerobot", "HinataBot"],
        "phases": [
            ("scanning", "T1595.001", "Mass scanning for vulnerable IoT devices",
             ["SYN scan burst on port 23/2323", "Telnet brute-force detected"]),
            ("exploitation", "T1190", "Default credentials or RCE exploit used",
             ["Successful Telnet login with default creds", "Exploit payload delivered"]),
            ("installation", "T1059.004", "Bot binary downloaded and executed",
             ["wget/curl to unknown C2", "New binary in /tmp"]),
            ("c2_communication", "T1071.001", "Bot registers with C2 infrastructure",
             ["HTTP beacon to known botnet C2", "IRC channel join detected"]),
            ("attack", "T1498", "DDoS attack launched from compromised devices",
             ["Volumetric traffic spike", "UDP flood detected"]),
        ],
        "cve_pool": ["CVE-2023-26801", "CVE-2024-22768", "CVE-2023-33246"],
    },
}

# Observable enrichment pools
OBSERVABLE_MODIFIERS = [
    "from {src_ip}", "targeting {dst_ip}", "on port {port}",
    "by user {user}", "at {timestamp}", "via {protocol}",
]

IP_POOL = [
    "185.220.101.45", "91.240.118.172", "45.155.205.233", "103.75.201.4",
    "194.163.175.120", "5.188.206.14", "23.129.64.210", "162.247.74.27",
    "198.98.56.149", "209.141.45.189", "89.248.167.131", "185.56.80.65",
]

DOMAIN_POOL = [
    "pastebin-clone.ru", "cdn-update.xyz", "api-telemetry.cc",
    "secure-login-verify.net", "cloud-sync-service.io", "update-check.top",
    "analytics-cdn.cc", "data-backup-service.xyz", "infra-monitor.ru",
]

HASH_POOL = [
    "a3f5e8c2d1b4a9f7e6c3b2a1d0e9f8c7", "b4c6d8e0f1a2b3c4d5e6f7a8b9c0d1e2",
    "c7d9e1f3a5b7c9d1e3f5a7b9c1d3e5f7", "d8e0f2a4b6c8d0e2f4a6b8c0d2e4f6a8",
    "e9f1a3b5c7d9e1f3a5b7c9d1e3f5a7b9", "f0a2b4c6d8e0f2a4b6c8d0e2f4a6b8c0",
]


# ---------------------------------------------------------------------------
# Dynamic scenario generator
# ---------------------------------------------------------------------------

class RealisticScenarioGenerator:
    """
    Generates fully dynamic, reproducible cyber-attack scenarios.

    Key capabilities:
    - Supports 7 attack archetypes with randomized parameters
    - Curriculum-level integration for progressive difficulty
    - Seed-based reproducibility for consistent RL training
    - Higher difficulty → more steps, phases, observables, and complexity
    """

    # Curriculum level → (min_steps, max_steps, min_phases, max_phases)
    LEVEL_CONFIG = {
        1: (15, 20, 3, 4),
        2: (18, 25, 3, 5),
        3: (22, 30, 4, 5),
        4: (28, 40, 4, 6),
        5: (32, 45, 5, 6),
        6: (38, 55, 5, 7),
        7: (45, 65, 6, 7),
        8: (55, 80, 6, 7),
    }

    def __init__(self):
        self._archetype_keys = list(ATTACK_ARCHETYPES.keys())

    def generate(
        self,
        difficulty: float = 0.5,
        seed: Optional[int] = None,
        curriculum_level: int = 1,
        attack_type: Optional[str] = None,
    ) -> Dict:
        """
        Generate a dynamic attack scenario.

        Args:
            difficulty: 0.0–1.0 base difficulty modifier.
            seed: Random seed for reproducibility.
            curriculum_level: 1–8, controls episode length and complexity.
            attack_type: Force a specific archetype key, or None for random.

        Returns:
            Full scenario dict with initial_state, ground_truth, timeline, etc.
        """
        rng = random.Random(seed)
        level = max(1, min(8, curriculum_level))

        # --- Pick attack archetype ---
        if attack_type and attack_type in ATTACK_ARCHETYPES:
            archetype_key = attack_type
        else:
            archetype_key = rng.choice(self._archetype_keys)
        archetype = ATTACK_ARCHETYPES[archetype_key]

        # --- Determine dimensions from curriculum level ---
        min_steps, max_steps, min_phases, max_phases = self.LEVEL_CONFIG[level]
        total_max_steps = rng.randint(min_steps, max_steps)

        # Select phases (subset for lower levels, full for higher)
        all_phases = list(archetype["phases"])
        num_phases = min(len(all_phases), rng.randint(min_phases, max_phases))
        selected_phases = self._select_phases(rng, all_phases, num_phases)

        # --- Build attack identity ---
        vendor = rng.choice(archetype["vendors"])
        name_template = rng.choice(archetype["name_templates"])
        # Templates may use {vendor}, {group}, {target}, {provider}, {role}, {family}, {component}
        attack_name = name_template.format(
            vendor=vendor, group=vendor, target=vendor,
            provider=vendor, role=vendor, family=vendor, component=vendor,
        )

        # CVEs
        num_cves = rng.randint(1, min(3, max(1, len(archetype["cve_pool"]))))
        cve_ids = rng.sample(archetype["cve_pool"], num_cves) if archetype["cve_pool"] else []

        # --- Build timeline ---
        timeline, mitre_techniques = self._build_timeline(
            rng, selected_phases, total_max_steps, difficulty, level, vendor
        )

        # --- Build IOCs ---
        iocs = self._generate_iocs(rng, level)

        # --- Build initial state and ground truth ---
        initial_state, ground_truth = self._build_state(
            rng, timeline, difficulty, level, iocs
        )

        # --- Assemble RealWorldAttack dataclass ---
        attack = RealWorldAttack(
            name=attack_name,
            difficulty=difficulty,
            cve_ids=cve_ids,
            mitre_techniques=mitre_techniques,
            timeline=timeline,
            iocs=iocs,
        )

        return {
            "name": attack.name,
            "attack_type": archetype_key,
            "difficulty": attack.difficulty,
            "curriculum_level": level,
            "cve_ids": attack.cve_ids,
            "mitre_techniques": attack.mitre_techniques,
            "timeline": [vars(t) for t in attack.timeline],
            "iocs": attack.iocs,
            "initial_state": initial_state,
            "ground_truth": ground_truth,
            "max_steps": total_max_steps,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_phases(
        self, rng: random.Random, all_phases: list, num_phases: int
    ) -> list:
        """Select phases preserving logical order (early phases first)."""
        if num_phases >= len(all_phases):
            return list(all_phases)
        # Always include first and last phase; fill middle randomly
        middle = all_phases[1:-1]
        rng.shuffle(middle)
        selected_middle = sorted(
            middle[: num_phases - 2],
            key=lambda p: all_phases.index(p),
        )
        return [all_phases[0]] + selected_middle + [all_phases[-1]]

    def _build_timeline(
        self,
        rng: random.Random,
        phases: list,
        max_steps: int,
        difficulty: float,
        level: int,
        vendor: str,
    ) -> Tuple[List[AttackTimeline], List[str]]:
        """Build a randomized timeline distributing phases across steps."""
        timeline = []
        mitre_techniques = []
        num_phases = len(phases)

        # Distribute step numbers across the episode
        step_slots = sorted(rng.sample(range(0, max_steps), min(num_phases, max_steps)))
        if len(step_slots) < num_phases:
            step_slots = list(range(0, num_phases))

        for idx, (phase_name, technique, desc_template, obs_templates) in enumerate(phases):
            step_num = step_slots[idx] if idx < len(step_slots) else idx

            # Randomize description
            description = desc_template.format(
                target=vendor, component=vendor, vendor=vendor,
                group=vendor, provider=vendor, role=vendor, family=vendor,
            )

            # Randomize observables (pick subset and add modifiers)
            num_obs = rng.randint(1, len(obs_templates))
            observables = rng.sample(obs_templates, num_obs)
            # Add contextual modifier to some observables at higher levels
            if level >= 3 and rng.random() < 0.5:
                modifier = rng.choice(OBSERVABLE_MODIFIERS).format(
                    src_ip=rng.choice(IP_POOL),
                    dst_ip=f"10.0.{rng.randint(0, 5)}.{rng.randint(10, 250)}",
                    port=rng.choice([22, 80, 443, 445, 3389, 8080, 8443]),
                    user=f"user_{rng.randint(100, 999)}",
                    timestamp=f"2026-04-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
                    protocol=rng.choice(["TCP", "UDP", "HTTP", "DNS", "SMB"]),
                )
                observables[0] = f"{observables[0]} {modifier}"

            # Randomize alert generation (not every phase generates an alert)
            alert = None
            if rng.random() < (0.5 + difficulty * 0.3):
                alert = {
                    "id": f"ALT-{30000 + idx * 100 + rng.randint(1, 99)}",
                    "confidence": round(rng.uniform(0.3, 0.99), 2),
                }

            # Randomize prevention opportunities (harder = fewer windows)
            prevention = None
            prevention_chance = max(0.1, 0.6 - difficulty * 0.4 - level * 0.03)
            if rng.random() < prevention_chance:
                prevention = {
                    "step_limit": step_num + rng.randint(1, 3),
                    "reward_bonus": rng.randint(20, 60),
                    "condition": rng.choice(["query_host", "isolate_host", "block_ip", "revoke_token"]),
                    "target": f"host-{rng.randint(1, 5):02d}",
                }

            entry = AttackTimeline(
                step=step_num,
                phase=phase_name,
                technique=technique,
                description=description,
                observables=observables,
                alert_generated=alert,
                prevention_opportunity=prevention,
            )
            timeline.append(entry)
            mitre_techniques.append(technique)

        return timeline, list(set(mitre_techniques))

    def _generate_iocs(self, rng: random.Random, level: int) -> Dict[str, List[str]]:
        """Generate randomized IOCs scaled by curriculum level."""
        num_ips = rng.randint(1, min(level + 1, len(IP_POOL)))
        num_domains = rng.randint(1, min(level, len(DOMAIN_POOL)))
        num_hashes = rng.randint(1, min(level, len(HASH_POOL)))

        return {
            "malicious_ips": rng.sample(IP_POOL, num_ips),
            "malicious_domains": rng.sample(DOMAIN_POOL, num_domains),
            "file_hashes": rng.sample(HASH_POOL, num_hashes),
        }

    def _build_state(
        self,
        rng: random.Random,
        timeline: List[AttackTimeline],
        difficulty: float,
        level: int,
        iocs: Dict,
    ) -> Tuple[Dict, Dict]:
        """Build initial_state and ground_truth from the timeline."""

        # --- Alerts ---
        alerts = []
        gt_alerts = {}
        alert_type_map = {
            "initial_compromise": "Initial Access", "initial_access": "Initial Access",
            "reconnaissance": "Reconnaissance", "scanning": "Reconnaissance",
            "execution": "Execution", "persistence": "Persistence",
            "privilege_escalation": "Privilege Escalation",
            "defense_evasion": "Defense Evasion",
            "credential_theft": "Credential Access", "credential_access": "Credential Access",
            "discovery": "Discovery", "lateral_movement": "Lateral Movement",
            "collection": "Collection", "staging": "Staging",
            "exfiltration": "Exfiltration", "impact": "Impact",
            "c2_communication": "Command and Control", "attack": "Impact",
            "abuse_of_access": "Initial Access",
        }

        for entry in timeline:
            if entry.alert_generated:
                alert_id = entry.alert_generated["id"]
                alert = {
                    "alert_id": alert_id,
                    "description": entry.description,
                    "confidence": entry.alert_generated["confidence"],
                    "alert_type": alert_type_map.get(entry.phase, "Unknown"),
                    "source_host": f"host-{rng.randint(1, 3 + level // 2):02d}",
                    "timestamp": f"2026-04-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
                }
                alerts.append(alert)
                gt_alerts[alert_id] = {"is_true_positive": True}

        # Add false-positive noise alerts (more at higher levels)
        num_fp = rng.randint(1, 2 + level // 2)
        fp_types = ["Benign Admin Activity", "Scheduled Maintenance", "Automated Scan",
                     "Health Check Probe", "Backup Process"]
        for i in range(num_fp):
            fp_id = f"ALT-FP-{rng.randint(50000, 59999)}"
            alerts.append({
                "alert_id": fp_id,
                "description": f"{rng.choice(fp_types)} detected",
                "confidence": round(rng.uniform(0.15, 0.55), 2),
                "alert_type": rng.choice(fp_types),
                "source_host": f"host-{rng.randint(1, 3 + level // 2):02d}",
                "timestamp": f"2026-04-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
            })
            gt_alerts[fp_id] = {"is_true_positive": False}

        # Shuffle alert order so agent can't rely on position
        rng.shuffle(alerts)

        # --- Hosts ---
        num_hosts = rng.randint(2, 3 + level // 2)
        roles = ["web", "database", "app", "ci-runner", "mail", "vpn-gw", "file-server"]
        hosts = []
        gt_hosts = {}
        for i in range(1, num_hosts + 1):
            host_id = f"host-{i:02d}"
            role = rng.choice(roles)
            compromised = rng.random() < (0.3 + difficulty * 0.3)
            c2_active = compromised and rng.random() < (0.2 + difficulty * 0.3)
            hosts.append({
                "host_id": host_id,
                "hostname": f"{role}-{i}",
                "ip": f"10.0.{rng.randint(0, 3)}.{10 + i}",
                "role": role,
                "status": "online",
                "criticality": rng.choice(["low", "medium", "high", "critical"]),
            })
            gt_hosts[host_id] = {"compromised": compromised, "c2_active": c2_active}

        initial_state = {"alerts": alerts, "hosts": hosts}
        ground_truth = {"alerts": gt_alerts, "hosts": gt_hosts}

        return initial_state, ground_truth
