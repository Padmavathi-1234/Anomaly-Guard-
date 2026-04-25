"""
Procedural Attack Generator
==============================
Generates novel, non-repeating attack scenarios using compositional
building blocks from MITRE ATT&CK.
"""
from __future__ import annotations
import hashlib, random as _random
from typing import Any, Dict, List, Optional

MITRE_TECH = {
    "T1595": {"name": "Active Scanning", "tactic": "Recon", "requires": []},
    "T1566.001": {"name": "Spearphishing Attachment", "tactic": "Initial Access", "requires": []},
    "T1190": {"name": "Exploit Public-Facing App", "tactic": "Initial Access", "requires": []},
    "T1078": {"name": "Valid Accounts", "tactic": "Initial Access", "requires": []},
    "T1195.001": {"name": "Supply Chain Compromise", "tactic": "Initial Access", "requires": []},
    "T1059.001": {"name": "PowerShell", "tactic": "Execution", "requires": ["T1566.001", "T1190", "T1078"]},
    "T1059.003": {"name": "Windows Cmd Shell", "tactic": "Execution", "requires": ["T1190", "T1078"]},
    "T1505.003": {"name": "Web Shell", "tactic": "Persistence", "requires": ["T1190"]},
    "T1547.001": {"name": "Registry Run Keys", "tactic": "Persistence", "requires": ["T1059.001"]},
    "T1543.003": {"name": "Windows Service", "tactic": "Persistence", "requires": ["T1059.001"]},
    "T1098": {"name": "Account Manipulation", "tactic": "Persistence", "requires": ["T1078"]},
    "T1548.002": {"name": "UAC Bypass", "tactic": "Priv Esc", "requires": ["T1059.001"]},
    "T1562.001": {"name": "Disable Security Tools", "tactic": "Def Evasion", "requires": ["T1548.002"]},
    "T1027": {"name": "Obfuscated Files", "tactic": "Def Evasion", "requires": ["T1059.001"]},
    "T1003.001": {"name": "LSASS Memory", "tactic": "Cred Access", "requires": ["T1548.002"]},
    "T1552.001": {"name": "Credentials In Files", "tactic": "Cred Access", "requires": ["T1059.001"]},
    "T1087": {"name": "Account Discovery", "tactic": "Discovery", "requires": ["T1059.001"]},
    "T1083": {"name": "File Discovery", "tactic": "Discovery", "requires": ["T1059.001"]},
    "T1580": {"name": "Cloud Infra Discovery", "tactic": "Discovery", "requires": ["T1078"]},
    "T1021.001": {"name": "RDP", "tactic": "Lateral Movement", "requires": ["T1003.001"]},
    "T1021.002": {"name": "SMB Shares", "tactic": "Lateral Movement", "requires": ["T1003.001"]},
    "T1005": {"name": "Local Data", "tactic": "Collection", "requires": ["T1083"]},
    "T1530": {"name": "Cloud Storage Data", "tactic": "Collection", "requires": ["T1580"]},
    "T1560.001": {"name": "Archive via Utility", "tactic": "Collection", "requires": ["T1005"]},
    "T1213": {"name": "Info Repos", "tactic": "Collection", "requires": ["T1078"]},
    "T1041": {"name": "Exfil Over C2", "tactic": "Exfiltration", "requires": ["T1005"]},
    "T1567.002": {"name": "Exfil to Cloud", "tactic": "Exfiltration", "requires": ["T1560.001"]},
    "T1048.003": {"name": "Exfil Unencrypted", "tactic": "Exfiltration", "requires": ["T1005"]},
    "T1486": {"name": "Data Encrypted", "tactic": "Impact", "requires": ["T1021.001"]},
    "T1496": {"name": "Resource Hijacking", "tactic": "Impact", "requires": ["T1078"]},
}

ATTACK_CHAINS = {
    "supply_chain": {"entry": ["T1195.001"], "mid": ["T1059.001", "T1543.003", "T1552.001", "T1087"], "end": ["T1530", "T1567.002"]},
    "ransomware": {"entry": ["T1566.001", "T1190"], "mid": ["T1059.001", "T1548.002", "T1562.001", "T1021.001"], "end": ["T1486"]},
    "data_exfiltration": {"entry": ["T1566.001", "T1078"], "mid": ["T1059.001", "T1003.001", "T1083", "T1005"], "end": ["T1560.001", "T1041"]},
    "zero_day": {"entry": ["T1190"], "mid": ["T1059.003", "T1505.003", "T1003.001", "T1083"], "end": ["T1005", "T1041"]},
    "apt": {"entry": ["T1595", "T1078"], "mid": ["T1547.001", "T1027", "T1083", "T1560.001"], "end": ["T1041"]},
    "insider_threat": {"entry": ["T1078"], "mid": ["T1213", "T1005"], "end": ["T1048.003", "T1567.002"]},
    "cloud_attack": {"entry": ["T1078"], "mid": ["T1098", "T1580", "T1530"], "end": ["T1496"]},
}

_C2_IPS = ["185.220.{a}.{b}", "91.240.{a}.{b}", "45.155.{a}.{b}", "103.75.{a}.{b}", "194.163.{a}.{b}"]
_C2_DOMAINS = ["update-svc-{n}.xyz", "cdn-telemetry-{n}.cc", "api-health-{n}.ru", "secure-auth-{n}.net"]
_TOOLS = ["mimikatz", "Cobalt Strike", "BloodHound", "Rubeus", "PsExec", "Impacket", "PowerSploit", "LaZagne"]


class ProceduralAttackGenerator:
    """Generates novel attack scenarios with MITRE ATT&CK technique chains."""

    def __init__(self):
        self._generated_hashes: set = set()

    def generate(self, seed: int, difficulty: float = 0.5, pattern: Optional[str] = None) -> Dict[str, Any]:
        rng = _random.Random(seed)
        selected = pattern if pattern and pattern in ATTACK_CHAINS else rng.choice(list(ATTACK_CHAINS.keys()))
        chain_tpl = ATTACK_CHAINS[selected]
        path_len = rng.randint(3, 3 + int(difficulty * 7))
        techniques = self._build_chain(rng, chain_tpl, path_len, difficulty)
        iocs = self._gen_iocs(rng, difficulty)
        timeline = self._build_timeline(rng, techniques, iocs, difficulty)
        prev_windows = self._gen_prevention(rng, timeline, difficulty)
        h = hashlib.sha256(f"{selected}:{','.join(techniques)}:{seed}".encode()).hexdigest()[:16]
        self._generated_hashes.add(h)
        return {
            "pattern": selected, "pattern_display": selected.replace("_", " ").title(),
            "techniques": techniques,
            "technique_names": [MITRE_TECH.get(t, {}).get("name", t) for t in techniques],
            "iocs": iocs, "timeline": timeline, "prevention_windows": prev_windows,
            "difficulty": difficulty, "path_length": len(techniques), "scenario_hash": h,
        }

    def _build_chain(self, rng, tpl, target_len, difficulty):
        chain = [rng.choice(tpl["entry"])]
        mid = list(tpl["mid"]); rng.shuffle(mid)
        for t in mid:
            if len(chain) >= target_len - 1: break
            deps = MITRE_TECH.get(t, {}).get("requires", [])
            if not deps or any(d in chain for d in deps):
                chain.append(t)
        if difficulty > 0.6 and len(chain) < target_len - 1:
            extras = list(MITRE_TECH.keys()); rng.shuffle(extras)
            for t in extras:
                if t in chain or len(chain) >= target_len - 1: continue
                deps = MITRE_TECH[t].get("requires", [])
                if not deps or any(d in chain for d in deps):
                    chain.append(t)
        chain.append(rng.choice(tpl["end"]))
        return chain

    def _gen_iocs(self, rng, difficulty):
        ips = [rng.choice(_C2_IPS).format(a=rng.randint(1,254), b=rng.randint(1,254)) for _ in range(rng.randint(1, 2+int(difficulty*4)))]
        domains = [rng.choice(_C2_DOMAINS).format(n=rng.randint(100,999)) for _ in range(rng.randint(1, 1+int(difficulty*3)))]
        hashes = [hashlib.md5(f"mal-{rng.randint(0,999999)}".encode()).hexdigest() for _ in range(rng.randint(1, 2+int(difficulty*3)))]
        tools = rng.sample(_TOOLS, min(rng.randint(1, 1+int(difficulty*2)), len(_TOOLS)))
        return {"malicious_ips": ips, "malicious_domains": domains, "file_hashes": hashes, "tools_used": tools}

    def _build_timeline(self, rng, techniques, iocs, difficulty):
        timeline, hour, minute = [], rng.randint(0,12), rng.randint(0,59)
        obs_map = {
            "Recon": ["Port scan from {ip}", "DNS enumeration detected"],
            "Initial Access": ["Suspicious email from {domain}", "Exploit on port {port}"],
            "Execution": ["PowerShell encoded command", "Script interpreter invoked"],
            "Persistence": ["Registry autostart entry", "Scheduled task created"],
            "Priv Esc": ["UAC bypass detected", "Elevated process spawned"],
            "Def Evasion": ["AV service stopped", "Event log cleared"],
            "Cred Access": ["LSASS access detected", "Credential file accessed"],
            "Discovery": ["Network share enumeration", "AD query burst"],
            "Lateral Movement": ["RDP session to {host}", "SMB admin share access"],
            "Collection": ["Bulk file access", "Archive created in temp"],
            "Exfiltration": ["Large outbound to {ip}", "DNS tunneling detected"],
            "Impact": ["Mass file encryption", "Crypto-mining process"],
        }
        for idx, tech_id in enumerate(techniques):
            info = MITRE_TECH.get(tech_id, {})
            tactic = info.get("tactic", "Unknown")
            minute += rng.randint(5, 45)
            if minute >= 60: hour += minute // 60; minute %= 60
            templates = obs_map.get(tactic, ["Suspicious activity"])
            num_obs = max(1, 3 - int(difficulty * 2))
            obs = [t.format(ip=rng.choice(iocs.get("malicious_ips",["10.0.0.1"])),
                           domain=rng.choice(iocs.get("malicious_domains",["unknown.xyz"])),
                           port=rng.choice([22,80,443,445,3389]),
                           host=f"HOST-{rng.randint(1,10):03d}")
                   for t in rng.sample(templates, min(num_obs, len(templates)))]
            conf = round(rng.uniform(0.4, 0.95) * (1.0 - difficulty * 0.4), 2)
            has_alert = rng.random() < max(0.3, 0.8 - difficulty * 0.4)
            timeline.append({
                "step": idx, "technique_id": tech_id,
                "technique_name": info.get("name", tech_id), "tactic": tactic,
                "timestamp": f"2026-04-25T{hour:02d}:{minute:02d}:00Z",
                "observables": obs, "generates_alert": has_alert,
                "alert_confidence": conf if has_alert else 0.0,
                "is_critical": tactic in ("Impact", "Exfiltration", "Lateral Movement"),
            })
        return timeline

    def _gen_prevention(self, rng, timeline, difficulty):
        windows = []
        chance = max(0.15, 0.6 - difficulty * 0.4)
        for e in timeline:
            if rng.random() < chance:
                windows.append({
                    "at_step": e["step"], "technique": e["technique_id"],
                    "action_required": rng.choice(["isolate_host", "block_ip", "disable_account", "query_host"]),
                    "window_steps": max(1, rng.randint(1, 3 - int(difficulty * 2))),
                    "reward_bonus": round(rng.uniform(0.1, 0.3), 2),
                })
        return windows
