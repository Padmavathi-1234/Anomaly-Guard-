import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class LiveThreatIntel:
    def __init__(self):
        self.last_update = datetime.min
        self.cache = {"malicious_ips": [], "malicious_domains": []}

    def fetch_latest(self) -> Dict[str, List[Dict]]:
        now = datetime.now()
        if now - self.last_update < timedelta(hours=1):
            return self.cache

        malicious_domains = []
        malicious_ips = []

        try:
            # URLhaus API
            url = "https://urlhaus-api.abuse.ch/v1/urls/recent/"
            response = requests.get(url, timeout=5)
            data = response.json()
            from urllib.parse import urlparse
            for entry in data.get("urls", [])[:20]:
                domain = urlparse(entry["url"]).netloc
                malicious_domains.append({
                    "url": entry["url"],
                    "domain": domain,
                    "threat_type": entry["threat"],
                    "first_seen": entry["dateadded"],
                    "source": "URLhaus",
                    "confidence": 0.95
                })
        except Exception:
            pass

        try:
            # ThreatFox API
            url = "https://threatfox-api.abuse.ch/api/v1/"
            payload = {"query": "get_iocs", "days": 1}
            response = requests.post(url, json=payload, timeout=5)
            data = response.json()
            for ioc in data.get("data", [])[:20]:
                if ioc["ioc_type"] == "ip:port":
                    malicious_ips.append({
                        "ip": ioc["ioc"].split(":")[0],
                        "malware": ioc.get("malware_printable"),
                        "source": "ThreatFox",
                        "confidence": 0.90
                    })
        except Exception:
            pass
            
        # If APIs fail completely, we return whatever we have (could be empty if both fail)
        if not malicious_ips and not malicious_domains:
             print("[WARNING] Live threat intel APIs failed. No IOCs available.")

        self.cache = {
            "malicious_ips": malicious_ips,
            "malicious_domains": malicious_domains
        }
        self.last_update = now
        return self.cache

    def inject_into_scenario(self, scenario: Dict) -> Dict:
        live_iocs = self.fetch_latest()
        
        # Replace fake IPs/domains with real ones from feeds if available
        injected_count = 0
        if live_iocs["malicious_ips"]:
            for alert in scenario.get("initial_state", {}).get("alerts", []):
                if any(k in alert.get("alert_type", "") for k in ["C2", "Initial", "Exfiltration"]):
                    real_ip = live_iocs["malicious_ips"][injected_count % len(live_iocs["malicious_ips"])]["ip"]
                    alert["description"] += f" (Live IOC: {real_ip})"
                    # Also update ground truth if it exists
                    alert_id = alert.get("alert_id")
                    if alert_id in scenario.get("ground_truth", {}).get("alerts", {}):
                        scenario["ground_truth"]["alerts"][alert_id]["ioc"] = real_ip
                    injected_count += 1
                    if injected_count > 3: break
        
        if live_iocs["malicious_domains"] and injected_count < 5:
            for alert in scenario.get("initial_state", {}).get("alerts", []):
                if "domain" in alert.get("description", "").lower() or "URL" in alert.get("alert_type", ""):
                    real_domain = live_iocs["malicious_domains"][0]["domain"]
                    alert["description"] += f" (Live Domain: {real_domain})"
                    injected_count += 1
                    break
        
        scenario["live_threat_intel"] = {
            "updated": datetime.now().isoformat(),
            "sources": ["URLhaus", "ThreatFox"],
            "ioc_count": len(live_iocs["malicious_ips"]) + len(live_iocs["malicious_domains"]),
            "injected_into_alerts": injected_count
        }
        
        return scenario
