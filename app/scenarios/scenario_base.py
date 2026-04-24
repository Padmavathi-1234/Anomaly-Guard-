import random
from typing import Dict, Any

def generate_basic_scenario(task_id: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    
    # Generate Task 1 (Alert Triage)
    num_alerts = rng.randint(6, 10)
    
    tp_types = ["C2_Communication", "Credential_Dumping", "Lateral_Movement"]
    fp_types = ["Benign_Admin_Activity", "Scheduled_Task"]
    severities = ["low", "medium", "high", "critical"]
    
    alerts = []
    ground_truth_alerts = {}
    
    for i in range(num_alerts):
        alert_id = f"ALT-{10001 + i}"
        
        is_true_positive = rng.random() < 0.6
        if is_true_positive:
            alert_type = rng.choice(tp_types)
            mitre = "T1071" if alert_type == "C2_Communication" else "T1003" if alert_type == "Credential_Dumping" else "T1570"
        else:
            alert_type = rng.choice(fp_types)
            mitre = None
            
        alert = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "severity": rng.choice(severities),
            "confidence": round(rng.uniform(0.5, 0.99), 2),
            "description": f"Detected {alert_type} on host",
            "source_host": f"HOST-{rng.randint(1, 5):03d}",
            "timestamp": f"2026-04-23T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z"
        }
        
        alerts.append(alert)
        ground_truth_alerts[alert_id] = {
            "is_true_positive": is_true_positive,
            "mitre_technique": mitre
        }
        
    hosts = []
    ground_truth_hosts = {}
    roles = ["web", "database", "app"]
    
    for i in range(1, 6):
        host_id = f"HOST-{i:03d}"
        role = rng.choice(roles)
        
        compromised = rng.random() < 0.4
        c2_active = compromised and rng.random() < 0.5
        
        host = {
            "host_id": host_id,
            "hostname": f"{role}-server-{i}",
            "ip_address": f"10.0.0.{10+i}",
            "role": role,
            "criticality": "high" if role == "database" else "medium",
            "status": "active"
        }
        
        hosts.append(host)
        ground_truth_hosts[host_id] = {
            "compromised": compromised,
            "c2_active": c2_active
        }
        
    return {
        "task_id": task_id,
        "max_steps": 15,
        "initial_state": {
            "alerts": alerts,
            "hosts": hosts
        },
        "ground_truth": {
            "alerts": ground_truth_alerts,
            "hosts": ground_truth_hosts
        }
    }
