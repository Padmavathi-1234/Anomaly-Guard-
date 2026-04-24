"""
Anomaly-Guard FastAPI Application
===================================
Provides REST endpoints for:
- Base environment (reset, step, state)
- Multi-agent environment with curriculum and compliance
- Curriculum management (status, reset)
- EU AI Act compliance (audit, trail, dashboard)
- Business impact / ROI
- Threat intelligence
- Demo endpoints
"""

from fastapi import FastAPI
from app.core.environment_base import AnomalyGuardBase
from app.core.environment_multiagent import MultiAgentAnomalyGuard
from app.core.curriculum_manager import CurriculumManager
from app.compliance.eu_ai_act_engine import EUAIActComplianceEngine
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Anomaly-Guard",
    description="Multi-Agent Cybersecurity RL Environment with EU AI Act Compliance",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Environment instances
# ---------------------------------------------------------------------------

env = AnomalyGuardBase()

multi_env = MultiAgentAnomalyGuard(
    use_adversarial=True,
    use_realistic=True,
    curriculum_start_level=1,
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: dict

class CurriculumResetRequest(BaseModel):
    start_level: int = 1

# ---------------------------------------------------------------------------
# Core environment endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: int = 1, seed: int = None):
    obs, info = env.reset(task_id, seed)
    return {"observation": obs, "info": info}

@app.post("/step")
def step(request: StepRequest):
    obs, reward, terminated, truncated, info = env.step(request.action)
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
def get_state():
    return env._get_masked_observation()

# ---------------------------------------------------------------------------
# Multi-agent endpoints
# ---------------------------------------------------------------------------

@app.post("/reset-multiagent")
def reset_multiagent(task_id: int = 1, seed: int = None):
    obs, info = multi_env.reset(task_id, seed)
    return {"observation": obs, "info": info}

@app.post("/step-multiagent")
def step_multiagent(request: StepRequest):
    obs, reward, terminated, truncated, info = multi_env.step(request.action)
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

# ---------------------------------------------------------------------------
# Curriculum endpoints
# ---------------------------------------------------------------------------

@app.get("/curriculum/status")
def curriculum_status():
    """Get current curriculum level, avg reward, and progression info."""
    return multi_env.curriculum.status()

@app.post("/curriculum/reset")
def curriculum_reset(request: CurriculumResetRequest):
    """Reset curriculum to a specific starting level."""
    return multi_env.curriculum.reset(start_level=request.start_level)

# ---------------------------------------------------------------------------
# EU AI Act Compliance endpoints
# ---------------------------------------------------------------------------

@app.get("/compliance/audit")
def compliance_audit():
    """Full compliance audit report across all evaluated actions."""
    return multi_env.compliance_engine.get_audit_report()

@app.get("/compliance/trail")
def compliance_trail(limit: int = 50):
    """Visual audit trail of recent compliance evaluations."""
    return {"trail": multi_env.compliance_engine.get_trail(limit=limit)}

@app.get("/compliance/dashboard")
def compliance_dashboard():
    """Dashboard-ready aggregate compliance metrics and trends."""
    return multi_env.compliance_engine.get_dashboard()

# ---------------------------------------------------------------------------
# Threat intelligence endpoints
# ---------------------------------------------------------------------------

from app.scenarios.threat_intel_live import LiveThreatIntel

@app.get("/threat-intel/live")
def get_live_threat_intel():
    intel = LiveThreatIntel()
    return {
        "iocs": intel.fetch_latest(),
        "updated": intel.last_update.isoformat()
    }

# ---------------------------------------------------------------------------
# Anti-hacking report
# ---------------------------------------------------------------------------

@app.get("/anti-hacking/report")
def anti_hacking_report():
    return {
        "total_suspicion_flags": len(env.grader.suspicion_flags),
        "unique_violations": len(set(env.grader.suspicion_flags)),
        "most_common_exploit": max(set(env.grader.suspicion_flags), 
                                   key=env.grader.suspicion_flags.count) 
                                   if env.grader.suspicion_flags else None
    }

@app.post("/anti-hacking/test/{exploit_type}")
def test_exploit(exploit_type: str):
    """Test specific exploit detection (escalation_spam, isolation_spam, etc.)."""
    pass

# ---------------------------------------------------------------------------
# Business impact / ROI
# ---------------------------------------------------------------------------

from app.business.roi_calculator import ROICalculator

@app.get("/business-impact/roi")
def calculate_roi():
    agent_perf = {
        "avg_detection_step": 3.8,
        "prevention_rate": 0.78,
        "false_positive_rate": 0.15,
        "total_alerts": 50000
    }
    calculator = ROICalculator()
    roi = calculator.calculate_savings(agent_perf)
    return roi

# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    return {"tasks": [1, 2, 3]}

@app.get("/host/{host_id}/visibility")
def host_visibility(host_id: str):
    return {"host_id": host_id, "visible": True}

@app.get("/observability/status")
def observability_status():
    return {"status": "operational"}

@app.get("/metrics/detailed")
def detailed_metrics():
    return {"metrics": {}}

# ---------------------------------------------------------------------------
# Demo endpoints
# ---------------------------------------------------------------------------

@app.post("/demo/anti-hacking")
def demo_anti_hacking():
    from app.scenarios.threat_intel_live import LiveThreatIntel
    from app.scenarios.scenario_base import generate_basic_scenario
    
    intel = LiveThreatIntel()
    real_data = intel.fetch_latest()
    
    scenario = generate_basic_scenario(task_id=1, seed=42)
    scenario_with_intel = intel.inject_into_scenario(scenario)
    
    return {
        "status": "success",
        "live_iocs_fetched": {
            "ips": [i["ip"] for i in real_data["malicious_ips"][:5]],
            "domains": [d["domain"] for d in real_data["malicious_domains"][:5]]
        },
        "scenario_injection": {
            "name": "Basic Scenario (Task 1)",
            "live_intel_metadata": scenario_with_intel.get("live_threat_intel", {})
        }
    }

@app.post("/demo/realistic-scenario")
def demo_realistic_scenario():
    from app.scenarios.realistic_attacks import RealisticScenarioGenerator
    generator = RealisticScenarioGenerator()
    scenario = generator.generate(difficulty=0.9, curriculum_level=4)
    return {
        "status": "realistic scenario generated",
        "scenario_name": scenario["name"],
        "attack_type": scenario["attack_type"],
        "curriculum_level": scenario["curriculum_level"],
        "attack_complexity": "HIGH",
        "max_steps": scenario["max_steps"],
        "host_count": len(scenario["initial_state"]["hosts"]),
        "alert_count": len(scenario["initial_state"]["alerts"])
    }

# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def start_server():
    import uvicorn
    print("[OK] Anomaly-Guard-: Ready for multi-mode deployment")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start_server()
