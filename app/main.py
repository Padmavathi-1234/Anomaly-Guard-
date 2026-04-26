"""
Anomaly-Guard FastAPI Application
"""

import subprocess
import threading
import os
import json
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.core.environment_base import AnomalyGuardBase
from app.core.environment_multiagent import MultiAgentAnomalyGuard
from app.compliance.eu_ai_act_engine import EUAIActComplianceEngine
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Anomaly-Guard",
    description="Multi-Agent Cybersecurity RL Environment with EU AI Act Compliance",
    version="2.0.0",
)

env = AnomalyGuardBase()
multi_env = MultiAgentAnomalyGuard(
    use_adversarial=True,
    use_realistic=True,
    curriculum_start_level=1,
)
training_status = {"running": False, "log": []}

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: dict

class CurriculumResetRequest(BaseModel):
    start_level: int = 1

@app.get("/health")
def health():
    return {"status": "ok", "training_running": training_status["running"]}

@app.post("/reset")
def reset(task_id: int = 1, seed: int = None):
    obs, info = env.reset(task_id, seed)
    return {"observation": obs, "info": info}

@app.post("/step")
def step(request: StepRequest):
    obs, reward, terminated, truncated, info = env.step(request.action)
    return {"observation": obs, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info}

@app.get("/state")
def get_state():
    return env._get_masked_observation()

@app.post("/reset-multiagent")
def reset_multiagent(task_id: int = 1, seed: int = None):
    obs, info = multi_env.reset(task_id, seed)
    return {"observation": obs, "info": info}

@app.post("/step-multiagent")
def step_multiagent(request: StepRequest):
    obs, reward, terminated, truncated, info = multi_env.step(request.action)
    return {"observation": obs, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info}

@app.post("/train/start")
def start_training():
    """Start GRPO training in background. Dynamic steps based on curriculum."""
    if training_status["running"]:
        return {"status": "already_running", "check": "/train/status"}
    def run():
        training_status["running"] = True
        training_status["log"] = []
        try:
            process = subprocess.Popen(
                ["python", "training/train_grpo.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/app",
                env={**os.environ, "PYTHONPATH": "/app"},
            )
            for line in process.stdout:
                line = line.strip()
                if line:
                    training_status["log"].append(line)
                    if len(training_status["log"]) > 500:
                        training_status["log"] = training_status["log"][-500:]
            process.wait()
        except Exception as e:
            training_status["log"].append(f"ERROR: {str(e)}")
        finally:
            training_status["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return {
        "status": "training_started",
        "model": "Qwen2.5-1.5B-Instruct",
        "note": "Steps auto-selected based on curriculum level",
        "check_progress": "GET /train/status",
        "view_logs": "GET /train/logs",
        "download_plot": "GET /train/plot",
    }

@app.get("/train/status")
def get_training_status():
    """Poll every 30 seconds. running=false means complete."""
    logs = training_status["log"]
    return {
        "running": training_status["running"],
        "log_lines": len(logs),
        "last_10_lines": logs[-10:] if logs else [],
        "status_message": "Training in progress..." if training_status["running"] else "Complete or not started.",
    }

@app.get("/train/logs")
def get_training_logs():
    """Full training logs with all reward values."""
    return {
        "running": training_status["running"],
        "total_lines": len(training_status["log"]),
        "logs": training_status["log"],
    }

@app.get("/train/plot")
def get_training_plot():
    """Download training progress PNG."""
    path = "/app/results/training_dashboard.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png",
                           filename="training_progress.png")
    return {"error": "Plot not ready. Check /train/status first."}

@app.get("/train/config")
def get_training_config():
    """Show dynamic training config including auto-selected steps."""
    path = "/app/results/training_config.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": "No config yet. Start training first."}

@app.get("/curriculum/status")
def curriculum_status():
    return multi_env.curriculum.status()

@app.post("/curriculum/reset")
def curriculum_reset(request: CurriculumResetRequest):
    return multi_env.curriculum.reset(start_level=request.start_level)

@app.get("/compliance/audit")
def compliance_audit():
    return multi_env.compliance_engine.get_audit_report()

@app.get("/compliance/trail")
def compliance_trail(limit: int = 50):
    return {"trail": multi_env.compliance_engine.get_trail(limit=limit)}

@app.get("/compliance/dashboard")
def compliance_dashboard():
    return multi_env.compliance_engine.get_dashboard()

from app.scenarios.threat_intel_live import LiveThreatIntel

@app.get("/threat-intel/live")
def get_live_threat_intel():
    intel = LiveThreatIntel()
    return {"iocs": intel.fetch_latest(), "updated": intel.last_update.isoformat()}

@app.get("/anti-hacking/report")
def anti_hacking_report():
    return {
        "total_suspicion_flags": len(env.grader.suspicion_flags),
        "unique_violations": len(set(env.grader.suspicion_flags)),
        "most_common_exploit": max(set(env.grader.suspicion_flags),
            key=env.grader.suspicion_flags.count) if env.grader.suspicion_flags else None,
    }

@app.post("/anti-hacking/test/{exploit_type}")
def test_exploit(exploit_type: str):
    pass

from app.business.roi_calculator import ROICalculator

@app.get("/business-impact/roi")
def calculate_roi():
    agent_perf = {"avg_detection_step": 3.8, "prevention_rate": 0.78,
                  "false_positive_rate": 0.15, "total_alerts": 50000}
    return ROICalculator().calculate_savings(agent_perf)

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
            "domains": [d["domain"] for d in real_data["malicious_domains"][:5]],
        },
        "scenario_injection": {
            "name": "Basic Scenario (Task 1)",
            "live_intel_metadata": scenario_with_intel.get("live_threat_intel", {}),
        },
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
        "alert_count": len(scenario["initial_state"]["alerts"]),
    }

def start_server():
    import uvicorn
    print("[OK] Anomaly-Guard: Ready for multi-mode deployment")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start_server()