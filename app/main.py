"""
AnomalyGuard — FastAPI Application
Wraps AnomalyGuardEnvironment with OpenEnv-compliant HTTP API.
NO dashboard. NO HTML. Pure RL environment API only.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from .models import Action, TaskGraderResult
from .environment import AnomalyGuardEnvironment
from .grader import grade_episode

app = FastAPI(
    title="AnomalyGuard",
    description="Explainable AI RL Environment for Cybersecurity — OpenEnv Compatible",
    version="1.0.0",
)

_env = AnomalyGuardEnvironment()


@app.get("/")
def root():
    return {
        "name": "AnomalyGuard",
        "version": "1.0.0",
        "description": "Explainable AI RL environment for cybersecurity incident response",
        "openenv": True,
        "tasks": 3,
        "endpoints": {
            "reset":    "POST /reset?task_id=1&seed=42",
            "step":     "POST /step",
            "state":    "GET /state",
            "grader":   "POST /grader?task_id=1",
            "tasks":    "GET /tasks",
            "health":   "GET /health",
            "baseline": "POST /baseline?task_id=1",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "AnomalyGuard", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": 1, "name": "alert_triage", "difficulty": "easy",
                "description": "Triage 6-10 SIEM alerts with evidence-based justifications",
                "max_steps": 15, "target_score": 0.6,
            },
            {
                "id": 2, "name": "incident_containment", "difficulty": "medium",
                "description": "Contain active breach across 10-15 hosts",
                "max_steps": 20, "target_score": 0.5,
            },
            {
                "id": 3, "name": "full_incident_response", "difficulty": "hard",
                "description": "Full IR lifecycle: detect, contain, eradicate, recover",
                "max_steps": 30, "target_score": 0.4,
            },
        ],
    }


@app.post("/reset")
def reset(task_id: Optional[int] = Query(None, ge=1, le=3), seed: int = Query(42)):
    obs = _env.reset(task_id=task_id, seed=seed)
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    """Execute action via OpenEnv interface"""
    try:
        # Call OpenEnv step (returns 5-tuple)
        observation, reward, terminated, truncated, info = _env.step(action)
        
        # Convert observation to dict
        if hasattr(observation, 'model_dump'):
            obs_dict = observation.model_dump()
        elif hasattr(observation, 'dict'):
            obs_dict = observation.dict()
        else:
            obs_dict = observation
        
        return {
            "observation": obs_dict,
            "reward": float(reward),
            "done": bool(terminated),
            "truncated": bool(truncated),
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
def get_state():
    try:
        obs = _env.get_state()
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grader")
def grade(task_id: int = Query(..., ge=1, le=3)):
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode")
    result = grade_episode(_env._state, task_id)
    return result.model_dump()


@app.post("/baseline")
def run_baseline(task_id: int = Query(1, ge=1, le=3), seed: int = Query(42)):
    from .baseline import run_rule_based_baseline
    return run_rule_based_baseline(task_id=task_id, seed=seed, env=_env)


@app.get("/difficulty")
def get_difficulty():
    return {"current_difficulty": _env._difficulty}


# ═══════════════════════════════════════════════════════════════════
# EU AI Act Compliance Endpoints
# ═══════════════════════════════════════════════════════════════════

import logging
logger = logging.getLogger(__name__)

@app.get("/compliance/audit", response_model=dict)
def get_compliance_audit():
    """
    Returns EU AI Act Article 14 compliance audit report.
    
    Evaluates the current episode against 5 compliance checks:
    1. All actions justified (Article 14.4)
    2. Explanation quality adequate (Article 13.1)
    3. Human oversight available (Article 14.1)
    4. High-risk actions documented (Article 14.4(c))
    5. No classification bias (Article 10.2(f))
    
    Returns:
        AuditReport with compliance_checks, overall compliant status, and risk_level
        
    Raises:
        HTTPException 400: If no active episode (call /reset first)
        HTTPException 500: If audit generation fails
    """
    if _env._state is None:
        raise HTTPException(
            status_code=400, 
            detail="No active episode. Call POST /reset first."
        )
    
    try:
        report = _env.generate_audit_report()
        return report.model_dump()
    except Exception as e:
        logger.error("Audit report generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Audit generation failed: {str(e)}"
        )


@app.get("/compliance/trail", response_model=dict)
def get_audit_trail():
    """
    Returns the full action audit trail for regulatory inspection.
    
    Each entry includes:
    - step, action_type, target
    - correctness, explanation quality, reward
    - justification metadata (reasoning length, evidence count, risk assessment)
    - timestamp
    
    Returns:
        Dictionary with task_id, seed, steps, and full action_history
        
    Raises:
        HTTPException 400: If no active episode
    """
    if _env._state is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    
    return {
        "task_id": _env._task_id,
        "seed": _env._seed,
        "steps": _env._step_count,
        "actions": _env._state["action_history"],
        "total_reward": _env._state.get("total_reward", 0.0),
    }

@app.get("/host/{host_id}/dependencies", response_model=dict)
def get_restore_dependencies(host_id: str):
    """Check what's blocking restoration for Feature 4"""
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode.")
        
    # Validation logic from Feature 4
    can_restore, blocking_issues = _env._validate_restore_dependencies(host_id)
    
    return {
        'host_id': host_id,
        'can_restore': can_restore,
        'blocking_issues': blocking_issues,
        'ready': can_restore
    }

@app.get("/metrics/spread", response_model=dict)
def get_spread_metrics():
    """Get malware spread statistics for Feature 2"""
    return {
        'total_infected': len(_env._infected_hosts),
        'infected_hosts': list(_env._infected_hosts),
        'spread_events': _env._spread_history[-10:],  # Last 10 events
        'spread_probability': _env._spread_probability,
        'network_topology': _env._network_graph
    }

@app.get("/curriculum/status", response_model=dict)
def get_curriculum_status():
    """Get current curriculum level and performance"""
    return {
        "current_level": _env._curriculum_level,
        "difficulty_tier": _env._get_level_parameters(_env._curriculum_level)["difficulty_tier"],
        "total_episodes": _env._total_episodes,
        "episodes_at_current_level": _env._episodes_at_current_level,
        "recent_scores": _env._episode_history,
        "average_recent_score": (
            round(sum(_env._episode_history) / len(_env._episode_history), 3)
            if _env._episode_history else 0.0
        ),
        "level_parameters": _env._get_level_parameters(_env._curriculum_level),
        "thresholds": _env._curriculum_config,
        # Legacy fields
        "enabled": _env._curriculum_enabled,
        "current_difficulty": _env._difficulty,
    }

@app.get("/episodes/diversity", response_model=dict)
def get_episode_diversity():
    """Get episode diversity statistics"""
    return _env.get_diversity_stats()

@app.post("/curriculum/toggle", response_model=dict)
def toggle_curriculum():
    """Enable/disable curriculum"""
    _env._curriculum_enabled = not _env._curriculum_enabled
    return {
        'curriculum_enabled': _env._curriculum_enabled
    }

def start_server():
    """
    Entry point for OpenEnv deployment.
    This function is called when running: server
    """
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
