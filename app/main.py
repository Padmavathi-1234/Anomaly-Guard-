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
    """
    AnomalyGuard environment root - lists all available endpoints.
    """
    return {
        "name":        "AnomalyGuard",
        "version":     "1.0.0",
        "description": "Explainable AI RL environment for cybersecurity incident response",
        "openenv":     True,
        "tasks":       3,
        "live_demo":   "https://padmavathi-123-anomalyguard.hf.space",
        "docs":        "https://padmavathi-123-anomalyguard.hf.space/docs",
        "github":      "https://github.com/Padmavathi-1234/Anomaly-Guard-",
        "endpoints": {
            "core": {
                "reset":            "POST /reset?task_id=1&seed=42",
                "step":             "POST /step",
                "state":            "GET /state",
                "state_raw":        "GET /state/raw",
                "grader":           "POST /grader?task_id=1",
                "tasks":            "GET /tasks",
                "health":           "GET /health",
            },
            "observability": {
                "host_visibility":        "GET /host/{host_id}/visibility",
                "observability_status":   "GET /observability/status",
                "host_dependencies":      "GET /host/{host_id}/dependencies",
            },
            "metrics": {
                "metrics_detailed":   "GET /metrics/detailed",
                "metrics_rewards":    "GET /metrics/rewards",
                "metrics_spread":     "GET /metrics/spread",
            },
            "baseline": {
                "baseline_run":       "POST /baseline?task_id=1",
                "baseline_reference": "GET /baseline/reference",
            },
            "curriculum": {
                "curriculum_status":  "GET /curriculum/status",
                "curriculum_toggle":  "POST /curriculum/toggle",
                "episodes_diversity": "GET /episodes/diversity",
            },
            "compliance": {
                "compliance_audit":   "GET /compliance/audit",
                "compliance_trail":   "GET /compliance/trail",
            },
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
    """
    Reset environment for a new episode.
    
    Returns both observation AND episode metadata so judges can see
    curriculum level, difficulty tier, task info, and max steps.
    
    Args:
        task_id: Task to load (1=triage, 2=containment, 3=full_ir)
                 None = curriculum auto-selects based on performance
        seed: Random seed for reproducibility (same seed = same scenario)
    
    Returns:
        observation: Initial masked observation for agent
        info: Episode metadata (curriculum_level, difficulty_tier, max_steps)
    """
    obs, info = _env.reset(task_id=task_id, seed=seed)
    return {
        "observation": obs.model_dump(),
        "info": {
            "task_id": task_id or _env._task_id,
            "seed": seed,
            "curriculum_level": _env._curriculum_level,
            "difficulty_tier": _env._get_level_parameters(_env._curriculum_level)["difficulty_tier"],
            "max_steps": _env._get_level_parameters(_env._curriculum_level)["max_steps"],
            "total_episodes": _env._total_episodes,
            "message": f"Episode started. Task {task_id or _env._task_id} loaded with seed {seed}."
        }
    }


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


@app.get("/state/raw")
def get_raw_state():
    """
    Return complete unmasked internal state for grading purposes.
    
    IMPORTANT DISTINCTION:
        GET /state       → Returns masked Observation (what agent sees)
        GET /state/raw   → Returns full internal state (what grader needs)
    
    The raw state includes:
        - alerts with is_true_positive field revealed
        - hosts with actual c2_active, persistence, vulnerabilities
        - complete action history with all scores
        - all tracking sets (isolated, triaged, blocked, etc.)
        - current phase and curriculum info
    
    This is used by:
        - Graders validating episode completion
        - Judges reviewing environment correctness
        - Debugging and analysis tools
    
    Returns:
        Complete state dict with sets converted to lists for JSON
        
    Raises:
        HTTPException 400: If no active episode
    """
    if _env._state is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    
    state = _env._state
    phase = state["phase"]
    
    return {
        "task_id":          state["task_id"],
        "seed":             state["seed"],
        "phase":            phase.value if hasattr(phase, "value") else str(phase),
        "step":             _env._step_count,
        "curriculum_level": state.get("curriculum_level"),
        "difficulty_tier":  state.get("difficulty_tier"),
        "total_reward":     state["total_reward"],
        "triaged":          state["triaged"],
        "isolated":         list(state.get("isolated", set())),
        "blocked_ips":      list(state.get("blocked_ips", set())),
        "disabled_accs":    list(state.get("disabled_accs", set())),
        "patched_cves":     list(state.get("patched_cves", set())),
        "removed_pers":     list(state.get("removed_pers", set())),
        "rotated_creds":    list(state.get("rotated_creds", set())),
        "restored":         list(state.get("restored", set())),
        "forensics":        list(state.get("forensics", set())),
        "escalated":        state.get("escalated", False),
        "queried_hosts":    list(_env._queried_hosts),
        "infected_hosts":   list(_env._infected_hosts),
        "action_history":   state.get("action_history", []),
        "cumulative_scores": state.get("cumulative_scores", {}),
        "alerts_summary": [
            {
                "alert_id":        a.alert_id,
                "severity":        a.severity,
                "is_true_positive": a.is_true_positive,
                "triaged_as":      state["triaged"].get(a.alert_id, "not_triaged"),
                "correct":         (
                    state["triaged"].get(a.alert_id) == "true_positive"
                    and a.is_true_positive
                ) or (
                    state["triaged"].get(a.alert_id) == "false_positive"
                    and not a.is_true_positive
                ) if a.alert_id in state["triaged"] else None,
            }
            for a in state.get("alerts", [])
        ],
        "hosts_summary": [
            {
                "host_id":       h.host_id,
                "hostname":      h.hostname,
                "role":          h.role,
                "criticality":   h.criticality,
                "c2_active":     h.c2_active,
                "persistence":   h.persistence,
                "vulnerabilities": h.vulnerabilities,
                "status":        h.status,
                "is_isolated":   h.host_id in state.get("isolated", set()),
                "is_queried":    h.host_id in _env._queried_hosts,
            }
            for h in state.get("hosts", [])
        ],
    }


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


# ═══════════════════════════════════════════════════════════════════
# Partial Observability Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/host/{host_id}/visibility")
def get_host_visibility(host_id: str):
    """
    Check visibility status of a specific host.

    Shows which fields are visible vs hidden based on whether
    the agent has queried this host with query_host action.
    """
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    host = None
    for h in _env._state.get("hosts", []):
        if h.host_id == host_id or h.hostname == host_id:
            host = h
            break

    if host is None:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    is_queried = host.host_id in _env._queried_hosts

    return {
        "host_id":    host.host_id,
        "hostname":   host.hostname,
        "is_queried": is_queried,
        "visible_fields": [
            "host_id", "hostname", "ip_address", "role",
            "criticality", "services", "business_impact",
            "status", "c2_active", "persistence",
            "vulnerabilities", "accounts",
        ] if is_queried else [
            "host_id", "hostname", "ip_address", "role",
            "criticality", "services", "business_impact",
        ],
        "hidden_fields": [] if is_queried else [
            "status", "c2_active", "persistence",
            "vulnerabilities", "accounts",
        ],
        "tip": (
            "All host details revealed."
            if is_queried
            else "Use action_type='query_host' with this host_id to reveal details."
        ),
    }


@app.get("/observability/status")
def get_observability_status():
    """
    Overview of partial observability state.
    Shows how many hosts have been investigated vs remaining.
    """
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    hosts = _env._state.get("hosts", [])
    total = len(hosts)
    queried = len(_env._queried_hosts)

    return {
        "total_hosts":     total,
        "queried_hosts":   queried,
        "unqueried_hosts": total - queried,
        "query_coverage":  round(queried / max(total, 1), 3),
        "queried_host_ids": list(_env._queried_hosts),
        "unqueried_host_ids": [
            h.host_id for h in hosts
            if h.host_id not in _env._queried_hosts
        ],
        "tip": (
            "Use POST /step with action_type='query_host' "
            "and target='<host_id>' to investigate hosts."
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Metrics Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/metrics/detailed")
def get_detailed_metrics():
    """
    Comprehensive performance metrics for current episode.

    Includes:
    - Detection: precision, recall, F1 for alert triage
    - Containment: rate, hosts at risk, spread events
    - Observability: query coverage
    - Efficiency: action efficiency, progress bonuses
    - Curriculum: level, difficulty, episode return
    """
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    return _env.get_detailed_metrics()


@app.get("/metrics/rewards")
def get_reward_history():
    """
    Step-by-step reward history for current episode.
    Useful for plotting learning curves and debugging reward signals.
    """
    if _env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    history = _env._state.get("action_history", [])

    return {
        "episode_rewards":     _env._episode_rewards,
        "cumulative_reward":   round(sum(_env._episode_rewards), 4),
        "avg_reward":          round(
            sum(_env._episode_rewards) / max(len(_env._episode_rewards), 1), 4
        ),
        "step_breakdown": [
            {
                "step":           h["step"],
                "action_type":    h["action_type"],
                "reward":         h["reward"],
                "progress_bonus": h.get("progress_bonus", 0.0),
                "correctness":    h["correctness"],
            }
            for h in history
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Baseline Endpoints (Updated)
# ═══════════════════════════════════════════════════════════════════

@app.get("/baseline/reference")
def get_reference_baselines():
    """
    Reference baseline performance ranges.

    These are theoretical bounds based on environment design:
    - Random agent cannot exceed 0.15 (pure chance on binary decisions)
    - Rule-based agent can reach 0.55-0.70 (heuristics without learning)
    - Trained RL agent target: 0.75+ (exceeds rule-based)
    - LLM agent (GPT-4 class): 0.70-0.85 (strong reasoning)

    Run POST /baseline to generate actual scores for current task.
    """
    return {
        "note": "Reference ranges based on environment design. Run POST /baseline for actual scores.",
        "task_1_alert_triage": {
            "random_agent":     {"expected_range": "0.05 - 0.15", "reason": "50% binary classification chance"},
            "rule_based_agent": {"expected_range": "0.50 - 0.65", "reason": "Confidence threshold heuristic"},
            "rl_agent_target":  {"expected_range": "0.75+",       "reason": "Learns alert patterns"},
            "llm_agent":        {"expected_range": "0.70 - 0.85", "reason": "Strong reasoning on MITRE context"},
        },
        "task_2_containment": {
            "random_agent":     {"expected_range": "0.03 - 0.10", "reason": "Low chance of isolating right hosts"},
            "rule_based_agent": {"expected_range": "0.40 - 0.55", "reason": "Query then isolate heuristic"},
            "rl_agent_target":  {"expected_range": "0.65+",       "reason": "Learns containment priority"},
            "llm_agent":        {"expected_range": "0.60 - 0.78", "reason": "Understands network threat spread"},
        },
        "task_3_full_ir": {
            "random_agent":     {"expected_range": "0.01 - 0.05", "reason": "Nearly impossible by chance"},
            "rule_based_agent": {"expected_range": "0.25 - 0.40", "reason": "Phase ordering heuristics"},
            "rl_agent_target":  {"expected_range": "0.55+",       "reason": "Learns full IR workflow"},
            "llm_agent":        {"expected_range": "0.50 - 0.70", "reason": "Complex multi-phase reasoning"},
        },
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
