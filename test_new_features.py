"""Quick validation script for all new features."""

print("=" * 60)
print("TESTING ALL NEW FEATURES")
print("=" * 60)

# ---- 1. Dynamic Realistic Attacks ----
print("\n[1] Dynamic Realistic Attacks")
from app.scenarios.realistic_attacks import RealisticScenarioGenerator

gen = RealisticScenarioGenerator()

s1 = gen.generate(difficulty=0.3, seed=42, curriculum_level=1)
print(f"  Level 1: {s1['name']} | type={s1['attack_type']} | steps={s1['max_steps']} | alerts={len(s1['initial_state']['alerts'])}")

s4 = gen.generate(difficulty=0.6, seed=42, curriculum_level=4)
print(f"  Level 4: {s4['name']} | type={s4['attack_type']} | steps={s4['max_steps']} | alerts={len(s4['initial_state']['alerts'])}")

s8 = gen.generate(difficulty=0.95, seed=42, curriculum_level=8)
print(f"  Level 8: {s8['name']} | type={s8['attack_type']} | steps={s8['max_steps']} | alerts={len(s8['initial_state']['alerts'])}")

# Reproducibility
s8b = gen.generate(difficulty=0.95, seed=42, curriculum_level=8)
assert s8["name"] == s8b["name"], "Reproducibility FAILED"
assert s8["max_steps"] == s8b["max_steps"], "Reproducibility FAILED"
print("  Reproducibility: PASS")

# Forced attack type
sf = gen.generate(difficulty=0.5, seed=99, attack_type="ransomware")
assert sf["attack_type"] == "ransomware"
print(f"  Forced type: {sf['name']} | type={sf['attack_type']}")

# Long horizon check
assert s1["max_steps"] <= 20, f"Level 1 should be <=20, got {s1['max_steps']}"
assert s4["max_steps"] <= 40, f"Level 4 should be <=40, got {s4['max_steps']}"
assert s8["max_steps"] <= 80, f"Level 8 should be <=80, got {s8['max_steps']}"
print(f"  Long-horizon: L1={s1['max_steps']} L4={s4['max_steps']} L8={s8['max_steps']} - PASS")
print("  [OK] All realistic attacks tests passed")

# ---- 2. Curriculum Manager ----
print("\n[2] Self-Evolving Curriculum Manager")
from app.core.curriculum_manager import CurriculumManager

cm = CurriculumManager(window_size=10, start_level=1)
assert cm.current_level == 1
print(f"  Initial level: {cm.current_level} ({cm.level_config.name})")
print(f"  Max steps: {cm.max_steps}")

# Simulate high rewards to trigger promotion
for i in range(12):
    result = cm.record_episode(0.85)
print(f"  After 12 good episodes: level={cm.current_level} result={result['action']}")
assert cm.current_level == 2, f"Expected level 2, got {cm.current_level}"

# Simulate poor rewards to trigger demotion
for i in range(12):
    result = cm.record_episode(0.2)
print(f"  After 12 poor episodes: level={cm.current_level} result={result['action']}")
assert cm.current_level == 1, f"Expected level 1, got {cm.current_level}"

# Status API
status = cm.status()
assert "current_level" in status
assert "avg_reward" in status
assert "level_history" in status
print(f"  Status keys: {list(status.keys())}")

# Reset
cm.reset(start_level=5)
assert cm.current_level == 5
print(f"  Reset to level 5: PASS")
print("  [OK] All curriculum tests passed")

# ---- 3. EU AI Act Compliance Engine ----
print("\n[3] EU AI Act Compliance Engine")
from app.compliance.eu_ai_act_engine import EUAIActComplianceEngine

engine = EUAIActComplianceEngine()

action = {"action_type": "isolate_host", "target_id": "host-01"}
justification = {
    "reasoning": "Host host-01 shows clear C2 communication patterns with known malicious IP. "
                 "Multiple high-confidence alerts confirm compromise. Isolating to prevent lateral movement.",
    "evidence": ["C2_Communication detected on host-01", "Malicious IP 185.220.101.45 contacted"],
    "confidence": 0.88,
    "human_review_requested": True,
    "reversible": True,
}
state = {
    "alerts": [
        {"alert_id": "ALT-001", "description": "C2_Communication detected on host-01", "confidence": 0.92},
    ],
    "hosts": [
        {"host_id": "host-01", "status": "online", "criticality": "high"},
    ],
    "query_history": ["host-01"],
}

record = engine.evaluate_action(action, justification, state, [action])
print(f"  Overall score: {record.overall_score}")
print(f"  Compliant: {record.compliant}")
print(f"  Risk level: {record.risk_level}")
for c in record.checks:
    print(f"    {c.dimension}: {c.score:.2f} ({'PASS' if c.passed else 'FAIL'})")

# Audit report
report = engine.get_audit_report()
assert report["total_actions_evaluated"] == 1
print(f"  Audit report: {report['total_actions_evaluated']} actions, rate={report['compliance_rate']}")

# Trail
trail = engine.get_trail()
assert len(trail) == 1
print(f"  Trail entries: {len(trail)}")

# Dashboard
dash = engine.get_dashboard()
assert "summary" in dash
assert "compliance_trend" in dash
print(f"  Dashboard keys: {list(dash.keys())}")
print("  [OK] All compliance tests passed")

# ---- 4. Integration Test ----
print("\n[4] Integration Test (MultiAgent + Curriculum + Compliance)")
from app.core.environment_multiagent import MultiAgentAnomalyGuard, AgentRole

env = MultiAgentAnomalyGuard(
    use_adversarial=False,
    use_realistic=True,
    curriculum_start_level=1,
)

obs, info = env.reset(task_id=1, seed=42)
assert "curriculum" in info
print(f"  Curriculum in reset info: level={info['curriculum']['current_level']}")
print(f"  Max steps: {env.max_steps}")
assert env.max_steps <= 20, f"Level 1 max_steps should be <=20, got {env.max_steps}"

# Take a step
alert_id = obs[AgentRole.TRIAGE]["alerts"][0]["alert_id"]
actions = {
    AgentRole.TRIAGE: {
        "action_type": "triage_alert",
        "target_id": alert_id,
        "classification": "TP",
        "justification": {
            "reasoning": f"Alert {alert_id} shows suspicious C2 communication activity on host requiring immediate triage classification as true positive.",
            "evidence": [{"source": alert_id}],
            "risk_assessment": {"threat_level": "HIGH"},
            "confidence": 0.75,
        },
    },
}
obs2, rewards, term, trunc, step_info = env.step(actions)
assert "compliance" in step_info
print(f"  Compliance in step: {step_info['compliance'].get(AgentRole.TRIAGE, {}).get('compliant', 'N/A')}")
print(f"  Triage reward: {rewards[AgentRole.TRIAGE]:.3f}")
print("  [OK] Integration test passed")

print("\n" + "=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY")
print("=" * 60)
