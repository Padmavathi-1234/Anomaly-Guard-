import pytest
from app.core.environment_base import AnomalyGuardBase

def test_environment_reproducibility():
    env1 = AnomalyGuardBase()
    obs1, info1 = env1.reset(task_id=1, seed=42)
    
    env2 = AnomalyGuardBase()
    obs2, info2 = env2.reset(task_id=1, seed=42)
    
    assert obs1 == obs2
    assert env1.ground_truth == env2.ground_truth

def test_masked_observation():
    env = AnomalyGuardBase()
    obs, info = env.reset(task_id=1, seed=42)
    
    for alert in obs["alerts"]:
        assert "is_true_positive" not in alert
        assert "mitre_technique" not in alert
        
    for host in obs["hosts"]:
        assert "compromised" not in host
        assert "c2_active" not in host

def test_query_host():
    env = AnomalyGuardBase()
    obs, info = env.reset(task_id=1, seed=42)
    
    host_id_to_query = obs["hosts"][0]["host_id"]
    action = {"action_type": "query_host", "target_id": host_id_to_query}
    
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    assert host_id_to_query in new_obs["query_history"]
    
    queried_host = next(h for h in new_obs["hosts"] if h["host_id"] == host_id_to_query)
    assert "compromised" in queried_host
    assert "c2_active" in queried_host

def test_all_actions():
    env = AnomalyGuardBase()
    obs, info = env.reset(task_id=1, seed=42)
    
    alert_id = obs["alerts"][0]["alert_id"]
    
    # triage_alert
    action1 = {"action_type": "triage_alert", "target_id": alert_id, "classification": "TP"}
    obs, r, term, trunc, info = env.step(action1)
    assert info["action_executed"] == "triage_alert"
    
    # query_host
    action2 = {"action_type": "query_host", "target_id": "HOST-001"}
    obs, r, term, trunc, info = env.step(action2)
    assert info["action_executed"] == "query_host"
    
    # isolate_host
    action3 = {"action_type": "isolate_host", "target_id": "HOST-001"}
    obs, r, term, trunc, info = env.step(action3)
    assert info["action_executed"] == "isolate_host"
    assert next(h for h in obs["hosts"] if h["host_id"] == "HOST-001")["status"] == "isolated"
    
    # block_ip
    action4 = {"action_type": "block_ip", "ip": "1.2.3.4"}
    obs, r, term, trunc, info = env.step(action4)
    assert info["action_executed"] == "block_ip"
    
    # escalate_to_human
    action5 = {"action_type": "escalate_to_human"}
    obs, r, term, trunc, info = env.step(action5)
    assert info["action_executed"] == "escalate_to_human"
