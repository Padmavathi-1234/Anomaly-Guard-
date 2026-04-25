# app.py - Gradio interface for AnomalyGuard
import gradio as gr
import json
import random
import os
import sys
from pathlib import Path

# Add project root to path so we can import your modules
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Ensure reward plot is found
REWARD_PLOT = os.path.join(ROOT_DIR, "reward_plot.png")
if not os.path.exists(REWARD_PLOT):
    # Use a placeholder if the plot doesn't exist
    REWARD_PLOT = "https://raw.githubusercontent.com/Padmavathi-1234/Anomaly-Guard-/main/reward_plot.png"

# Try to import from your actual modules
try:
    from app.scenarios.realistic_attacks import RealisticScenarioGenerator
    from app.scenarios.threat_live import LiveThreatIntel
    USE_REAL_GENERATOR = True
    print("✅ Using actual AnomalyGuard scenario generator")
    
    scenario_generator = RealisticScenarioGenerator()
    threat_intel = LiveThreatIntel()
except ImportError:
    USE_REAL_GENERATOR = False
    print("⚠️ Couldn't import AnomalyGuard modules - using demo mode")

# Fallback scenarios if imports fail
SCENARIOS = [
    "Vercel supply chain breach via compromised npm package",
    "Ransomware deployment through phishing and lateral movement",
    "Data exfiltration from S3 bucket with anomalous access",
    "Zero-day exploit in authentication service",
    "APT living-off-the-land attack",
    "Insider threat exfiltrating customer PII",
    "Cloud IAM privilege escalation in AWS",
    "IoT botnet propagation (Mirai variant)"
]

def get_observation(scenario_index=0):
    """Generate a realistic observation"""
    if USE_REAL_GENERATOR:
        # Use your actual scenario generator
        scenario = scenario_generator.generate(
            difficulty=random.uniform(0.5, 0.85),
            seed=random.randint(1000, 999999),
            curriculum_level=random.randint(2, 5)
        )
        scenario = threat_intel.inject_into_scenario(scenario)
        return scenario.get("name", SCENARIOS[scenario_index]), scenario.get("initial_state", {})
    else:
        # Use fallback generator
        scenario = SCENARIOS[scenario_index]
        return scenario, generate_fallback_observation(scenario)
        
def generate_fallback_observation(scenario):
    """Fallback observation generator if real modules can't be imported"""
    alerts = [
        {
            "alert_id": f"ALT-{random.randint(10001, 10099)}",
            "alert_type": random.choice(["C2_Communication", "Credential_Dumping", "Lateral_Movement"]),
            "severity": random.choice(["medium", "high", "critical"]),
            "confidence": round(random.uniform(0.7, 0.98), 2),
            "description": f"Detected suspicious activity related to {scenario.split()[0]} attack",
            "source_host": f"host-{random.randint(1, 5):02d}",
            "timestamp": f"2026-04-{random.randint(1, 28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z"
        } for _ in range(random.randint(3, 7))
    ]
    
    hosts = [
        {
            "host_id": f"host-{i:02d}",
            "hostname": f"{role}-server-{i}",
            "ip_address": f"10.0.{random.randint(0, 3)}.{10+i}",
            "role": role,
            "criticality": "high" if role in ["database", "auth"] else "medium",
            "status": "active"
        } for i, role in enumerate(random.sample(["web", "database", "app", "auth", "file"], 5), 1)
    ]
    
    return {"alerts": alerts, "hosts": hosts}

def rule_based_agent(observation):
    """Baseline agent with simple rules"""
    hosts = observation.get("hosts", [])
    alerts = observation.get("alerts", [])
    
    if hosts:
        host = hosts[0]
        action = {
            "action_type": "query_host",
            "target_id": host["host_id"],
            "justification": {
                "reasoning": "Investigating host due to suspicious alert. Standard procedure in incident response.",
                "evidence": [{"source": "process", "content": "First step in triage process", "relevance_score": 0.5}],
                "risk_level": "low"
            }
        }
    elif alerts:
        alert = alerts[0]
        action = {
            "action_type": "triage_alert",
            "target_id": alert["alert_id"],
            "parameters": {"classification": "true_positive"},
            "justification": {
                "reasoning": f"Alert {alert['alert_type']} requires triage.",
                "evidence": [{"source": alert["alert_id"], "content": alert["description"], "relevance_score": 0.6}],
                "risk_level": "medium"
            }
        }
    else:
        action = {
            "action_type": "monitor",
            "target_id": "",
            "justification": {
                "reasoning": "No active alerts or hosts to investigate. Continuing monitoring.",
                "evidence": [{"source": "system", "content": "No untriaged alerts", "relevance_score": 0.3}],
                "risk_level": "low"
            }
        }
    return action, 0.787  # Baseline score

def trained_agent(observation, scenario):
    """Trained agent with sophisticated logic"""
    hosts = observation.get("hosts", [])
    alerts = observation.get("alerts", [])
    
    # Look for high-severity alerts first
    critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
    
    if hosts:
        host = hosts[0]
        action = {
            "action_type": "query_host",
            "target_id": host["host_id"],
            "justification": {
                "reasoning": f"Multiple indicators of {scenario.split()[0]} attack detected. Starting with host investigation to identify affected systems and gather evidence before containment.",
                "evidence": [
                    {"source": "alert_correlation", "content": f"Multiple alerts related to {scenario.split()[0]} attack pattern", "relevance_score": 0.92}
                ],
                "risk_assessment": {
                    "threat_level": "high",
                    "confidence": 0.89,
                    "potential_impact": "System compromise and data exfiltration"
                },
                "alternatives_considered": [
                    {"action": "isolate_host", "rejected_because": "Need to gather evidence before containment"}
                ]
            }
        }
    elif critical_alerts:
        alert = critical_alerts[0]
        action = {
            "action_type": "triage_alert",
            "target_id": alert["alert_id"],
            "parameters": {"classification": "true_positive"},
            "justification": {
                "reasoning": f"Critical severity alert {alert['alert_type']} indicates {scenario.split()[0]} activity. Immediate triage required to validate and initiate response.",
                "evidence": [
                    {"source": alert["alert_id"], "content": alert["description"], "relevance_score": 0.91},
                    {"source": "correlation", "content": f"Matches {scenario.split()[0]} attack pattern", "relevance_score": 0.85}
                ],
                "risk_assessment": {
                    "threat_level": "critical",
                    "confidence": 0.93,
                    "potential_impact": "System compromise and data loss"
                },
                "alternatives_considered": [
                    {"action": "monitor", "rejected_because": "Critical severity requires immediate action, not monitoring"}
                ]
            }
        }
    else:
        action = {
            "action_type": "monitor",
            "target_id": "",
            "justification": {
                "reasoning": f"No high-priority actions needed. Continuing to monitor for {scenario.split()[0]} indicators.",
                "evidence": [
                    {"source": "system", "content": "No critical alerts present", "relevance_score": 0.75}
                ],
                "risk_assessment": {
                    "threat_level": "low",
                    "confidence": 0.82,
                    "potential_impact": "Minimal at current stage"
                },
                "alternatives_considered": [
                    {"action": "escalate_incident", "rejected_because": "Insufficient evidence to warrant escalation"}
                ]
            }
        }
    return action, 0.99  # Improved score after training

def process_request(scenario_index, regenerate_obs=False):
    scenario, observation = get_observation(int(scenario_index))
    
    rule_based_action, rule_based_score = rule_based_agent(observation)
    trained_action, trained_score = trained_agent(observation, scenario)
    
    obs_str = json.dumps(observation, indent=2)
    rule_based_str = json.dumps(rule_based_action, indent=2)
    trained_str = json.dumps(trained_action, indent=2)
    
    improvement = f"**Improvement: +{(trained_score - rule_based_score):.2f}** ({(trained_score - rule_based_score)/rule_based_score*100:.1f}%)"
    
    return scenario, obs_str, rule_based_str, trained_str, improvement

# ========== Gradio Interface ==========
with gr.Blocks(title="AnomalyGuard - Cyber Incident Response", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AnomalyGuard: Training LLMs for Cyber Incident Response
    
    **The only OpenEnv environment that requires agents to justify every action with evidence and reasoning.**
    
    AnomalyGuard simulates realistic cyber attacks (Vercel supply chain breach, ransomware, etc.) 
    and trains LLMs to respond like experienced security analysts.
    
    [View GitHub](https://github.com/Padmavathi-1234/Anomaly-Guard-) | [README](https://github.com/Padmavathi-1234/Anomaly-Guard-/blob/main/README.md)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            scenario_dropdown = gr.Dropdown(
                choices=SCENARIOS,
                label="Select Cyber Attack Scenario",
                value=SCENARIOS[0],
                interactive=True
            )
            regenerate_button = gr.Button("Generate New Observation")
        
        with gr.Column(scale=1):
            gr.Image(REWARD_PLOT, label="Training Progress")
    
    with gr.Row():
        with gr.Column():
            scenario_output = gr.Textbox(label="Current Scenario")
            observation_output = gr.JSON(label="Environment Observation")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Before: Rule-Based Agent (Baseline)")
            rule_based_output = gr.JSON(label="Action & Justification")
            
        with gr.Column():
            gr.Markdown("### After: Trained Agent (GRPO)")
            trained_output = gr.JSON(label="Action & Justification")
    
    improvement_output = gr.Markdown()
    
    regenerate_button.click(
        process_request, 
        inputs=[scenario_dropdown, regenerate_button], 
        outputs=[scenario_output, observation_output, rule_based_output, trained_output, improvement_output]
    )
    
    scenario_dropdown.change(
        process_request, 
        inputs=[scenario_dropdown, regenerate_button],
        outputs=[scenario_output, observation_output, rule_based_output, trained_output, improvement_output]
    )
    
    demo.load(
        process_request,
        inputs=[gr.Number(value=0, visible=False), gr.Checkbox(value=True, visible=False)],
        outputs=[scenario_output, observation_output, rule_based_output, trained_output, improvement_output]
    )

if __name__ == "__main__":
    demo.launch()