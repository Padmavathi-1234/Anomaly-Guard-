title: AnomalyGuard
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:

- openenv
- reinforcement-learning
- cybersecurity
- incident-response
- explainable-ai
- curriculum-learning
- eu-ai-act
- multi-agent
- grpo

_AnomalyGuard_
An RL Environment That Trains AI to Think Like a Cybersecurity Analyst

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon](https://img.shields.io/badge/OpenEnv-Hackathon%202026-orange)](https://scaler.com)
[![Space](https://img.shields.io/badge/HuggingFace-Live-brightgreen)](https://padmavathi-123-anomalyguard.hf.space)

Why This Exists

AI is everywhere now. In hospitals. In banks. In the systems that run power grids and financial markets. And wherever AI goes, attackers follow.
Security teams today receive thousands of alerts per day. They cannot investigate all of them fast enough. One missed alert, one wrong classification, one moment of hesitation can mean the difference between containing a breach in an hour and losing customer data for months.
When I saw how fast cyber attacks were growing and when the Vercel breach happened, I realized the problem was not that security teams did not know what to do. The problem was that there were too many alerts, too little time, and too much noise to filter
through manually.
I wanted to build an AI that does not just detect.That investigates. That explains its work. That a human analyst can actually supervise and trust.
That became AnomalyGuard.

What It Does

AnomalyGuard is an OpenEnv reinforcement learning environment where an LLM learns to act as a
Security Operations Center analyst. The agent receives real SIEM alerts and must work
through a complete incident response lifecycle — investigating hosts, classifying alerts, isolating compromised systems, removing persistence mechanisms, and restoring clean hosts to production.
Every action must be justified with specific evidence. The agent cannot simply isolate a host. It must explain which alert triggered the decision, what it found when it queried the host, why it chose this action over alternatives, and what the risk assessment was.
This is enforced at the reward level. Unjustified actions receive lower scores. This is how EU AI Act compliance is built into the training signal itself.

Links

| Resource         | URL                                               |
| ---------------- | ------------------------------------------------- |
| Live Environment | https://padmavathi-123-anomalyguard.hf.space      |
| API Docs         | https://padmavathi-123-anomalyguard.hf.space/docs |
| GitHub           | https://github.com/Padmavathi-1234/Anomaly-Guard- |

Try It Right Now

Start an investigation:

```bash
curl -X POST \
  "https://padmavathi-123-anomalyguard.hf.space/reset?task_id=1&seed=42"
Take an action:

Bash

curl -X POST \
  "https://padmavathi-123-anomalyguard.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "triage_alert",
    "target": "ALT-10001",
    "parameters": {"classification": "true_positive"},
    "justification": {
      "reasoning": "Alert ALT-10001 shows C2 beacon to 185.220.101.45 confidence 0.89. MITRE T1071 confirms active command and control requiring immediate classification.",
      "evidence": [{"source": "ALT-10001", "content": "C2 beacon detected", "relevance_score": 0.95}],
      "risk_assessment": {"threat_level": "CRITICAL", "confidence": 0.89, "potential_impact": "Active C2 allows attacker persistence", "business_disruption_estimate": "High"},
      "alternatives_considered": [{"action": "monitor", "rejected_because": "Confidence 0.89 too high to ignore"}]
    }
  }'
Check EU AI Act compliance:

Bash

curl "https://padmavathi-123-anomalyguard.hf.space/compliance/audit"
Training Results
The model was trained using GRPO (Group Relative Policy Optimization) with dynamic step selection based on curriculum complexity. What Makes This Different Feature	AnomalyGuard	Typical RL Env Action justification required	Mandatory	None EU AI Act compliance engine	Built-in	None Partial observability	Query-based	Full visibility MITRE ATT&CK integration	Real techniques	Abstract Malware spread simulation	Topology-based	Static Anti-hacking protection	Multi-layer	None Adaptive curriculum	10 levels	Fixed Multi-agent architecture	3 roles	Single agent The Design Partial Observability Host details are hidden until the agent calls query_host. An agent that isolates a host without investigating first gets penalized. This forces strategic investigation over blind action-taking, mirroring how real SOC analysts work.
Three Coordinated Agents
Agent	Responsibility :
Triage Agent	Classifies alerts as true or false positives
Containment Agent	Isolates hosts and blocks malicious IPs
Forensics Agent	Removes persistence and restores systems
Agents cannot act out of order. Triage before containment. Containment before eradication.
Eradication before recovery.

Adaptive Curriculum
The environment watches agent performance and adjusts difficulty automatically. When success
exceeds 75 percent it advances. Below 35 percent it regresses. Training steps scale with complexity —
50 steps for beginners, 300 for expert scenarios.

Real Attack Scenarios
Every scenario uses real MITRE ATT&CK techniques.
The IP 185.220.101.45 in training data is a real
known malicious IP linked to ransomware campaigns.

EU AI Act Compliance
Every episode generates a 5-check compliance audit.

Check	Article	Validates
All actions justified	14.4(b)	reasoning >= 50 chars
Explanation quality	13.1	avg score >= 0.60
Human oversight	14.1	escalate always available
High-risk documented	14.4(c)	isolate/disable justified
No bias	10.2(f)	TP/FP ratio balanced
No other OpenEnv environment enforces regulatory
compliance at the reward level.

Validation
text

openenv validate  -> [OK] Ready for multi-mode deployment
Reproducibility   -> Verified (same seed = identical scenario)
Partial Obs       -> Verified (query_host reveals hidden state)
Termination       -> Verified (terminated vs truncated correct)
Grader            -> Deterministic (no random, no time-based logic)
Deployment        -> Live on Hugging Face Spaces
API Reference
Core
Endpoint	Method	Description
/health	GET	Health check
/reset	POST	Start new episode
/step	POST	Execute action
/state	GET	Current observation
/grader	POST	Grade episode
Training
Endpoint	Method	Description
/train/start	POST	Start GRPO training
/train/status	GET	Check progress
/train/logs	GET	Full training logs
/train/plot	GET	Download reward curve
Compliance
Endpoint	Method	Description
/compliance/audit	GET	EU AI Act audit
/compliance/trail	GET	Action audit trail
/compliance/dashboard	GET	Compliance metrics
Multi-Agent
Endpoint	Method	Description
/reset-multiagent	POST	Start multi-agent episode
/step-multiagent	POST	Execute multi-agent step
/curriculum/status	GET	Current level
/anti-hacking/report	GET	Cheating detection
/threat-intel/live	GET	Live IOCs
Local Setup
Bash

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
Run training:

Bash

python training/train_grpo.py
Run demo:

Bash

python demo.py
Themes Covered
World Modeling (Professional) — The agent operates
in a partially observable enterprise network and
must build an internal model of the incident through
systematic investigation.

Multi-Agent Interactions — Three specialized agents
coordinate on incident response with explicit
dependencies and shared threat intelligence.

Self-Improvement — The adaptive curriculum scales
difficulty and training duration based on agent
performance, driving recursive capability growth.

Known Limitations
Simulated network, not real packet captures
Malware spread is probabilistic, not adaptive
Maximum 25 hosts per scenario
Discrete action space only
About
Meta PyTorch OpenEnv HuggingFace X Scaler Hackathon 2026
Solo Participant

Author: VSSK Sri Padmavathi

Themes: World Modeling, Multi-Agent Interactions,
Self-Improvement
```
