---
title: AnomalyGuard
emoji: 🛡️
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
---

# AnomalyGuard — EU AI Act Compliant RL Environment for Cybersecurity
### The only OpenEnv environment with mandatory action justification and built-in regulatory compliance

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon](https://img.shields.io/badge/Meta%20OpenEnv-Hackathon-orange)](https://scaler.com)
[![Validate](https://img.shields.io/badge/openenv%20validate-PASSED-brightgreen)](https://padmavathi-123-anomalyguard.hf.space)

## ✅ Validation Status

```
openenv validate  → [OK] Ready for multi-mode deployment
Reproducibility   → Verified (same seed = identical scenario)
Partial Obs       → Verified (query_host reveals hidden state)
Termination       → Verified (terminated vs truncated correct)
Grader            → Deterministic (no random, no time-based logic)
Deployment        → Live on Hugging Face Spaces
```

---

## 🚀 Live Demo

**Try it now:** [https://padmavathi-123-anomalyguard.hf.space](https://padmavathi-123-anomalyguard.hf.space)

**Interactive API Docs:** [https://padmavathi-123-anomalyguard.hf.space/docs](https://padmavathi-123-anomalyguard.hf.space/docs)

**GitHub:** [https://github.com/Padmavathi-1234/Anomaly-Guard-](https://github.com/Padmavathi-1234/Anomaly-Guard-)

---

## ⚡ Test It Right Now

**Step 1 — Start an episode:**

```bash
curl -X POST "https://padmavathi-123-anomalyguard.hf.space/reset?task_id=1&seed=42"
```

**Step 2 — Take an action with full justification:**

```bash
curl -X POST "https://padmavathi-123-anomalyguard.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "triage_alert",
    "target": "ALT-10001",
    "parameters": {"classification": "true_positive"},
    "justification": {
      "reasoning": "Alert ALT-10001 shows C2 beacon pattern with confidence 0.89 matching known malicious IP in threat intel. MITRE T1071 technique confirms command and control communication.",
      "evidence": [{"source": "ALT-10001", "content": "C2 beacon to 185.220.101.45 confidence 0.89", "relevance_score": 0.95}],
      "risk_assessment": {"threat_level": "CRITICAL", "confidence": 0.89, "potential_impact": "Active C2 channel allows attacker persistence", "business_disruption_estimate": "High — active breach ongoing"},
      "alternatives_considered": [{"action": "monitor", "rejected_because": "Confidence 0.89 is too high to ignore without classification"}]
    }
  }'
```

**Step 3 — Check EU AI Act compliance:**

```bash
curl "https://padmavathi-123-anomalyguard.hf.space/compliance/audit"
```

**Step 4 — View detailed metrics:**

```bash
curl "https://padmavathi-123-anomalyguard.hf.space/metrics/detailed"
```

**Interactive API (easiest):** [https://padmavathi-123-anomalyguard.hf.space/docs](https://padmavathi-123-anomalyguard.hf.space/docs)

---

## Overview

AnomalyGuard is the **first OpenEnv reinforcement learning environment
requiring agents to JUSTIFY every action** with evidence-based reasoning,
risk assessment, and alternative analysis.

> ### 🇪🇺 EU AI Act Compliance — Built Into the Environment
> AnomalyGuard is the **only OpenEnv environment** designed around
> EU AI Act Articles 10, 13, and 14. Every episode automatically
> generates a 5-check compliance audit. Every action requires
> structured justification. This is not optional — it is enforced
> at the reward level. Non-compliant actions receive lower scores.

**Core Reward Formula:**
Reward = action_correctness × 0.60 + explanation_quality × 0.40

---

## 🏆 Key Differentiators

| Feature                       | AnomalyGuard       | Typical RL Env     |
| ----------------------------- | ------------------ | ------------------ |
| Action justification required | ✅ Mandatory       | ❌ None            |
| EU AI Act compliance engine   | ✅ Built-in        | ❌ None            |
| Partial observability         | ✅ Query-based     | ❌ Full visibility |
| MITRE ATT&CK integration      | ✅ Real techniques | ❌ Abstract        |
| Malware spread simulation     | ✅ Topology-based  | ❌ Static          |
| Adaptive curriculum           | ✅ 10 levels       | ❌ Fixed           |
| Dense progress rewards        | ✅ Milestone-based | ❌ Sparse          |

---

## 🚀 Key Features

### Core Capabilities

- 🔍 **Evidence-Based Reasoning**: Every action requires citing specific alert IDs, host IDs, CVEs, and MITRE techniques
- 🎯 **MITRE ATT&CK Integration**: 8 real-world attack patterns with genuine IOCs, file hashes, and C2 IPs
- 📊 **Multi-Dimensional Grading**: Scoring across action correctness, reasoning clarity, evidence validity, risk accuracy
- 🔄 **Deterministic Reproducibility**: Same seed + task always produces identical scenario
- 🤖 **OpenEnv Compatible**: Fully extends `openenv.env.env.Env` base class

### Advanced Features

- 🦠 **Malware Spread Simulation**: Topology-based propagation across enterprise network graph
- 📈 **Adaptive Curriculum Learning**: 10-level difficulty auto-adjusts based on agent performance
- 🔗 **Strict Task Dependencies**: Realistic IR workflow — detect → contain → eradicate → recover
- 🕵️ **Partial Observability**: Host details hidden until agent uses `query_host` action
- 🏆 **Dense Progress Rewards**: Milestone bonuses with correct `terminated` vs `truncated` separation
- ⚖️ **EU AI Act Compliance Engine**: 5-check audit against Articles 10, 13, 14

---

## 🏗️ Architecture

```text
anomalyguard/
├── app/
│   ├── main.py              # FastAPI — OpenEnv HTTP interface
│   ├── models.py            # Pydantic v2 models
│   ├── environment.py       # Core RL environment (extends openenv.env.env.Env)
│   ├── grader.py            # Deterministic grader (0.0–1.0)
│   ├── scenarios.py         # MITRE ATT&CK reproducible scenario generator
│   ├── explainability.py    # Explanation quality scorer
│   ├── real_data.py         # Attack patterns, CVEs, IOCs database
│   └── baseline.py          # RandomAgent + RuleBasedAgent
├── tests/
│   └── test_environment.py  # pytest validation suite
├── inference.py             # LLM agent example
├── Dockerfile
├── requirements.txt
├── openenv.yaml
└── README.md
```

---

## 📋 Observation Space

### What the Agent Sees

```python
{
    "task_id":           int,          # 1, 2, or 3
    "step":              int,          # Current step (0 to max_steps)
    "max_steps":         int,          # 15 / 20 / 30 depending on task
    "alerts":            List[Alert],  # SIEM alerts (is_true_positive HIDDEN)
    "hosts":             List[Host],   # Network hosts (details MASKED until queried)
    "incident_phase":    str,          # detection/containment/eradication/recovery
    "time_remaining":    int,          # Steps left in episode
    "available_actions": List[str],    # Valid action types for current task
    "score_so_far":      float,        # Running score estimate
    "threat_intel":      ThreatIntel,  # IOCs, malicious IPs, CVEs
}
```

### Partial Observability — Host Masking

Hosts show limited information until investigated with `query_host`:

| Field                         | Before `query_host`      | After `query_host` |
| ----------------------------- | ------------------------ | ------------------ |
| host_id, hostname, ip_address | ✅ Visible               | ✅ Visible         |
| role, criticality, services   | ✅ Visible               | ✅ Visible         |
| c2_active                     | ❌ Hidden (shows False)  | ✅ Revealed        |
| persistence                   | ❌ Hidden (shows [])     | ✅ Revealed        |
| vulnerabilities               | ❌ Hidden (shows [])     | ✅ Revealed        |
| accounts                      | ❌ Hidden (shows [])     | ✅ Revealed        |
| status                        | ❌ Hidden (shows online) | ✅ Revealed        |

**Why this matters:** Agents must investigate before acting — mirrors real SOC analyst workflow.

---

## 🎯 Task Hierarchy (Curriculum)

| ID  | Name                   | Difficulty | Max Steps | Objective                                                     |
| --- | ---------------------- | ---------- | --------- | ------------------------------------------------------------- |
| 1   | Alert Triage           | Easy       | 15        | Classify 6-10 SIEM alerts as TP/FP with justification         |
| 2   | Incident Containment   | Medium     | 20        | Contain active breach across 10-15 hosts                      |
| 3   | Full Incident Response | Hard       | 30        | Complete IR lifecycle: detect → contain → eradicate → recover |

### Curriculum Learning (10 Levels)

| Level | Tier         | Difficulty | Max Steps |
| ----- | ------------ | ---------- | --------- |
| 1-3   | Beginner     | 0.3 - 0.5  | 15        |
| 4-6   | Intermediate | 0.5 - 0.7  | 20        |
| 7-10  | Expert       | 0.7 - 1.0  | 30        |

Auto-advances when avg score > 0.75, regresses when < 0.35.

---

## 🧠 Scoring & Explainability

Every action payload must include `ActionJustification`:

```json
{
  "action_type": "isolate_host",
  "target": "HOST-003",
  "parameters": {},
  "justification": {
    "reasoning": "Host HOST-003 shows active C2 communication to known malicious IP 185.220.101.45 matching threat intel. Isolation prevents lateral movement to db-server-01.",
    "evidence": [
      {
        "source": "ALT-10001",
        "content": "C2 beacon detected to 185.220.101.45 every 300s",
        "relevance_score": 0.95
      }
    ],
    "risk_assessment": {
      "threat_level": "CRITICAL",
      "confidence": 0.92,
      "potential_impact": "Lateral movement to database tier",
      "business_disruption_estimate": "High — web tier isolated"
    },
    "alternatives_considered": [
      {
        "action": "block_ip",
        "rejected_because": "Does not stop existing C2 session already established"
      }
    ]
  }
}
```

### Grading Formulas

- **Task 1:** `final = triage_accuracy × 0.70 + avg_explanation × 0.30`
- **Task 2:** `final = (triage × 0.5 + containment × 0.5) × 0.80 + avg_explanation × 0.20`
- **Task 3:** `final = (triage × 0.20 + containment × 0.30 + eradication × 0.25 + recovery × 0.25) × 0.75 + avg_explanation × 0.25`

---

## ⚖️ EU AI Act Compliance

AnomalyGuard is the **only OpenEnv environment built around EU AI Act compliance**.

Every episode generates a 5-check compliance audit:

| Check                        | Article         | What It Validates                     |
| ---------------------------- | --------------- | ------------------------------------- |
| All Actions Justified        | Article 14.4(b) | Every action has reasoning ≥50 chars  |
| Explanation Quality          | Article 13.1    | Average explanation score ≥0.60       |
| Human Oversight Available    | Article 14.1    | `escalate_incident` always accessible |
| High-Risk Actions Documented | Article 14.4(c) | isolate/disable/restore all justified |
| No Classification Bias       | Article 10.2(f) | TP/FP ratio within acceptable range   |

```bash
curl "https://padmavathi-123-anomalyguard.hf.space/compliance/audit"
```

**Why this matters:** The EU AI Act requires high-risk AI systems to maintain human oversight, provide transparent reasoning, and document all decisions. AnomalyGuard enforces these requirements at the environment level — making it suitable for EU-regulated deployments.

---

## 📊 Baseline Performance (Measured)

| Agent Type | Task 1 | Task 2 | Task 3 | Average |
|-----------|--------|--------|--------|---------|
| Random Agent | 0.05-0.15 | 0.03-0.10 | 0.01-0.05 | ~0.08 |
| **Rule-Based (Measured)** | **0.9023** | **0.9366** | **0.5232** | **0.7874** |
| RL Agent Target | 0.75+ | 0.65+ | 0.55+ | 0.65+ |

> ✅ **Rule-based agent exceeds RL target (0.79 > 0.75)** — demonstrates environment provides strong, learnable reward signal.

**Note on Task 3:** The rule-based agent times out because it lacks the logic for eradication (remove_persistence) and recovery (restore_host) phases. A trained RL agent or LLM agent would learn these patterns and complete the full IR lifecycle.

---

## 🔄 Reproducibility

```bash
# Same seed always produces identical scenario
curl -X POST "https://padmavathi-123-anomalyguard.hf.space/reset?task_id=1&seed=42"
curl -X POST "https://padmavathi-123-anomalyguard.hf.space/reset?task_id=1&seed=42"
# Both calls produce IDENTICAL alerts, hosts, threat intel
```

Guaranteed by `random.Random(seed)` throughout `scenarios.py` with no global state mutation.

---

## 🎬 Run the Demo

See the environment in action with one command:

```bash
python demo.py
```

**Sample output:**

```
════════════════════════════════════════════════════════════
  AnomalyGuard — Live Environment Demo
════════════════════════════════════════════════════════════
  Health: OK ✓

Task 1 - Alert Triage:
  Final Score:      0.9023
  Action Quality:   1.0000
  Explanation:      0.6744
  Precision:        0.571
  Recall:           1.000
  F1 Score:         0.727
  EU AI Act:        COMPLIANT ✓ (5/5 checks) Risk: LOW
  Steps:            12

Task 2 - Incident Containment:
  Final Score:      0.9366
  Action Quality:   1.0000
  Explanation:      0.6830
  Containment:      1.00
  EU AI Act:        COMPLIANT ✓ (5/5 checks) Risk: LOW
  Steps:            17

Task 3 - Full Incident Response:
  Final Score:      0.5232
  Action Quality:   0.5000
  Containment:      1.00
  EU AI Act:        COMPLIANT ✓ (4/5 checks) Risk: MEDIUM
  Steps:            30 (timeout)

AVERAGE:            0.7874
```

---

## 🌐 API Endpoints

### Core

| Endpoint                   | Method | Description                       |
| -------------------------- | ------ | --------------------------------- |
| `/health`                  | GET    | Health check                      |
| `/tasks`                   | GET    | List all tasks                    |
| `/reset?task_id=1&seed=42` | POST   | Start new episode                 |
| `/step`                    | POST   | Execute action with justification |
| `/state`                   | GET    | Current masked observation        |
| `/state/raw`               | GET    | Full unmasked state for grading   |
| `/grader?task_id=1`        | POST   | Grade completed episode           |

### Observability

| Endpoint                       | Method | Description                         |
| ------------------------------ | ------ | ----------------------------------- |
| `/host/{host_id}/visibility`   | GET    | Check what agent can see for a host |
| `/observability/status`        | GET    | Query coverage across all hosts     |
| `/host/{host_id}/dependencies` | GET    | Prerequisites before restoration    |

### Metrics

| Endpoint            | Method | Description                             |
| ------------------- | ------ | --------------------------------------- |
| `/metrics/detailed` | GET    | Precision, recall, F1, containment rate |
| `/metrics/rewards`  | GET    | Step-by-step reward history             |
| `/metrics/spread`   | GET    | Malware spread statistics               |

### Baseline

| Endpoint              | Method | Description                 |
| --------------------- | ------ | --------------------------- |
| `/baseline?task_id=1` | POST   | Run rule-based agent        |
| `/baseline/reference` | GET    | Expected performance ranges |

### Curriculum

| Endpoint              | Method | Description                   |
| --------------------- | ------ | ----------------------------- |
| `/curriculum/status`  | GET    | Current level and performance |
| `/curriculum/toggle`  | POST   | Enable/disable curriculum     |
| `/episodes/diversity` | GET    | Scenario diversity statistics |

### EU AI Act Compliance

| Endpoint            | Method | Description                    |
| ------------------- | ------ | ------------------------------ |
| `/compliance/audit` | GET    | 5-check EU AI Act audit report |
| `/compliance/trail` | GET    | Full action audit trail        |

---

## ⚙️ Quick Start

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t anomalyguard .
docker run -p 7860:7860 anomalyguard
```

### Run LLM Agent

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"
python inference.py
```

---

## ⚠️ Known Limitations

- **Single-agent only** — no multi-agent coordination
- **Simulated network** — not real packet captures
- **Static adversary** — malware spreads probabilistically but does not adapt to agent
- **Max 25 hosts** — real enterprises have thousands
- **Discrete actions** — no continuous parameter tuning

---

## 📚 Related Work

- **OpenAI Gym** — Standard RL interface (we extend with explainability)
- **CyberBattleSim** (Microsoft) — Network security simulation (we add MITRE ATT&CK)
- **BRAWL** (MITRE) — Autonomous cyber ops (we add curriculum learning)
- **EU AI Act** — Regulatory framework driving our justification requirement

---

## 🏆 About This Project

Built for the **Meta OpenEnv Hackathon** hosted by **Scaler**, **OpenEnv**, **Meta AI**, and **PyTorch**.

This environment was designed to push the boundaries of what an OpenEnv environment can do — combining real cybersecurity scenarios, EU AI Act compliance, and advanced RL training features into a single deployable package.
