# AnomalyGuard — Explainable AI for Cybersecurity Incident Response

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AnomalyGuard is the **first OpenEnv reinforcement learning environment requiring agents to JUSTIFY every action** with evidence-based reasoning, risk assessment, and alternative analysis. Designed explicitly to address the stringent requirements of the **EU AI Act** for explainable AI in critical infrastructure, AnomalyGuard ensures that AI decision-making in cybersecurity is transparent, traceable, and robust.

**Core Innovation:** `Reward = action_correctness × 0.60 + explanation_quality × 0.40`

By prioritizing action correctness while maintaining meaningful explanation incentives, AnomalyGuard ensures agents can succeed through proper incident response while still developing EU AI Act-compliant transparency.

---

## 🚀 Key Features

### Core Capabilities
- 🔍 **Evidence-Based Reasoning**: Every action requires citing specific alert IDs, host IDs, CVEs, and MITRE techniques.
- 🎯 **MITRE ATT&CK Integration**: Incorporates 8 real-world attack patterns with genuine Indicators of Compromise (IOCs), file hashes, and Command & Control (C2) IPs.
- 📊 **Multi-Dimensional Grading**: Comprehensive scoring across action correctness, reasoning clarity, evidence validity, and risk accuracy.
- 🔄 **Deterministic Reproducibility**: Seed-based scenario generation ensures identical scenarios for reliable benchmarking and debugging.
- 🤖 **OpenEnv Compatible**: Fully extends the `openenv.env.env.Env` base class for seamless integration with the OpenEnv ecosystem.

### Advanced Implemented Features
- 🦠 **Malware Spread Simulation (Topology-Based)**: Realistically models how malware propagates through an enterprise network based on defined network topology and host vulnerabilities.
- 📈 **Adaptive Curriculum Learning**: An intelligent difficulty scaling system that automatically adjusts scenario complexity based on the agent's historical performance.
- 🔗 **Strict Task Dependency Chain**: Implements a rigid, realistic incident response workflow (e.g., you cannot eradicate a threat before containing it, or contain it before detection).
- 🖥️ **Interactive Web-Based Dashboard**: A real-time monitoring UI for visualizing agent actions, network state, and explanation quality.
- ⚖️ **Realistic Baseline Agent**: A rule-based baseline agent implementation to provide grounded performance metrics and prevent score inflation.
- 🇪🇺 **EU AI Act Compliance Engine**: Built-in validation to ensure all automated actions meet strict explicability, transparency, and human-in-the-loop auditability standards.

---

## 🏗️ Architecture

```text
anomalyguard/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI wrapping the openenv environment
│   ├── models.py            # Pydantic v2 models
│   ├── environment.py       # Core class extending openenv.env.env.Env
│   ├── grader.py            # Deterministic grader (0.0–1.0)
│   ├── scenarios.py         # MITRE ATT&CK reproducible scenario generator
│   ├── explainability.py    # Explanation quality scorer
│   ├── real_data.py         # Attack patterns, CVEs, IOCs database
│   └── baseline.py          # Rule-based baseline agent
├── inference.py             # OpenAI client LLM agent
├── Dockerfile
├── requirements.txt
├── openenv.yaml
└── README.md
```

## ⚡ Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the FastAPI server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

1. **Build the image:**
```bash
docker build -t anomalyguard .
```

2. **Run the container:**
```bash
docker run -p 7860:7860 anomalyguard
```

---

## 🌐 API Endpoints

The environment exposes a robust FastAPI backend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check / API status |
| `/tasks` | GET | List available curriculum tasks |
| `/reset?task_id=1&seed=42` | POST | Reset environment for a new deterministic episode |
| `/step` | POST | Execute an IR action with required justification |
| `/state` | GET | Retrieve the current network/alert observation |
| `/grader?task_id=1` | POST | Grade the completed episode |
| `/baseline?task_id=1&seed=42`| POST | Run the rule-based baseline agent for comparison |

---

## 🎯 Task Hierarchy (Curriculum)

| ID | Name | Difficulty | Max Steps | Description |
|----|------|-----------|-----------|-------------|
| 1 | Alert Triage | Easy | 15 | Triage 6-10 SIEM alerts with evidence-based justifications |
| 2 | Incident Containment | Medium | 20 | Contain active topological breach across 10-15 hosts |
| 3 | Full Incident Response | Hard | 30 | Full IR dependency lifecycle: detect → contain → eradicate → recover |

---

## 🧠 Scoring & Explainability

To satisfy the **EU AI Act**, every action payload must include an `ActionJustification` structured as follows:
- **Reasoning** (min 50 chars): Explicitly citing specific evidence.
- **Evidence** (min 1 item): Referencing actual observation data (IPs, hashes, etc.).
- **Risk Assessment**: Threat level, confidence score, and potential impact.
- **Alternatives Considered**: Documenting what other actions were evaluated and why they were rejected.

### Grading Formulas

- **Task 1:** `final = triage_accuracy × 0.70 + avg_explanation × 0.30`
- **Task 2:** `final = (triage × 0.5 + containment × 0.5) × 0.80 + avg_explanation × 0.20`
- **Task 3:** `final = (triage × 0.20 + containment × 0.30 + eradication × 0.25 + recovery × 0.25) × 0.75 + avg_explanation × 0.25`

**Design Rationale:** Action correctness (60-80%) ensures agents master incident response fundamentals, while explanation quality (20-40%) provides significant bonus for transparency and auditability without creating a failure-prone bottleneck.

---

## 🤖 Running Inference

You can test the environment using the provided OpenAI-powered LLM agent:

```bash
# Export required environment variables (see .env.example)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"

# Run the inference script
python inference.py
```

---

## ⚙️ Environment Variables

Refer to `.env.example` for secure deployment. 

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / OpenAI API key | — |
| `ENV_URL` | OpenEnv Backend URL | `http://localhost:7860` |

---

## 🏆 Hackathon Details

Built for the **Meta OpenEnv Hackathon** hosted by **Scaler**, **OpenEnv**, **Meta AI**, and **PyTorch**.
