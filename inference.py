"""
AnomalyGuard — Inference Script
Uses OpenAI client for all LLM calls.
Runtime < 20 minutes on 2 vCPU / 8 GB RAM.

Required environment variables:
  API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — Hugging Face / OpenAI API key
  ENV_URL       — Environment URL (default: http://localhost:7860)
"""
from __future__ import annotations

import json
import os
import sys
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", ""))
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
SEED         = 42
MAX_STEPS    = {1: 15, 2: 20, 3: 30}
TASK_NAMES   = {
    1: "Alert Triage",
    2: "Incident Containment",
    3: "Full Incident Response",
}

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


class EnvClient:
    """HTTP client for AnomalyGuard environment."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: int, seed: int = 42) -> dict:
        r = self.session.post(
            f"{self.base}/reset",
            params={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.session.post(
            f"{self.base}/step",
            json=action,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: int) -> dict:
        r = self.session.post(
            f"{self.base}/grader",
            params={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


SYSTEM_PROMPT = """You are an expert cybersecurity SOC analyst operating inside AnomalyGuard,
an RL environment for cybersecurity incident response.

YOUR MISSION: Analyze SIEM alerts and network data, then take actions to detect,
contain, and remediate security incidents.

CRITICAL: Every action MUST include a complete justification. No exceptions.

OUTPUT FORMAT: Respond with ONLY a valid JSON object. No markdown. No explanation outside JSON.

ACTION SCHEMA:
{
  "action_type": "<one of the available actions>",
  "target": "<alert_id | host_id | ip_address | cve_id | account | persistence_type>",
  "parameters": {},
  "justification": {
    "reasoning": "<MINIMUM 50 CHARS — cite specific alert IDs, host IDs, CVEs, IPs, MITRE techniques>",
    "evidence": [
      {
        "source": "<alert_id or host_id from current observation>",
        "content": "<specific data point you observed>",
        "relevance_score": 0.0-1.0
      }
    ],
    "risk_assessment": {
      "threat_level": "LOW|MEDIUM|HIGH|CRITICAL",
      "confidence": 0.0-1.0,
      "potential_impact": "<what happens if you do NOT act>",
      "business_disruption_estimate": "<disruption caused by this action>"
    },
    "alternatives_considered": [
      {
        "action": "<alternative you considered>",
        "rejected_because": "<specific reason, min 20 chars>"
      }
    ]
  }
}

AVAILABLE ACTION TYPES:
- triage_alert: parameters={"classification": "true_positive"|"false_positive"}
- isolate_host: no extra parameters
- block_ip: no extra parameters
- disable_account: no extra parameters
- patch_vulnerability: no extra parameters (target=CVE-ID)
- remove_persistence: no extra parameters (target=persistence_type)
- rotate_credentials: no extra parameters (target=account_name)
- restore_host: no extra parameters (host must be isolated first)
- collect_forensics: no extra parameters
- escalate_incident: target="tier2"

INCIDENT PHASES:
- detection: Triage alerts — identify true vs false positives
- containment: Isolate threats — stop lateral movement
- eradication: Remove malware persistence, patch CVEs
- recovery: Restore systems, rotate credentials

SCORING: Your score = action_correctness * 35% + explanation_quality * 65%
Your EXPLANATION QUALITY is more important than the action itself.
Cite specific IDs from the observation. Be specific and technical."""


def build_user_prompt(obs: dict) -> str:
    alerts = []
    for a in obs.get("alerts", [])[:8]:
        alerts.append({
            "id":       a["alert_id"],
            "severity": a["severity"],
            "type":     a["alert_type"],
            "host":     a["source_host"],
            "src_ip":   a["source_ip"],
            "desc":     a["description"][:200],
            "iocs":     a.get("ioc_matches", []),
            "mitre":    a["mitre_technique"]["technique_id"] if a.get("mitre_technique") else None,
            "triaged":  a.get("agent_classification"),
        })

    hosts = []
    for h in obs.get("hosts", [])[:10]:
        hosts.append({
            "id":          h["host_id"],
            "hostname":    h["hostname"],
            "ip":          h["ip_address"],
            "status":      h["status"],
            "c2_active":   h["c2_active"],
            "persistence": h["persistence"],
            "vulns":       h.get("vulnerabilities", [])[:3],
            "accounts":    h.get("accounts", [])[:3],
        })

    intel = obs.get("threat_intel", {})

    return f"""CURRENT ENVIRONMENT STATE:
Phase: {obs.get("incident_phase")} | Step: {obs.get("step")}/{obs.get("max_steps")} | Score: {obs.get("score_so_far", 0):.3f}
{obs.get("message", "")}

SIEM ALERTS:
{json.dumps(alerts, indent=2)}

NETWORK HOSTS:
{json.dumps(hosts, indent=2)}

THREAT INTELLIGENCE:
- Campaign: {intel.get("attack_campaign")}
- Malicious IPs: {intel.get("malicious_ips", [])[:5]}
- Known CVEs: {intel.get("known_cves", [])[:4]}
- Malicious Hashes: {intel.get("malicious_hashes", [])[:3]}
- C2 Domains: {intel.get("malicious_domains", [])[:3]}

Available actions: {obs.get("available_actions", [])}
Score breakdown: {obs.get("score_breakdown", {})}

Choose the MOST IMPORTANT action. Untriaged alerts have triaged=null.
Output ONLY the JSON action object."""


def get_llm_action(obs: dict, attempt: int = 0) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue

        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"  JSON parse error (attempt {attempt}): {e}")
        if attempt < 2:
            return get_llm_action(obs, attempt + 1)
        return _fallback_action(obs)

    except Exception as e:
        print(f"  LLM error: {e}")
        return _fallback_action(obs)


def _fallback_action(obs: dict) -> dict:
    """Deterministic fallback when LLM fails."""
    for a in obs.get("alerts", []):
        if a.get("agent_classification") is None:
            has_ioc  = bool(a.get("ioc_matches"))
            high_sev = a["severity"] in ("critical", "high")
            is_tp    = has_ioc or high_sev
            return {
                "action_type": "triage_alert",
                "target":      a["alert_id"],
                "parameters":  {"classification": "true_positive" if is_tp else "false_positive"},
                "justification": {
                    "reasoning": (
                        f"Fallback triage for alert {a['alert_id']}: severity={a['severity']}, "
                        f"ioc_matches={a.get('ioc_matches', [])}, type={a['alert_type']}. "
                        f"Classification based on severity and IOC presence per IR playbook."
                    ),
                    "evidence": [{
                        "source":          a["alert_id"],
                        "content":         f"severity={a['severity']}, iocs={a.get('ioc_matches', [])}",
                        "relevance_score": 0.7,
                    }],
                    "risk_assessment": {
                        "threat_level":                "HIGH" if high_sev else "MEDIUM",
                        "confidence":                   0.65,
                        "potential_impact":             "Security incident if true positive is missed",
                        "business_disruption_estimate": "Triage is non-disruptive — monitoring only",
                    },
                    "alternatives_considered": [{
                        "action":           "collect_forensics",
                        "rejected_because": "Triage must precede forensics collection in IR workflow",
                    }],
                },
            }

    return {
        "action_type": "escalate_incident",
        "target":      "tier2",
        "parameters":  {},
        "justification": {
            "reasoning": (
                "No untriaged alerts remain and no clear next action identified. "
                "Escalating to Tier-2 SOC analysts for manual review of any remaining "
                "threats that automated analysis has not addressed."
            ),
            "evidence": [{
                "source":          "system",
                "content":         "No actionable untriaged alerts in current observation",
                "relevance_score": 0.5,
            }],
            "risk_assessment": {
                "threat_level":                "MEDIUM",
                "confidence":                   0.5,
                "potential_impact":             "Delayed response if undetected threats remain",
                "business_disruption_estimate": "Escalation is non-disruptive",
            },
            "alternatives_considered": [{
                "action":           "collect_forensics",
                "rejected_because": "No specific unanalyzed hosts identified for forensic collection",
            }],
        },
    }


def run_episode(env_client: EnvClient, task_id: int, seed: int = 42) -> dict:
    print(f"\n{'='*60}")
    print(f"  Task {task_id}: {TASK_NAMES[task_id]}")
    print(f"  Seed: {seed} | Max Steps: {MAX_STEPS[task_id]}")
    print(f"{'='*60}")

    obs    = env_client.reset(task_id=task_id, seed=seed)
    done   = False
    step   = 0
    rewards = []
    t_start = time.time()

    print(f"  Phase: {obs.get('incident_phase')} | "
          f"Alerts: {len(obs.get('alerts', []))} | "
          f"Hosts: {len(obs.get('hosts', []))}")

    while not done and step < MAX_STEPS[task_id]:
        # Per-task 6-minute limit
        if time.time() - t_start > 360:
            print("  Time limit for this task reached")
            break

        step += 1
        print(f"\n  Step {step}/{MAX_STEPS[task_id]} | "
              f"Score: {obs.get('score_so_far', 0):.3f} | "
              f"Phase: {obs.get('incident_phase')}")

        action = get_llm_action(obs)
        print(f"  -> {action.get('action_type')} | {action.get('target')}")

        try:
            result  = env_client.step(action)
            obs     = result["observation"]
            reward  = result["reward"]
            done    = result["done"]
            rewards.append(reward["value"])

            print(f"  <- reward={reward['value']:+.3f} | "
                  f"correct={reward['action_correctness']:.2f} | "
                  f"explain={reward['explanation_quality']:.2f}")

            msg = result.get("info", {}).get("action_result", {}).get("message", "")
            if msg:
                print(f"     {msg}")

        except Exception as e:
            print(f"  Step error: {e}")
            break

    grade   = env_client.grade(task_id)
    elapsed = time.time() - t_start

    print(f"\n  {'-'*50}")
    print(f"  FINAL SCORE:        {grade['final_score']:.4f}")
    print(f"  Action Correctness: {grade['action_correctness']:.4f}")
    print(f"  Explanation Quality:{grade['explanation_quality']:.4f}")
    print(f"  Threats Detected:   {grade['threats_detected']}")
    print(f"  Threats Missed:     {grade['threats_missed']}")
    print(f"  Containment Rate:   {grade['containment_rate']:.2f}")
    print(f"  Steps:              {grade['steps_taken']} | Time: {elapsed:.1f}s")

    return {
        "task_id":      task_id,
        "task_name":    TASK_NAMES[task_id],
        "seed":         seed,
        "final_score":  grade["final_score"],
        "steps":        step,
        "total_reward": round(sum(rewards), 4),
        "elapsed_s":    round(elapsed, 1),
        "grade":        grade,
    }


def main():
    print("\n" + "="*60)
    print("  AnomalyGuard — LLM Agent Inference")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print("="*60)

    env = EnvClient(ENV_URL)

    print("\n  Waiting for environment...")
    for i in range(30):
        if env.health():
            print("  Environment ready")
            break
        print(f"  Attempt {i+1}/30...")
        time.sleep(2)
    else:
        print("  ERROR: Environment not reachable")
        print("  Start with: uvicorn app.main:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    results  = []
    t_global = time.time()

    for task_id in [1, 2, 3]:
        if time.time() - t_global > 1080:  # 18 min global limit
            print("Global time limit reached")
            break
        result = run_episode(env, task_id=task_id, seed=SEED)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<32} {'Score':>8} {'Steps':>6} {'Time':>8}")
    print(f"  {'-'*56}")
    for r in results:
        print(f"  {r['task_name']:<32} {r['final_score']:>8.4f} "
              f"{r['steps']:>6} {r['elapsed_s']:>7.1f}s")

    avg = sum(r["final_score"] for r in results) / max(len(results), 1)
    print(f"  {'-'*56}")
    print(f"  {'AVERAGE':<32} {avg:>8.4f}")

    elapsed = time.time() - t_global
    print(f"\n  Total: {elapsed:.1f}s | {'OK' if elapsed < 1200 else 'OVER LIMIT'}")
    print("="*60)

    with open("results.json", "w") as f:
        json.dump({
            "model":         MODEL_NAME,
            "seed":          SEED,
            "results":       results,
            "average_score": round(avg, 4),
            "total_elapsed": round(elapsed, 1),
        }, f, indent=2)
    print("  Saved: results.json")


if __name__ == "__main__":
    main()
