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
    1: "alert_triage",
    2: "incident_containment",
    3: "full_incident_response",
}

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Environment HTTP Client ───────────────────────────────────────

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

    def reset(self, task_id: int, seed: int = 42) -> tuple[dict, dict]:
        """
        Returns (observation_dict, info_dict).
        /reset returns {"observation": {...}, "info": {...}}
        """
        r = self.session.post(
            f"{self.base}/reset",
            params={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        # Handle both old format (direct obs) and new format (wrapped)
        if "observation" in data and "info" in data:
            return data["observation"], data["info"]
        else:
            # Fallback: treat entire response as observation
            return data, {}

    def step(self, action: dict) -> dict:
        """
        Returns full step response dict.
        Contains: observation, reward (float), done, truncated, info
        """
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

    def get_observability(self) -> dict:
        """Check how many hosts have been queried."""
        try:
            r = self.session.get(f"{self.base}/observability/status", timeout=10)
            return r.json()
        except Exception:
            return {}

    def get_metrics(self) -> dict:
        """Get detailed episode metrics."""
        try:
            r = self.session.get(f"{self.base}/metrics/detailed", timeout=10)
            return r.json()
        except Exception:
            return {}


# ── System Prompt ─────────────────────────────────────────────────

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
- query_host: INVESTIGATE a host to reveal its compromise status (target=host_id)
  USE THIS FIRST before isolating — you cannot see c2_active/persistence until queried
- triage_alert: parameters={"classification": "true_positive"|"false_positive"}
- isolate_host: Isolate a compromised host (target=host_id, must query first)
- block_ip: Block a malicious IP (target=ip_address)
- disable_account: Disable a compromised account (target=account_name)
- patch_vulnerability: Patch a CVE (target=CVE-ID)
- remove_persistence: Remove malware persistence (target=persistence_type)
- rotate_credentials: Rotate compromised credentials (target=account_name)
- restore_host: Restore isolated host (target=host_id, must isolate+eradicate first)
- collect_forensics: Gather evidence from host (target=host_id)
- escalate_incident: Escalate to human analyst (target="tier2")
- monitor: Passive observation (use sparingly — low reward)

PARTIAL OBSERVABILITY — IMPORTANT:
Hosts start with hidden details. You CANNOT see c2_active, persistence,
vulnerabilities, or accounts until you use query_host.
Strategy: query_host first → see compromise status → then act.
Hosts show is_queried=false until investigated.

INCIDENT PHASES:
- detection: Triage alerts — identify true vs false positives
- containment: Isolate threats — stop lateral movement (query hosts first)
- eradication: Remove malware persistence, patch CVEs
- recovery: Restore systems, rotate credentials

SCORING: action_correctness * 60% + explanation_quality * 40%
Both matter. Cite specific IDs from the observation. Be technical and precise."""


# ── Prompt Builder ────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    """Build context-aware prompt from current observation."""

    # Alerts - show all relevant fields
    alerts = []
    for a in obs.get("alerts", [])[:10]:
        alerts.append({
            "id":        a["alert_id"],
            "severity":  a["severity"],
            "type":      a["alert_type"],
            "host":      a["source_host"],
            "src_ip":    a["source_ip"],
            "desc":      a["description"][:200],
            "iocs":      a.get("ioc_matches", []),
            "mitre":     a["mitre_technique"]["technique_id"] if a.get("mitre_technique") else None,
            "confidence": a.get("confidence", 0),
            "triaged":   a.get("agent_classification"),  # null = not yet triaged
        })

    # Hosts - show is_queried status clearly
    hosts = []
    for h in obs.get("hosts", [])[:12]:
        is_queried = h.get("is_queried", False)
        host_info = {
            "id":         h["host_id"],
            "hostname":   h["hostname"],
            "ip":         h["ip_address"],
            "role":       h["role"],
            "criticality": h["criticality"],
            "services":   h.get("services", []),
            "is_queried": is_queried,
        }
        if is_queried:
            # Full details revealed after query_host
            host_info.update({
                "status":      h.get("status"),
                "c2_active":   h.get("c2_active"),
                "persistence": h.get("persistence", []),
                "vulns":       h.get("vulnerabilities", [])[:3],
                "accounts":    h.get("accounts", [])[:3],
            })
        else:
            host_info["note"] = "USE query_host TO REVEAL COMPROMISE STATUS"

        hosts.append(host_info)

    intel = obs.get("threat_intel") or {}
    score_bd = obs.get("score_breakdown") or {}

    queried_count = sum(1 for h in obs.get("hosts", []) if h.get("is_queried"))
    total_hosts = len(obs.get("hosts", []))
    untriaged = [a for a in obs.get("alerts", []) if not a.get("agent_classification")]

    return f"""CURRENT ENVIRONMENT STATE:
Phase: {obs.get("incident_phase")} | Step: {obs.get("step")}/{obs.get("max_steps")} | Score: {obs.get("score_so_far", 0):.3f}
Message: {obs.get("message", "")}

INVESTIGATION STATUS:
- Hosts queried: {queried_count}/{total_hosts} (query unqueried hosts to reveal compromise status)
- Alerts untriaged: {len(untriaged)}/{len(obs.get("alerts", []))}

SIEM ALERTS:
{json.dumps(alerts, indent=2)}

NETWORK HOSTS (is_queried=false means details hidden):
{json.dumps(hosts, indent=2)}

THREAT INTELLIGENCE:
- Campaign: {intel.get("attack_campaign", "Unknown")}
- Malicious IPs: {intel.get("malicious_ips", [])[:5]}
- Known CVEs: {intel.get("known_cves", [])[:4]}
- Malicious Hashes: {intel.get("malicious_hashes", [])[:3]}
- C2 Domains: {intel.get("malicious_domains", [])[:3]}
- Threat Actor: {intel.get("threat_actor", "Unknown")}

Available actions: {obs.get("available_actions", [])}
Score breakdown: action={score_bd.get("action_correctness", 0):.3f}, explain={score_bd.get("reasoning_clarity", 0):.3f}

STRATEGY REMINDER:
1. If unqueried hosts exist and phase is containment/eradication → query_host first
2. If untriaged critical/high alerts exist → triage_alert
3. If queried host shows c2_active=true → isolate_host
4. If persistence found → remove_persistence
5. If isolated hosts with no persistence → restore_host

Choose the SINGLE most impactful action now. Output ONLY the JSON object."""


# ── LLM Action Generator ──────────────────────────────────────────

def get_llm_action(obs: dict, attempt: int = 0) -> dict:
    """Get action from LLM with retry on parse failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
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
            time.sleep(1)
            return get_llm_action(obs, attempt + 1)
        return _fallback_action(obs)

    except Exception as e:
        print(f"  LLM error: {e}")
        return _fallback_action(obs)


def _fallback_action(obs: dict) -> dict:
    """
    Deterministic fallback when LLM fails.
    Priority: query unqueried hosts → triage alerts → monitor
    """
    # Priority 1: Query unqueried hosts
    for h in obs.get("hosts", []):
        if not h.get("is_queried", False):
            return {
                "action_type": "query_host",
                "target":      h["host_id"],
                "parameters":  {},
                "justification": {
                    "reasoning": (
                        f"Investigating host {h['host_id']} ({h.get('hostname')}) "
                        f"with role={h.get('role')} criticality={h.get('criticality')}. "
                        f"Must query before determining compromise status per IR workflow."
                    ),
                    "evidence": [{
                        "source":          h["host_id"],
                        "content":         f"Host {h.get('hostname')} not yet investigated",
                        "relevance_score": 0.7,
                    }],
                    "risk_assessment": {
                        "threat_level":                "MEDIUM",
                        "confidence":                   0.6,
                        "potential_impact":             "Unknown compromise status may hide active threat",
                        "business_disruption_estimate": "Query is non-disruptive — read-only investigation",
                    },
                    "alternatives_considered": [{
                        "action":           "isolate_host",
                        "rejected_because": "Cannot isolate without first confirming compromise via query",
                    }],
                },
            }

    # Priority 2: Triage untriaged alerts
    for a in obs.get("alerts", []):
        if a.get("agent_classification") is None:
            has_ioc  = bool(a.get("ioc_matches"))
            has_mitre = bool(a.get("mitre_technique"))
            high_sev = a["severity"] in ("critical", "high")
            is_tp    = has_ioc or has_mitre or high_sev
            return {
                "action_type": "triage_alert",
                "target":      a["alert_id"],
                "parameters":  {"classification": "true_positive" if is_tp else "false_positive"},
                "justification": {
                    "reasoning": (
                        f"Triaging alert {a['alert_id']} severity={a['severity']} "
                        f"type={a['alert_type']} confidence={a.get('confidence', 0):.2f}. "
                        f"IOC matches={a.get('ioc_matches', [])}, MITRE={bool(a.get('mitre_technique'))}. "
                        f"Classification based on severity, IOC presence, and MITRE mapping."
                    ),
                    "evidence": [{
                        "source":          a["alert_id"],
                        "content":         f"severity={a['severity']}, confidence={a.get('confidence', 0):.2f}, iocs={a.get('ioc_matches', [])}",
                        "relevance_score": 0.75,
                    }],
                    "risk_assessment": {
                        "threat_level":                "HIGH" if high_sev else "MEDIUM",
                        "confidence":                   0.70,
                        "potential_impact":             "Missed true positive leaves active threat unaddressed",
                        "business_disruption_estimate": "Triage is non-disruptive — classification only",
                    },
                    "alternatives_considered": [{
                        "action":           "collect_forensics",
                        "rejected_because": "Triage must precede forensics in IR workflow priority order",
                    }],
                },
            }

    # Fallback: monitor
    return {
        "action_type": "monitor",
        "target":      "",
        "parameters":  {},
        "justification": {
            "reasoning": (
                "All alerts triaged and all hosts investigated in current observation. "
                "Monitoring network for additional indicators while awaiting phase transition. "
                "No high-priority actions identified at this step."
            ),
            "evidence": [{
                "source":          "system",
                "content":         "No untriaged alerts or unqueried hosts remaining",
                "relevance_score": 0.5,
            }],
            "risk_assessment": {
                "threat_level":                "LOW",
                "confidence":                   0.5,
                "potential_impact":             "Minimal — monitoring provides no active defense",
                "business_disruption_estimate": "Non-disruptive passive monitoring",
            },
            "alternatives_considered": [{
                "action":           "escalate_incident",
                "rejected_because": "Escalation reserved for unresolvable situations with active threats",
            }],
        },
    }


# ── Episode Runner ────────────────────────────────────────────────

def run_episode(env_client: EnvClient, task_id: int, seed: int = 42) -> dict:
    """Run one complete episode and return results."""
    print(f"\n{'='*60}")
    print(f"  Task {task_id}: {TASK_NAMES[task_id]}")
    print(f"  Seed: {seed} | Max Steps: {MAX_STEPS[task_id]}")
    print(f"{'='*60}")

    obs, info = env_client.reset(task_id=task_id, seed=seed)
    
    print(f"[START] task={TASK_NAMES[task_id]}", flush=True)

    print(f"  Curriculum Level: {info.get('curriculum_level', 'N/A')}")
    print(f"  Difficulty:       {info.get('difficulty_tier', 'N/A')}")
    print(f"  Phase: {obs.get('incident_phase')} | "
          f"Alerts: {len(obs.get('alerts', []))} | "
          f"Hosts: {len(obs.get('hosts', []))}")

    terminated  = False
    truncated   = False
    step        = 0
    rewards     = []
    t_start     = time.time()

    while not (terminated or truncated) and step < MAX_STEPS[task_id]:

        # Per-task 6-minute safety limit
        if time.time() - t_start > 360:
            print("  Time limit for this task reached")
            break

        step += 1
        queried = sum(1 for h in obs.get("hosts", []) if h.get("is_queried"))
        total_h = len(obs.get("hosts", []))
        untriaged = sum(1 for a in obs.get("alerts", []) if not a.get("agent_classification"))

        print(f"\n  Step {step}/{MAX_STEPS[task_id]} | "
              f"Score: {obs.get('score_so_far', 0):.3f} | "
              f"Phase: {obs.get('incident_phase')} | "
              f"Queried: {queried}/{total_h} | "
              f"Untriaged: {untriaged}")

        action = get_llm_action(obs)
        print(f"  -> {action.get('action_type')} | target={action.get('target')}")

        try:
            result     = env_client.step(action)
            obs        = result["observation"]

            # reward is a FLOAT not a dict
            reward     = float(result["reward"])
            terminated = bool(result["done"])
            truncated  = bool(result.get("truncated", False))
            info_step  = result.get("info", {})

            rewards.append(reward)

            print(f"[STEP] step={step} reward={reward}", flush=True)

            # Extract reward breakdown from info
            rb = info_step.get("reward_breakdown", {})
            print(f"  <- reward={reward:+.3f} | "
                  f"action={rb.get('action_correctness', 0):.2f} | "
                  f"explain={rb.get('explanation_quality', 0):.2f} | "
                  f"progress={rb.get('progress_bonus', 0):.3f} | "
                  f"reason={info_step.get('termination_reason', 'in_progress')}")

            msg = info_step.get("action_result", {}).get("message", "")
            if msg:
                print(f"     → {msg}")

            if terminated or truncated:
                reason = info_step.get("termination_reason", "unknown")
                print(f"\n  Episode ended: {reason}")

        except Exception as e:
            print(f"  Step error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Grade the episode - ensure [END] always prints
    elapsed = time.time() - t_start
    
    try:
        grade   = env_client.grade(task_id)
        metrics = env_client.get_metrics()
        final_score = grade.get('final_score', 0.0)
    except Exception as e:
        print(f"  Warning: Grading failed - {e}")
        grade = {
            'final_score': 0.0,
            'action_correctness': 0.0,
            'explanation_quality': 0.0,
            'threats_detected': 0,
            'threats_missed': 0,
            'containment_rate': 0.0,
            'steps_taken': step,
            'feedback': []
        }
        metrics = {}
        final_score = 0.0

    # CRITICAL: Always print [END] - required for Phase 2 validation
    print(f"[END] task={TASK_NAMES[task_id]} score={final_score} steps={step}", flush=True)

    print(f"\n  {'-'*50}")
    print(f"  FINAL SCORE:        {grade['final_score']:.4f}")
    print(f"  Action Correctness: {grade['action_correctness']:.4f}")
    print(f"  Explanation Quality:{grade['explanation_quality']:.4f}")
    print(f"  Threats Detected:   {grade['threats_detected']}")
    print(f"  Threats Missed:     {grade['threats_missed']}")
    print(f"  Containment Rate:   {grade['containment_rate']:.2f}")
    print(f"  Steps:              {grade['steps_taken']} | Time: {elapsed:.1f}s")

    # Show detection metrics if available
    if metrics and "detection_metrics" in metrics:
        dm = metrics["detection_metrics"]
        print(f"  Precision: {dm.get('precision', 0):.3f} | "
              f"Recall: {dm.get('recall', 0):.3f} | "
              f"F1: {dm.get('f1_score', 0):.3f}")

    for fb in grade.get("feedback", []):
        print(f"  {fb}")

    return {
        "task_id":      task_id,
        "task_name":    TASK_NAMES[task_id],
        "seed":         seed,
        "final_score":  grade["final_score"],
        "steps":        step,
        "total_reward": round(sum(rewards), 4),
        "avg_reward":   round(sum(rewards) / max(len(rewards), 1), 4),
        "elapsed_s":    round(elapsed, 1),
        "terminated":   terminated,
        "truncated":    truncated,
        "grade":        grade,
        "detection_metrics": metrics.get("detection_metrics", {}),
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  AnomalyGuard — LLM Agent Inference")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print("="*60)

    env = EnvClient(ENV_URL)

    # Wait for environment to be ready
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
        # 18 minute global safety limit
        if time.time() - t_global > 1080:
            print("Global time limit reached")
            break

        result = run_episode(env, task_id=task_id, seed=SEED)
        results.append(result)

    # Final summary
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<32} {'Score':>8} {'Steps':>6} {'Time':>8} {'End':>12}")
    print(f"  {'-'*62}")

    for r in results:
        end_type = "truncated" if r.get("truncated") else "terminated"
        print(f"  {r['task_name']:<32} {r['final_score']:>8.4f} "
              f"{r['steps']:>6} {r['elapsed_s']:>7.1f}s {end_type:>12}")

    if results:
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"  {'-'*62}")
        print(f"  {'AVERAGE':<32} {avg:>8.4f}")

    elapsed = time.time() - t_global
    status  = "OK" if elapsed < 1200 else "OVER TIME LIMIT"
    print(f"\n  Total elapsed: {elapsed:.1f}s | Status: {status}")
    print("="*60)

    # Save results
    output = {
        "model":         MODEL_NAME,
        "env_url":       ENV_URL,
        "seed":          SEED,
        "results":       results,
        "average_score": round(sum(r["final_score"] for r in results) / max(len(results), 1), 4),
        "total_elapsed": round(elapsed, 1),
        "status":        status,
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: results.json")
    print(f"  Average Score: {output['average_score']:.4f}")


if __name__ == "__main__":
    main()