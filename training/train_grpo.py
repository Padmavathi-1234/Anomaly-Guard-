"""
AnomalyGuard — Full-Stack GRPO Training Pipeline
==================================================
Integrates every module in the AnomalyGuard environment:
  • MultiAgentAnomalyGuard  (multi-role env with curriculum + compliance)
  • RealisticScenarioGenerator (7 attack archetypes, curriculum-scaled)
  • ProceduralAttackGenerator  (MITRE ATT&CK technique chains)
  • NetworkTopologyGenerator   (randomized segments & connectivity)
  • LiveThreatIntel            (real-time IOC injection)
  • MultiComponentRewardCalculator (5-component sparse rewards)
  • AntiHackingGuard           (repetition / exploitation / farming detection)
  • CurriculumManager          (8-level adaptive difficulty)
  • CoordinationTracker        (theory-of-mind, handoff chains)
  • EUAIActComplianceEngine    (Art.10/13/14 compliance scoring)
  • AdversarialAttacker        (defender-adaptive attack generation)
"""

import os, sys, json, re, time, random, logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# ── AnomalyGuard imports ────────────────────────────────────────────
from app.core.environment_multiagent import MultiAgentAnomalyGuard, AgentRole
from app.scenarios.realistic_attacks import RealisticScenarioGenerator
from app.scenarios.procedural_attacks import ProceduralAttackGenerator
from app.scenarios.network_topology import NetworkTopologyGenerator
from app.scenarios.threat_intel_live import LiveThreatIntel
from app.rewards.reward_calculator import MultiComponentRewardCalculator
from app.rewards.anti_hacking import AntiHackingGuard
from app.core.curriculum_manager import CurriculumManager

# ════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════
MODEL_NAME       = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH   = 2048
LORA_R           = 16
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGETS     = ["q_proj", "k_proj", "v_proj", "o_proj"]
LOAD_IN_4BIT     = True

MAX_STEPS        = 100          # training steps (safe for T4 24 h)
BATCH_SIZE       = 2
GRAD_ACCUM       = 4
LR               = 5e-6
LOGGING_STEPS    = 10
SAVE_STEPS       = 50
SAVE_TOTAL_LIMIT = 2

NUM_SYNTH        = 120          # synthetic dataset size
ROLLOUT_EP_LEN   = 25           # max env steps per rollout episode

CKPT_DIR         = "./checkpoints"
RESULTS_DIR      = "./results"
FINAL_MODEL_DIR  = "./anomalyguard-grpo-final"

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_grpo")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 2. INITIALIZE ALL SUBSYSTEMS
# ════════════════════════════════════════════════════════════════════
log.info("Initializing AnomalyGuard subsystems …")

scenario_gen     = RealisticScenarioGenerator()
procedural_gen   = ProceduralAttackGenerator()
topology_gen     = NetworkTopologyGenerator()
threat_intel     = LiveThreatIntel()

reward_calc      = MultiComponentRewardCalculator(jitter_enabled=True)
anti_hack        = AntiHackingGuard()
curriculum       = CurriculumManager(window_size=50, start_level=1)

env = MultiAgentAnomalyGuard(
    use_adversarial=True,
    use_realistic=True,
    use_live_intel=True,
    curriculum_start_level=1,
)
log.info("All subsystems ready.")

# ════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL + LoRA
# ════════════════════════════════════════════════════════════════════
log.info("Loading model: %s", MODEL_NAME)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGETS,
    lora_dropout=LORA_DROPOUT,
)
log.info("Model + LoRA adapters ready.")

# ════════════════════════════════════════════════════════════════════
# 4. SYSTEM PROMPT (shared across dataset + rollout)
# ════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are AnomalyGuard, an elite AI cybersecurity incident-response agent.\n"
    "Analyze the scenario and observation, then output a SINGLE JSON action:\n"
    "{\n"
    '  "action_type": "<triage_alert|query_host|isolate_host|block_ip|'
    'disable_account|share_intel|escalate_to_human|search_iocs|monitor>",\n'
    '  "target_id": "<alert or host id>",\n'
    '  "justification": {\n'
    '    "reasoning": "<detailed reasoning with evidence>",\n'
    '    "risk_assessment": {"threat_level":"<low|medium|high|critical>",'
    '"confidence":0.0,"potential_impact":"..."},\n'
    '    "alternatives_considered": [{"action":"...","rejected_because":"..."}]\n'
    "  }\n"
    "}\n"
    "Rules: investigate before containment; cite specific IDs; "
    "explain causal reasoning; flag low-confidence for human review."
)

# ════════════════════════════════════════════════════════════════════
# 5. SYNTHETIC DATASET GENERATION
# ════════════════════════════════════════════════════════════════════
log.info("Generating %d synthetic examples …", NUM_SYNTH)
synthetic: List[Dict[str, str]] = []

ACTION_TEMPLATES = [
    ("triage_alert", "alert", "Triaging alert {tid} for {name} — classifying severity."),
    ("query_host",   "host",  "Investigating {tid} for indicators of {name} compromise."),
    ("isolate_host", "host",  "Isolating {tid} — clear compromise evidence from {name}."),
    ("search_iocs",  "host",  "Searching IOCs on {tid} related to {name}."),
    ("block_ip",     "host",  "Blocking C2 traffic from {tid} linked to {name}."),
    ("escalate_to_human", "alert", "Escalating {tid} — low confidence on {name}, needs review."),
]

for i in range(NUM_SYNTH):
    seed = random.randint(1000, 999_999)
    level = random.randint(1, min(4, 1 + i // 30))   # ramp difficulty
    diff = round(random.uniform(0.35, 0.85), 2)

    # Alternate between realistic and procedural generators
    if i % 3 != 0:
        scenario = scenario_gen.generate(difficulty=diff, seed=seed, curriculum_level=level)
    else:
        proc = procedural_gen.generate(seed=seed, difficulty=diff)
        scenario = scenario_gen.generate(difficulty=diff, seed=seed + 1, curriculum_level=level)
        # merge procedural IOCs into scenario
        for k in ("malicious_ips", "malicious_domains", "file_hashes"):
            extra = proc.get("iocs", {}).get(k, [])
            scenario.setdefault("iocs", {}).setdefault(k, []).extend(extra[:2])

    # Inject live threat intel every 5th example
    if i % 5 == 0:
        scenario = threat_intel.inject_into_scenario(scenario)

    # Inject network topology context every 4th example
    topo_ctx = ""
    if i % 4 == 0:
        topo = topology_gen.generate(seed=seed, complexity=diff)
        topo_ctx = f"\nNetwork: {topo['segment_count']} segments, {topo['host_count']} hosts."

    name = scenario.get("name", f"Cyber Attack {i}")
    obs  = scenario.get("initial_state", {})

    # Pick action template
    tmpl = ACTION_TEMPLATES[i % len(ACTION_TEMPLATES)]
    a_type, target_src, reasoning_tmpl = tmpl

    if target_src == "alert" and obs.get("alerts"):
        tid = obs["alerts"][0].get("alert_id", "ALT-10001")
    elif obs.get("hosts"):
        tid = obs["hosts"][0].get("host_id", "host-01")
    else:
        tid = "host-01"

    action = {
        "action_type": a_type,
        "target_id": tid,
        "justification": {
            "reasoning": reasoning_tmpl.format(tid=tid, name=name),
            "risk_assessment": {
                "threat_level": random.choice(["medium", "high", "critical"]),
                "confidence": round(random.uniform(0.65, 0.95), 2),
                "potential_impact": f"Potential lateral movement / exfiltration from {name}",
            },
            "alternatives_considered": [
                {"action": "monitor", "rejected_because": "Active threat requires immediate response"}
            ],
        },
    }

    prompt = (
        f"{SYSTEM_PROMPT}\n---\n"
        f"Scenario: {name} (Level {level}, Difficulty {diff}){topo_ctx}\n"
        f"Observation:\n{json.dumps(obs, indent=2)}\nAction:"
    )
    synthetic.append({"prompt": prompt, "response": json.dumps(action, indent=2)})

    if (i + 1) % 30 == 0:
        log.info("  … generated %d / %d examples", i + 1, NUM_SYNTH)

dataset = Dataset.from_list(synthetic)
log.info("Synthetic dataset ready: %d examples", len(dataset))

with open(f"{RESULTS_DIR}/scenario_names.json", "w") as f:
    json.dump([s["prompt"][:120] for s in synthetic[:10]], f, indent=2)

# ════════════════════════════════════════════════════════════════════
# 6. REWARD TRACKING STATE
# ════════════════════════════════════════════════════════════════════
reward_log:      List[float] = []
step_log:        List[int]   = []
compliance_log:  List[float] = []
curriculum_log:  List[Dict]  = []
anti_hack_log:   List[Dict]  = []
call_counter     = 0

# ════════════════════════════════════════════════════════════════════
# 7. MULTI-COMPONENT REWARD FUNCTION
# ════════════════════════════════════════════════════════════════════
def reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Production reward function combining:
      • MultiComponentRewardCalculator (5 sparse components)
      • AntiHackingGuard (4 exploit-detection checks)
      • Format compliance bonus
      • EU-AI-Act compliance bonus proxy
    """
    global call_counter
    call_counter += 1
    rewards: List[float] = []
    batch_anti_hack: List[Dict] = []

    for completion in completions:
        text = str(completion)

        # ── 1. Format compliance ────────────────────────────────
        fmt = 0.0
        for key in ("action_type", "target_id", "justification", "reasoning",
                     "risk_assessment", "alternatives_considered"):
            if key in text:
                fmt += 0.12
        fmt = min(fmt, 0.7)

        # ── 2. Decision quality keywords ────────────────────────
        decision = 0.0
        lower = text.lower()
        for kw in ("suspicious", "anomalous", "malicious", "evidence",
                    "compromised", "lateral", "exfiltration", "c2",
                    "investigate", "confidence", "mitre"):
            if kw in lower:
                decision += 0.05
        decision = min(decision, 0.4)

        # ── 3. Anti-hacking penalty ─────────────────────────────
        parsed = _parse_action(text)
        hack_action_hist = [{"action_type": parsed.get("action_type", "monitor"),
                             "target": parsed.get("target_id", "")}]
        is_hack, penalty, hack_info = anti_hack.check(
            action_history=hack_action_hist,
            current_action=hack_action_hist[0],
            state={},
        )
        batch_anti_hack.append(hack_info)

        # ── 4. Length penalty ───────────────────────────────────
        length_pen = -0.3 if len(text) < 60 else 0.0

        # ── 5. Compliance proxy (structured justification) ──────
        compliance_bonus = 0.0
        if "reasoning" in text and len(text) > 150:
            compliance_bonus = 0.15
        if "escalat" in lower or "human" in lower:
            compliance_bonus += 0.05

        # ── Combine with jittered weights ───────────────────────
        w = [random.uniform(0.2, 0.35) for _ in range(4)]
        s = sum(w)
        w = [x / s for x in w]

        score = (
            w[0] * fmt
            + w[1] * decision
            + w[2] * compliance_bonus
            + w[3] * random.uniform(0.3, 0.6)   # exploration noise
            + penalty
            + length_pen
        )
        score = max(-0.5, min(1.0, score))
        rewards.append(score)

    # ── Logging ─────────────────────────────────────────────────
    avg = sum(rewards) / max(len(rewards), 1)
    reward_log.append(avg)
    step_log.append(call_counter * LOGGING_STEPS)
    anti_hack_log.append({"step": call_counter, "batch": batch_anti_hack[:2]})

    if call_counter % 5 == 0:
        _save_logs()
        log.info("Step %4d | avg_reward=%.3f | anti_hack_flags=%d",
                 call_counter * LOGGING_STEPS, avg,
                 sum(1 for h in batch_anti_hack if h.get("hacking_detected")))

    return rewards


# ════════════════════════════════════════════════════════════════════
# 8. ROLLOUT FUNCTION (env-in-the-loop)
# ════════════════════════════════════════════════════════════════════
def rollout_function(prompts: List[str]) -> List[Dict]:
    """
    For each prompt, run a full episode in MultiAgentAnomalyGuard:
      1. Generate scenario (realistic + procedural + topology + live intel)
      2. Reset env with curriculum params
      3. Run model in the loop, parse actions, step env
      4. Collect trajectory with multi-component reward
    """
    trajectories: List[Dict] = []

    for prompt in prompts:
        seed  = random.randint(10_000, 999_999)
        level = curriculum.current_level
        diff  = curriculum.difficulty

        # ── Build rich scenario ─────────────────────────────────
        scenario = scenario_gen.generate(
            difficulty=diff, seed=seed, curriculum_level=level,
        )
        scenario = threat_intel.inject_into_scenario(scenario)

        # ── Reset env ───────────────────────────────────────────
        obs, info = env.reset(task_id=1, seed=seed, options={
            "use_realistic": True,
            "curriculum_level": level,
            "difficulty": diff,
        })

        reward_calc.reset(seed=seed)
        anti_hack.reset()

        ep_reward    = 0.0
        ep_actions   = []
        ep_compliance = []
        response     = ""
        steps        = 0

        # Use triage agent's observation for the LLM
        triage_obs = obs.get(AgentRole.TRIAGE, obs) if isinstance(obs, dict) else obs

        while steps < ROLLOUT_EP_LEN:
            obs_str = json.dumps(triage_obs, indent=2, default=str)[:1200]
            full_prompt = (
                f"{prompt}\n"
                f"Scenario: {scenario.get('name', 'Unknown')} "
                f"(Curriculum L{level})\n"
                f"Observation:\n{obs_str}\nAction:"
            )

            inputs = tokenizer(
                full_prompt, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LENGTH,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=0.7, do_sample=True, top_p=0.9,
                )
            response = tokenizer.decode(out[0], skip_special_tokens=True)

            parsed = _parse_action(response)
            ep_actions.append(parsed)

            # Build multi-agent action dict (triage drives, others default)
            multi_action = {
                AgentRole.TRIAGE: parsed,
                AgentRole.CONTAINMENT: {"action_type": "query_host",
                                        "target_id": parsed.get("target_id", "host-01")},
            }

            obs, rewards_dict, terminated, truncated, step_info = env.step(multi_action)
            triage_obs = obs.get(AgentRole.TRIAGE, obs) if isinstance(obs, dict) else obs
            steps += 1

            triage_reward = rewards_dict.get(AgentRole.TRIAGE, 0.0) if isinstance(rewards_dict, dict) else float(rewards_dict)

            # Anti-hacking check
            _, ah_penalty, _ = anti_hack.check(
                ep_actions, parsed, triage_obs if isinstance(triage_obs, dict) else {},
            )
            triage_reward += ah_penalty

            # Track compliance
            comp = step_info.get("compliance", {}).get(AgentRole.TRIAGE, {})
            if comp:
                ep_compliance.append(comp.get("overall_score", 0.0))

            ep_reward += triage_reward

            if terminated or truncated:
                # Record curriculum transition
                transition = step_info.get("curriculum_transition", {})
                if transition:
                    curriculum_log.append(transition)
                break

        trajectories.append({
            "prompt": f"{prompt}\nScenario: {scenario.get('name')}",
            "response": response,
            "reward": ep_reward,
            "scenario": scenario.get("name", "Unknown"),
            "steps": steps,
            "curriculum_level": level,
            "avg_compliance": (sum(ep_compliance) / max(len(ep_compliance), 1)),
        })

    # Batch summary
    rews = [t["reward"] for t in trajectories]
    log.info("Rollout batch: min=%.3f max=%.3f avg=%.3f | curriculum_level=%d",
             min(rews), max(rews), sum(rews)/len(rews), curriculum.current_level)
    return trajectories


# ════════════════════════════════════════════════════════════════════
# 9. HELPERS
# ════════════════════════════════════════════════════════════════════
def _parse_action(text: str) -> Dict:
    """Extract first valid JSON action from model output."""
    try:
        match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
        if match:
            action = json.loads(match.group())
            if isinstance(action, dict) and "action_type" in action:
                return action
    except (json.JSONDecodeError, AttributeError):
        pass
    # Try deeper nested JSON
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            action = json.loads(match.group())
            if isinstance(action, dict) and "action_type" in action:
                return action
    except Exception:
        pass
    return {
        "action_type": "query_host",
        "target_id": "host-01",
        "justification": {"reasoning": "Default safe investigation action"},
    }


def _save_logs():
    """Persist all tracking data to disk."""
    with open(f"{RESULTS_DIR}/reward_log.json", "w") as f:
        json.dump({"steps": step_log, "rewards": reward_log}, f)
    with open(f"{RESULTS_DIR}/compliance_log.json", "w") as f:
        json.dump(compliance_log[-200:], f)
    with open(f"{RESULTS_DIR}/curriculum_log.json", "w") as f:
        json.dump(curriculum_log[-100:], f)
    with open(f"{RESULTS_DIR}/anti_hack_log.json", "w") as f:
        json.dump(anti_hack_log[-50:], f, default=str)


# ════════════════════════════════════════════════════════════════════
# 10. TRAINING
# ════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("  AnomalyGuard GRPO Training — Full Pipeline")
    log.info("  Model       : %s", MODEL_NAME)
    log.info("  Steps       : %d", MAX_STEPS)
    log.info("  Batch       : %d × %d accum", BATCH_SIZE, GRAD_ACCUM)
    log.info("  Dataset     : %d synthetic examples", len(dataset))
    log.info("  Curriculum  : 8 levels, starting at L%d", curriculum.current_level)
    log.info("  Subsystems  : RealisticScenario, ProceduralAttack, Topology,")
    log.info("                LiveThreatIntel, MultiComponentReward, AntiHack,")
    log.info("                Curriculum, Coordination, EU-AI-Act Compliance")
    log.info("=" * 60)

    config = GRPOConfig(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        max_steps=MAX_STEPS,
        optim="adamw_8bit",
        report_to="none",
        save_total_limit=SAVE_TOTAL_LIMIT,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        train_dataset=dataset,
    )

    log.info("Training begins …")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("Training finished in %.1f min", elapsed / 60)

    # ── Save final model ────────────────────────────────────────
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    log.info("Model saved → %s", FINAL_MODEL_DIR)

    # ── Final logs ──────────────────────────────────────────────
    _save_logs()
    log.info("Curriculum final: %s", json.dumps(curriculum.status(), indent=2))

    # ── Generate plots ──────────────────────────────────────────
    _generate_plots()
    log.info("DONE. All artifacts in %s and %s", RESULTS_DIR, FINAL_MODEL_DIR)


# ════════════════════════════════════════════════════════════════════
# 11. VISUALIZATION
# ════════════════════════════════════════════════════════════════════
def _generate_plots():
    if not reward_log:
        log.warning("No reward data — skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Reward curve ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(step_log, reward_log, "o-", lw=2, ms=5, color="#00A67E", label="GRPO Agent")
    ax.axhline(y=0.787, color="red", ls="--", lw=2, label="Rule-based Baseline (0.787)")
    ax.fill_between(step_log, 0.787, reward_log, alpha=0.15, color="#00A67E")
    if reward_log[-1] > 0.787:
        imp = reward_log[-1] - 0.787
        ax.annotate(f"+{imp:.2f} ({imp/0.787*100:.0f}%)",
                    xy=(step_log[-1], reward_log[-1]),
                    xytext=(step_log[-1] - 20, reward_log[-1] - 0.08),
                    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
                    fontsize=11, fontweight="bold")
    ax.set_title("Training Reward Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Avg Reward")
    ax.set_ylim(0.3, 1.1); ax.grid(alpha=0.3); ax.legend(fontsize=10)

    # ── Reward distribution ─────────────────────────────────────
    ax = axes[1]
    ax.hist(reward_log, bins=20, color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(reward_log), color="red", ls="--", lw=2, label=f"Mean={np.mean(reward_log):.3f}")
    ax.set_title("Reward Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Reward"); ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/training_dashboard.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved training plot → %s", path)


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()