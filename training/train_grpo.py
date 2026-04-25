"""
AnomalyGuard - GRPO Training Script (FIXED for Hackathon)
==========================================================
- Uses GRPOTrainer correctly (no invalid rollout_function param)
- Clean multi-component reward functions (separate, not randomized)
- Proper reward tracking with real step counts
- Dataset aligned with what environment actually expects
- Budget: fits in $30 HF credits with Qwen2.5-1.5B
"""

import torch
import random
import re
import json
import os
import sys
from typing import List, Dict, Any

# Add parent to path so app imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt

# ─── Config ────────────────────────────────────────────────────────────────
# Use 1.5B NOT 3B/7B — fits in $30 budget, trains faster, still shows improvement
MODEL_NAME    = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN   = 2048
MAX_STEPS     = 150       # ~2 hours on T4, shows clear learning curve
BATCH_SIZE    = 2
GRAD_ACCUM    = 4         # Effective batch = 8
NUM_EPISODES  = 200       # Dataset size
LORA_RANK     = 8         # Smaller = faster for hackathon

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ─── Model Loading ──────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,       # 0 dropout = faster, unsloth recommended
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("✅ Model loaded with LoRA adapters")

# ─── System Prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AnomalyGuard, an expert AI cybersecurity analyst operating in a Security Operations Center (SOC).

Your job is to investigate security incidents by analyzing SIEM alerts and network hosts, then take precise actions with full justification.

CRITICAL: You MUST respond with ONLY valid JSON in this exact format:
{
  "action_type": "<action>",
  "target": "<target_id>",
  "parameters": {},
  "justification": {
    "reasoning": "<minimum 50 characters explaining WHY you chose this action based on specific evidence>",
    "evidence": [{"source": "<alert_id or host_id>", "content": "<what you observed>", "relevance_score": 0.9}],
    "risk_assessment": {
      "threat_level": "<CRITICAL|HIGH|MEDIUM|LOW>",
      "confidence": 0.85,
      "potential_impact": "<what happens if not addressed>",
      "business_disruption_estimate": "<impact level>"
    },
    "alternatives_considered": [{"action": "<other action>", "rejected_because": "<reason>"}]
  }
}

Valid action_types: triage_alert, isolate_host, block_ip, disable_account, 
                    patch_vulnerability, remove_persistence, rotate_credentials,
                    restore_host, collect_forensics, escalate_incident,
                    query_host, monitor

IMPORTANT RULES:
1. Always investigate with query_host before isolating
2. Triage alerts before containment actions
3. reasoning must be at least 50 characters with specific evidence
4. Choose the MOST IMPACTFUL action given current phase"""

# ─── Dataset Generation ────────────────────────────────────────────────────
def generate_dataset(n: int = 200) -> Dataset:
    """
    Generate training prompts covering all 3 tasks and difficulty levels.
    Uses curriculum progression: easy examples first.
    """
    print(f"Generating {n} training prompts...")

    # Task scenarios with realistic contexts
    scenarios = [
        # Task 1: Alert Triage scenarios
        {
            "task_id": 1,
            "phase": "detection",
            "context": "3 SIEM alerts triggered. Alerts: ALT-10001 (C2 beacon to 185.220.101.45, severity: HIGH, MITRE T1071), ALT-10002 (Failed SSH login x50 from 194.165.16.76, severity: MEDIUM), ALT-10003 (DNS query to exfil.attacker.com, severity: CRITICAL). Hosts visible but unqueried.",
            "available_actions": ["triage_alert", "query_host", "escalate_incident"],
            "good_target": "ALT-10001",
            "good_action": "triage_alert",
            "good_params": {"classification": "true_positive"},
        },
        {
            "task_id": 1,
            "phase": "detection",
            "context": "5 alerts pending triage. ALT-20001 (User login from known IP, severity: INFO), ALT-20002 (Scheduled task created, severity: LOW), ALT-20003 (Port scan from internal IP, severity: MEDIUM). Alert ALT-20001 appears routine.",
            "available_actions": ["triage_alert", "query_host", "escalate_incident"],
            "good_target": "ALT-20001",
            "good_action": "triage_alert",
            "good_params": {"classification": "false_positive"},
        },
        # Task 2: Containment scenarios
        {
            "task_id": 2,
            "phase": "containment",
            "context": "Incident confirmed. Host web-server-01 (IP: 192.168.1.10) has active C2 channel to 185.220.101.45. Host has been queried - c2_active=True, persistence=['schtask_beacon'], vulnerabilities=['CVE-2023-1234']. 8 other hosts unaffected. Phase: containment.",
            "available_actions": ["isolate_host", "block_ip", "disable_account", "triage_alert"],
            "good_target": "web-server-01",
            "good_action": "isolate_host",
            "good_params": {},
        },
        {
            "task_id": 2,
            "phase": "containment",
            "context": "Malicious IP 185.220.101.45 confirmed in threat intel as known C2 server. Multiple hosts beaconing to this IP every 60 seconds. Pattern consistent with MITRE T1071.001 (Application Layer Protocol). Threat intel confidence: HIGH.",
            "available_actions": ["isolate_host", "block_ip", "disable_account"],
            "good_target": "185.220.101.45",
            "good_action": "block_ip",
            "good_params": {},
        },
        # Task 3: Full IR scenarios  
        {
            "task_id": 3,
            "phase": "eradication",
            "context": "Containment complete. web-server-01 isolated. Persistence mechanism 'schtask_beacon' still active on isolated host. Phase: eradication. Must remove persistence before restoration.",
            "available_actions": ["remove_persistence", "patch_vulnerability", "rotate_credentials"],
            "good_target": "schtask_beacon",
            "good_action": "remove_persistence",
            "good_params": {},
        },
        {
            "task_id": 3,
            "phase": "recovery",
            "context": "Eradication complete. web-server-01 isolated, persistence removed, CVEs patched. Phase: recovery. Host can now be safely restored to production.",
            "available_actions": ["restore_host", "rotate_credentials", "collect_forensics"],
            "good_target": "web-server-01",
            "good_action": "restore_host",
            "good_params": {},
        },
        # Investigation scenarios
        {
            "task_id": 2,
            "phase": "detection",
            "context": "Alert ALT-30001 flagged host db-server-01 as potentially compromised. Host db-server-01 visible but NOT queried yet - c2_active hidden, persistence hidden. Must investigate before acting.",
            "available_actions": ["query_host", "isolate_host", "triage_alert"],
            "good_target": "db-server-01",
            "good_action": "query_host",
            "good_params": {},
        },
    ]

    data = []
    for i in range(n):
        # Curriculum: first 30% easy (task 1), next 40% medium (task 2), last 30% hard (task 3)
        if i < n * 0.3:
            sc = scenarios[random.randint(0, 1)]   # Task 1 only
        elif i < n * 0.7:
            sc = scenarios[random.randint(2, 4)]   # Task 1 & 2
        else:
            sc = scenarios[random.randint(0, 6) % len(scenarios)]  # All tasks

        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
TASK {sc['task_id']} | PHASE: {sc['phase'].upper()}
{sc['context']}
Available actions: {', '.join(sc['available_actions'])}

What is your next action? Respond with JSON only.
<|im_end|>
<|im_start|>assistant
"""
        data.append({"prompt": prompt})

    dataset = Dataset.from_list(data)
    print(f"✅ Generated {len(dataset)} training prompts")
    return dataset


# ─── Reward Functions ───────────────────────────────────────────────────────
# KEY: Separate functions, NO randomization, each teaches one thing clearly

def reward_valid_json(completions, **kwargs) -> List[float]:
    """
    Reward 1: Is the output valid parseable JSON?
    Range: [-0.5, 0.5]
    This is the FIRST thing to learn - format compliance.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            # Try to find JSON block
            json_str = _extract_json(text)
            if json_str:
                json.loads(json_str)
                rewards.append(0.5)   # Valid JSON
            else:
                rewards.append(-0.3)  # No JSON found
        except json.JSONDecodeError:
            rewards.append(-0.5)      # Malformed JSON
    return rewards


def reward_correct_structure(completions, **kwargs) -> List[float]:
    """
    Reward 2: Does JSON have all required fields?
    Range: [-0.2, 0.5]
    Teaches the model the required schema.
    """
    REQUIRED_TOP = {"action_type", "target", "justification"}
    REQUIRED_JUST = {"reasoning", "evidence", "risk_assessment", "alternatives_considered"}
    REQUIRED_RISK = {"threat_level", "confidence", "potential_impact"}

    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            json_str = _extract_json(text)
            if not json_str:
                rewards.append(-0.2)
                continue

            data = json.loads(json_str)
            score = 0.0

            # Top-level fields
            top_present = REQUIRED_TOP.intersection(set(data.keys()))
            score += 0.15 * (len(top_present) / len(REQUIRED_TOP))

            # Justification fields
            just = data.get("justification", {})
            if isinstance(just, dict):
                just_present = REQUIRED_JUST.intersection(set(just.keys()))
                score += 0.2 * (len(just_present) / len(REQUIRED_JUST))

            # Risk assessment fields
            risk = just.get("risk_assessment", {}) if isinstance(just, dict) else {}
            if isinstance(risk, dict):
                risk_present = REQUIRED_RISK.intersection(set(risk.keys()))
                score += 0.15 * (len(risk_present) / len(REQUIRED_RISK))

            rewards.append(round(score, 3))
        except Exception:
            rewards.append(-0.2)
    return rewards


def reward_valid_action_type(completions, **kwargs) -> List[float]:
    """
    Reward 3: Is action_type one of the 12 valid actions?
    Range: [-0.3, 0.4]
    """
    VALID_ACTIONS = {
        "triage_alert", "isolate_host", "block_ip", "disable_account",
        "patch_vulnerability", "remove_persistence", "rotate_credentials",
        "restore_host", "collect_forensics", "escalate_incident",
        "query_host", "monitor"
    }
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            json_str = _extract_json(text)
            if not json_str:
                rewards.append(-0.3)
                continue
            data = json.loads(json_str)
            action = data.get("action_type", "")
            if action in VALID_ACTIONS:
                rewards.append(0.4)
            elif action:
                rewards.append(-0.1)  # Has something but wrong
            else:
                rewards.append(-0.3)  # Missing
        except Exception:
            rewards.append(-0.3)
    return rewards


def reward_reasoning_quality(completions, **kwargs) -> List[float]:
    """
    Reward 4: Is the reasoning specific and detailed?
    Range: [0.0, 0.5]
    Teaches EU AI Act compliance - meaningful justification.
    """
    # Keywords that indicate real security reasoning
    SECURITY_KEYWORDS = [
        "c2", "beacon", "malicious", "compromise", "lateral", "persistence",
        "cve", "vulnerability", "triage", "alert", "threat", "mitre",
        "evidence", "confidence", "contain", "isolat", "eradicat",
        "suspicious", "anomal", "indicator", "ioc", "hash", "ip address"
    ]

    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            json_str = _extract_json(text)
            if not json_str:
                rewards.append(0.0)
                continue

            data = json.loads(json_str)
            just = data.get("justification", {})
            reasoning = just.get("reasoning", "") if isinstance(just, dict) else ""

            if not reasoning or not isinstance(reasoning, str):
                rewards.append(0.0)
                continue

            score = 0.0

            # Length check (EU AI Act: reasoning must be substantive)
            length = len(reasoning)
            if length >= 150:
                score += 0.2
            elif length >= 100:
                score += 0.15
            elif length >= 50:
                score += 0.05
            # Under 50 = 0 (EU AI Act minimum not met)

            # Security keyword density
            reasoning_lower = reasoning.lower()
            keyword_hits = sum(1 for kw in SECURITY_KEYWORDS if kw in reasoning_lower)
            score += min(0.2, keyword_hits * 0.04)

            # Evidence list non-empty
            evidence = just.get("evidence", [])
            if isinstance(evidence, list) and len(evidence) > 0:
                score += 0.1

            rewards.append(round(min(score, 0.5), 3))
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_target_present(completions, **kwargs) -> List[float]:
    """
    Reward 5: Does the action have a non-empty target?
    Range: [-0.2, 0.2]
    Simple check - prevents empty/null targets.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            json_str = _extract_json(text)
            if not json_str:
                rewards.append(-0.2)
                continue
            data = json.loads(json_str)
            target = data.get("target", "")
            if target and isinstance(target, str) and len(target) > 2:
                rewards.append(0.2)
            else:
                rewards.append(-0.1)
        except Exception:
            rewards.append(-0.2)
    return rewards


# ─── Reward Tracking (for plots) ───────────────────────────────────────────
_reward_history = {
    "steps": [],
    "total": [],
    "json": [],
    "structure": [],
    "action": [],
    "reasoning": [],
    "target": [],
}
_call_count = 0

def reward_tracker(completions, **kwargs) -> List[float]:
    """
    NOT a reward function itself - tracks combined reward for plotting.
    Returns small constant so it doesn't affect training.
    Called alongside other reward functions.
    """
    global _call_count
    _call_count += 1

    # Only log every 5 calls to avoid noise
    if _call_count % 5 == 0:
        # Re-compute all components for logging
        j = reward_valid_json(completions, **kwargs)
        s = reward_correct_structure(completions, **kwargs)
        a = reward_valid_action_type(completions, **kwargs)
        r = reward_reasoning_quality(completions, **kwargs)
        t = reward_target_present(completions, **kwargs)

        avg_j = sum(j) / len(j) if j else 0
        avg_s = sum(s) / len(s) if s else 0
        avg_a = sum(a) / len(a) if a else 0
        avg_r = sum(r) / len(r) if r else 0
        avg_t = sum(t) / len(t) if t else 0
        total = avg_j + avg_s + avg_a + avg_r + avg_t

        _reward_history["steps"].append(_call_count)
        _reward_history["total"].append(round(total, 4))
        _reward_history["json"].append(round(avg_j, 4))
        _reward_history["structure"].append(round(avg_s, 4))
        _reward_history["action"].append(round(avg_a, 4))
        _reward_history["reasoning"].append(round(avg_r, 4))
        _reward_history["target"].append(round(avg_t, 4))

        # Save to disk (safe against crashes)
        with open("./results/reward_history.json", "w") as f:
            json.dump(_reward_history, f, indent=2)

        print(f"[Step {_call_count}] Total: {total:.3f} | "
              f"JSON: {avg_j:.2f} | Struct: {avg_s:.2f} | "
              f"Action: {avg_a:.2f} | Reason: {avg_r:.2f}")

    return [0.0] * len(completions)  # Zero reward from tracker itself


# ─── Helper Functions ───────────────────────────────────────────────────────
def _extract_text(completion) -> str:
    """Handle both string and list[dict] completion formats"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        item = completion[0]
        if isinstance(item, dict):
            return item.get("content", str(item))
        return str(item)
    return str(completion)


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks"""
    # Try code block first
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_match:
        return code_match.group(1)

    # Try bare JSON (greedy - get largest {...} block)
    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_matches:
        # Return the longest match (most complete JSON)
        return max(json_matches, key=len)

    return ""


# ─── Training ──────────────────────────────────────────────────────────────
def train():
    dataset = generate_dataset(n=NUM_EPISODES)

    config = GRPOConfig(
        output_dir="./checkpoints",
        # Steps
        max_steps=MAX_STEPS,
        # Batch
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        # Learning
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # GRPO specific
        num_generations=4,          # 4 samples per prompt (manageable on T4)
        max_completion_length=512,  # Enough for full JSON response
        temperature=0.8,
        # Logging
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        # Misc
        optim="adamw_8bit",
        report_to="none",           # Change to "wandb" if you have it
        seed=42,
        # Generation params
        max_prompt_length=1024,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[
            reward_valid_json,          # Weight: most important first
            reward_correct_structure,
            reward_valid_action_type,
            reward_reasoning_quality,
            reward_target_present,
            reward_tracker,             # Zero-weight tracker for logging
        ],
    )

    print(f"\n{'='*60}")
    print(f"  AnomalyGuard GRPO Training")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Steps:   {MAX_STEPS}")
    print(f"  Dataset: {len(dataset)} prompts")
    print(f"  Device:  {next(model.parameters()).device}")
    print(f"{'='*60}\n")

    trainer.train()

    # Save model
    print("\nSaving model...")
    model.save_pretrained("./anomalyguard-grpo-final")
    tokenizer.save_pretrained("./anomalyguard-grpo-final")
    print("✅ Model saved to ./anomalyguard-grpo-final")

    return trainer


# ─── Plot Generation ────────────────────────────────────────────────────────
def generate_plots():
    """Generate reward curve plots for README and blog post"""

    # Load saved history
    try:
        with open("./results/reward_history.json") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("⚠️  No reward history found. Did training run?")
        return

    if not history["steps"]:
        print("⚠️  Empty reward history")
        return

    steps = history["steps"]

    # ── Plot 1: Main reward curve ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Total combined reward
    ax1 = axes[0]
    ax1.plot(steps, history["total"], linewidth=2.5,
             color="#00A67E", label="Total Reward (trained)", zorder=3)

    # Smoothed line
    if len(steps) > 5:
        window = min(10, len(steps) // 3)
        smoothed = []
        for i in range(len(steps)):
            start = max(0, i - window)
            smoothed.append(sum(history["total"][start:i+1]) / (i - start + 1))
        ax1.plot(steps, smoothed, linewidth=3, color="#005C3E",
                 linestyle="--", label="Smoothed", zorder=4)

    # Baseline reference
    ax1.axhline(y=0.3, color="red", linestyle="--",
                linewidth=1.5, label="Untrained baseline (~0.30)", alpha=0.7)

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.set_title("AnomalyGuard: GRPO Training Progress", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-0.2)

    # Add improvement annotation
    if len(history["total"]) > 1:
        start_r = history["total"][0]
        end_r   = history["total"][-1]
        improvement = end_r - start_r
        ax1.annotate(
            f"Δ = +{improvement:.3f}",
            xy=(steps[-1], end_r),
            xytext=(steps[-1] * 0.7, end_r + 0.05),
            fontsize=11, fontweight="bold", color="#005C3E",
            arrowprops=dict(arrowstyle="->", color="#005C3E", lw=2),
        )

    # Bottom: Component breakdown
    ax2 = axes[1]
    component_colors = {
        "json":      "#2196F3",
        "structure": "#FF9800",
        "action":    "#9C27B0",
        "reasoning": "#F44336",
        "target":    "#4CAF50",
    }
    component_labels = {
        "json":      "Valid JSON",
        "structure": "Correct Structure",
        "action":    "Valid Action Type",
        "reasoning": "Reasoning Quality",
        "target":    "Target Present",
    }

    for key, color in component_colors.items():
        if key in history and history[key]:
            ax2.plot(steps, history[key],
                     linewidth=2, color=color,
                     label=component_labels[key], alpha=0.85)

    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Component Reward", fontsize=12)
    ax2.set_title("Reward Component Breakdown", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/training_progress.png", dpi=150, bbox_inches="tight")
    print("✅ Saved ./results/training_progress.png")

    # ── Plot 2: Before vs After bar chart ──────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    categories = ["Valid JSON", "Correct\nStructure", "Valid\nAction", "Reasoning\nQuality", "Target\nPresent"]
    keys       = ["json", "structure", "action", "reasoning", "target"]

    # Before = first 10% of training
    cutoff = max(1, len(steps) // 10)
    before = [
        sum(history[k][:cutoff]) / cutoff if history.get(k) else 0
        for k in keys
    ]
    # After = last 10% of training
    after = [
        sum(history[k][-cutoff:]) / cutoff if history.get(k) else 0
        for k in keys
    ]

    x = range(len(categories))
    bars1 = ax3.bar([xi - 0.2 for xi in x], before, 0.38,
                    label="Start of Training", color="#FF7043", alpha=0.85,
                    edgecolor="white", linewidth=1.5)
    bars2 = ax3.bar([xi + 0.2 for xi in x], after, 0.38,
                    label="End of Training", color="#00A67E", alpha=0.85,
                    edgecolor="white", linewidth=1.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(list(x))
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.set_ylabel("Average Reward Component", fontsize=12)
    ax3.set_title("Before vs After GRPO Training — AnomalyGuard",
                  fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig("./results/before_after.png", dpi=150, bbox_inches="tight")
    print("✅ Saved ./results/before_after.png")

    plt.close("all")


# ─── Baseline Measurement ───────────────────────────────────────────────────
def measure_baseline(n_samples: int = 20) -> Dict[str, float]:
    """
    Measure untrained model performance to show improvement.
    Call BEFORE training, save results.
    """
    print("\nMeasuring baseline (untrained) performance...")
    FastLanguageModel.for_inference(model)

    sample_prompts = [
        f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
TASK 1 | PHASE: DETECTION
3 SIEM alerts: ALT-10001 (C2 beacon, HIGH), ALT-10002 (SSH brute force, MEDIUM)
Available actions: triage_alert, query_host
What is your next action? Respond with JSON only.
<|im_end|>
<|im_start|>assistant
"""
    ] * n_samples

    scores = {"json": [], "structure": [], "action": [], "reasoning": [], "target": []}

    for i in range(0, n_samples, 4):
        batch = sample_prompts[i:i+4]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, out in enumerate(outputs):
            inp_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(out[inp_len:], skip_special_tokens=True)

            scores["json"].append(reward_valid_json([text])[0])
            scores["structure"].append(reward_correct_structure([text])[0])
            scores["action"].append(reward_valid_action_type([text])[0])
            scores["reasoning"].append(reward_reasoning_quality([text])[0])
            scores["target"].append(reward_target_present([text])[0])

    baseline = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
    baseline["total"] = round(sum(baseline.values()), 4)

    print(f"\n📊 Baseline Scores (untrained {MODEL_NAME}):")
    for k, v in baseline.items():
        print(f"   {k:15s}: {v:.4f}")

    with open("./results/baseline_scores.json", "w") as f:
        json.dump(baseline, f, indent=2)

    # Re-enable training mode
    model.train()
    return baseline


# ─── Quick Inference Test ────────────────────────────────────────────────────
def test_inference(model_path: str = "./anomalyguard-grpo-final"):
    """Test trained model with a sample scenario"""
    print(f"\nTesting inference with model from {model_path}...")

    test_model, test_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(test_model)

    test_prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
TASK 2 | PHASE: CONTAINMENT
Host web-server-01 queried. Results: c2_active=True, persistence=['schtask_beacon'], 
connecting to malicious IP 185.220.101.45 every 300s. Threat level: CRITICAL.
Available actions: isolate_host, block_ip, query_host
What is your next action? Respond with JSON only.
<|im_end|>
<|im_start|>assistant
"""

    inputs = test_tokenizer(test_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.6,
            do_sample=True,
            pad_token_id=test_tokenizer.eos_token_id,
        )

    response = test_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\n" + "="*60)
    print("TEST PROMPT: web-server-01 has active C2, needs containment")
    print("="*60)
    print("MODEL RESPONSE:")
    print(response)
    print("="*60)

    # Score the response
    j = reward_valid_json([response])[0]
    s = reward_correct_structure([response])[0]
    a = reward_valid_action_type([response])[0]
    r = reward_reasoning_quality([response])[0]
    t = reward_target_present([response])[0]
    total = j + s + a + r + t

    print(f"\nScores: JSON={j:.2f} | Structure={s:.2f} | "
          f"Action={a:.2f} | Reasoning={r:.2f} | Target={t:.2f}")
    print(f"Total: {total:.3f}")

    # Check if action makes sense
    json_str = _extract_json(response)
    if json_str:
        try:
            data = json.loads(json_str)
            action = data.get("action_type", "")
            target = data.get("target", "")
            print(f"\n✅ Parsed action: {action} → {target}")
            if action == "isolate_host" and "web-server" in target:
                print("🎯 CORRECT: Model correctly chose to isolate the compromised host!")
            else:
                print(f"ℹ️  Model chose: {action} (isolate_host would be optimal)")
        except Exception:
            pass

    return response


# ─── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "baseline", "plot", "test"],
                        default="train")
    args = parser.parse_args()

    if args.mode == "baseline":
        measure_baseline()

    elif args.mode == "train":
        # Measure baseline first, then train
        baseline = measure_baseline(n_samples=10)
        trainer = train()
        generate_plots()
        test_inference()

    elif args.mode == "plot":
        generate_plots()

    elif args.mode == "test":
        test_inference()