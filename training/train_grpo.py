"""
AnomalyGuard - GRPO Training (Final Version for Hackathon)
Uses your own realistic_attacks.py, scenario_base.py and threat_live.py
"""

import torch
import random
import re
import json
from typing import List, Dict
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from app.core.environment_multiagent import MultiAgentAnomalyGuard
from datasets import Dataset

# Import from your scenarios folder
from app.scenarios.realistic_attacks import RealisticScenarioGenerator
from app.scenarios.threat_live import LiveThreatIntel

# ====================== CONFIG ======================
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
MAX_STEPS = 100                    # Safe for 24h hackathon on T4
BATCH_SIZE = 2

print("Loading your Realistic Scenario Generator...")
scenario_generator = RealisticScenarioGenerator()
threat_intel = LiveThreatIntel()

print("Initializing AnomalyGuard Environment...")
env = MultiAgentAnomalyGuard(use_realistic=True)

print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

print("✅ Setup Complete! Starting Training...\n")

# Create a dummy dataset to satisfy GRPOTrainer's requirements
dummy_prompts = [
    "You are an AI cybersecurity agent. Analyze the observation and respond with the best action.",
    "You are a cybersecurity incident responder. Based on the observation, determine the optimal action.",
    "As a security analyst, examine the observation and decide on the most appropriate action."
]
dummy_dataset = Dataset.from_list([{"prompt": p} for p in dummy_prompts])

# ====================== REWARD FUNCTION ======================
def reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 1.0
        if "action_type" in str(completion):
            score += 1.0
        if "justification" in str(completion):
            score += 0.5
        rewards.append(score)
    return rewards

# ====================== ROLLOUT FUNCTION ======================
def rollout_function(prompts: List[str]) -> List[Dict]:
    trajectories = []
    
    for prompt in prompts:
        # Generate fresh realistic scenario using YOUR generator
        scenario = scenario_generator.generate(
            difficulty=random.uniform(0.5, 0.85),
            seed=random.randint(10000, 999999),
            curriculum_level=random.randint(2, 6)
        )
        
        # Add live threat intel
        scenario = threat_intel.inject_into_scenario(scenario)
        
        obs = scenario.get("initial_state", {})
        done = False
        episode_reward = 0
        steps = 0
        response = ""

        while not done and steps < 25:
            full_prompt = f"""{prompt}
Scenario: {scenario.get('name', 'Unknown Cyber Attack')}
Observation: {json.dumps(obs, indent=2)}
Action:"""
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            action = parse_action(response)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break

        trajectories.append({
            "prompt": f"{prompt}\nScenario: {scenario.get('name')}",
            "response": response,
            "reward": episode_reward,
            "scenario": scenario.get('name', 'Unknown')
        })
    
    return trajectories


def parse_action(response: str) -> Dict:
    try:
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            action = json.loads(json_match.group())
            if isinstance(action, dict) and "action_type" in action:
                return action
    except:
        pass
    return {
        "action_type": "query_host",
        "target_id": "host-01",
        "justification": {"reasoning": "Default safe investigation"}
    }


# ====================== TRAINING ======================
print("Starting GRPO Training with your realistic dynamic scenarios...")

config = GRPOConfig(
    output_dir="./checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=10,
    max_steps=MAX_STEPS,
    optim="adamw_8bit",
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    reward_funcs=[reward_func],       # Added this
    train_dataset=dummy_dataset,      # Added this
    rollout_function=rollout_function,
)

trainer.train()

model.save_pretrained("./anomalyguard-grpo-final")
tokenizer.save_pretrained("./anomalyguard-grpo-final")

print("\n🎉 TRAINING COMPLETED!")
print("Model saved to: ./anomalyguard-grpo-final")
print("Now create reward plots and update your README.md")