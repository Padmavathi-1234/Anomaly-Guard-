"""
Optimized RL Post-Training for Hugging Face Credits
===================================================
Optimized for speed and stability on A100/H100 GPUs
"""

import torch
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import openenv
from app.core.environment_multiagent import MultiAgentAnomalyGuard
from typing import List, Dict

# ====================== CONFIG ======================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
EPOCHS = 2
MAX_STEPS = 1000          # Safety limit

print("Loading environment...")
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

print("Model loaded!")

# ====================== ROLLOUT ======================
def rollout_function(prompts: List[str]) -> List[Dict]:
    trajectories = []
    
    for prompt in prompts:
        obs, info = env.reset(task_id=3, seed=None)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 60:
            # Generate action from model
            full_prompt = f"{prompt}\nObservation: {obs}\nAction:"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse action
            action = parse_action(response)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        trajectories.append({
            "prompt": prompt,
            "response": response,
            "reward": episode_reward
        })
    
    return trajectories

def parse_action(response: str) -> Dict:
    import re, json
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            action = json.loads(json_match.group())
            if "action_type" in action:
                return action
    except:
        pass
    return {
        "action_type": "query_host",
        "target_id": "host-01",
        "justification": {"reasoning": "Default action"}
    }

# ====================== START TRAINING ======================
print("Starting RL Post-Training...")

config = GRPOConfig(
    output_dir="./checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    logging_steps=20,
    save_steps=200,
    max_steps=MAX_STEPS,
    evaluation_strategy="steps",
    eval_steps=200,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    tokenizer=tokenizer,
    rollout_function=rollout_function,
)

trainer.train()

# Save final model
model.save_pretrained("./anomalyguard-trained-final")
tokenizer.save_pretrained("./anomalyguard-trained-final")

print("\n✅ Training Completed!")
print("Model saved to: ./anomalyguard-trained-final")