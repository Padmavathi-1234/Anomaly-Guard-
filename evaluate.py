"""
Automatic Before vs After Evaluation
Run this after training is done
"""

from app.core.environment_multiagent import MultiAgentAnomalyGuard, AgentRole
import json
import time

def evaluate_model(is_trained=False, episodes=30, model=None):
    env = MultiAgentAnomalyGuard(use_realistic=True)
    
    total_prevention = 0
    total_detection = 0
    total_reward = 0
    total_coordination = 0
    total_compliance = 0
    compliance_steps = 0
    successful = 0

    print(f"Running {'Trained' if is_trained else 'Baseline'} evaluation...")

    for ep in range(episodes):
        obs, info = env.reset(task_id=3, seed=ep)
        done = False
        episode_reward = 0
        detection_step = None
        prevented = False
        step = 0

        while not done and step < 80:
            if is_trained and model:
                single_action = get_trained_action(obs, model)
            else:
                single_action = get_baseline_action(obs)

            # Wrap action for multi-agent environment (TRIAGE role)
            actions = {AgentRole.TRIAGE: single_action}
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Sum rewards across agents
            episode_reward += sum(rewards.values())
            step += 1

            # Capture compliance score if available
            comp = info.get("compliance", {}).get(AgentRole.TRIAGE, {})
            if "overall_score" in comp:
                total_compliance += comp["overall_score"]
                compliance_steps += 1

            if info.get("prevented", False):
                prevented = True
                detection_step = step
                break

            done = terminated or truncated

        total_reward += episode_reward
        if prevented:
            successful += 1
            total_detection += detection_step
        if info.get("coordination_score"):
            total_coordination += info["coordination_score"]

    return {
        "prevention_rate": successful / episodes,
        "avg_detection_step": total_detection / successful if successful > 0 else 0,
        "avg_reward": total_reward / episodes,
        "coordination_score": total_coordination / episodes,
        "compliance_score": total_compliance / compliance_steps if compliance_steps > 0 else 0,
        "episodes": episodes
    }

def get_baseline_action(obs):
    return {
        "action_type": "query_host",
        "target_id": "host-01",
        "justification": {"reasoning": "Basic investigation"}
    }

def get_trained_action(obs, model):
    # TODO: Add your trained model inference here later
    return get_baseline_action(obs)

# ====================== MAIN ======================
if __name__ == "__main__":
    print("="*70)
    print("AUTOMATIC BEFORE VS AFTER EVALUATION")
    print("="*70)

    # Run Baseline
    baseline = evaluate_model(is_trained=False, episodes=30)

    # Run Trained
    trained = evaluate_model(is_trained=True, episodes=30)

    # Show comparison
    print("\n" + "="*70)
    print("FINAL BEFORE VS AFTER RESULTS")
    print("="*70)
    print(f"{'Metric':<25} {'Before':<12} {'After':<12} {'Improvement'}")
    print("-"*70)
    print(f"Prevention Rate          {baseline['prevention_rate']:.1%}       {trained['prevention_rate']:.1%}      +{(trained['prevention_rate']-baseline['prevention_rate'])*100:+.1f}%")
    print(f"Avg Detection Step       {baseline['avg_detection_step']:.1f}          {trained['avg_detection_step']:.1f}         {trained['avg_detection_step']-baseline['avg_detection_step']:+.1f} steps")
    print(f"Avg Reward               {baseline['avg_reward']:.3f}         {trained['avg_reward']:.3f}        {trained['avg_reward']-baseline['avg_reward']:+.3f}")
    print(f"Coordination Score       {baseline['coordination_score']:.3f}         {trained['coordination_score']:.3f}        {trained['coordination_score']-baseline['coordination_score']:+.3f}")
    print("="*70)

    # Save results for dashboard
    results = {
        "baseline": baseline,
        "trained": trained,
        "timestamp": time.time()
    }
    with open("results/before_after_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("DONE: Results saved to results/before_after_results.json")
    print("You can now refresh your dashboard to see the latest comparison.")
