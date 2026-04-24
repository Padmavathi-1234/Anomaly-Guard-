from typing import Dict

class AdversarialAttacker:
    def __init__(self):
        self.successful_tactics = []
        self.failed_tactics = []
        self.defender_weaknesses = []
    
    def generate_attack(self, defender_performance: Dict) -> Dict:
        """Generate attack tailored to exploit defender gaps"""
        
        # Analyze defender weaknesses
        if defender_performance.get("detection_speed", 0) > 10:
            self.defender_weaknesses.append("slow_detection")
        
        if defender_performance.get("false_negative_rate", 0) > 0.3:
            self.defender_weaknesses.append("misses_low_confidence_alerts")
        
        # Generate attack exploiting weaknesses
        if "slow_detection" in self.defender_weaknesses:
            # Use fast-moving attack (exfil within 5 steps)
            attack_type = "rapid_exfiltration"
        elif "misses_low_confidence_alerts" in self.defender_weaknesses:
            # Use stealthy attack with low-confidence alerts
            attack_type = "low_and_slow"
        else:
            # Default attack
            attack_type = "standard"
        
        return self._create_attack_scenario(attack_type)
    
    def learn_from_episode(self, episode_result: Dict):
        """Update strategy based on outcome"""
        
        if episode_result.get("attacker_won", False):
            self.successful_tactics.append({
                "attack_type": episode_result.get("attack_type"),
                "why_worked": episode_result.get("defender_mistake"),
                "detection_step": episode_result.get("detection_step", 999)
            })
        else:
            self.failed_tactics.append({
                "attack_type": episode_result.get("attack_type"),
                "caught_at_step": episode_result.get("detection_step")
            })

    def _create_attack_scenario(self, attack_type: str) -> Dict:
        from app.scenarios.scenario_base import generate_basic_scenario
        # Use a consistent seed for base scenario and then mutate it
        scenario = generate_basic_scenario(task_id=99, seed=42)
        
        if attack_type == "rapid_exfiltration":
            scenario["name"] = "Rapid Data Exfiltration"
            # Increase difficulty and reduce max steps
            scenario["max_steps"] = 8
            # Add specific high-severity alert for exfiltration
            exfil_alert_id = "ALT-99999"
            scenario["initial_state"]["alerts"].append({
                "alert_id": exfil_alert_id,
                "alert_type": "Exfiltration",
                "severity": "critical",
                "confidence": 0.88,
                "description": "Rapid large outbound transfer detected",
                "source_host": "HOST-001",
                "timestamp": "2026-04-23T02:00:00Z"
            })
            scenario["ground_truth"]["alerts"][exfil_alert_id] = {
                "is_true_positive": True,
                "mitre_technique": "T1567"
            }
            return scenario
        
        elif attack_type == "low_and_slow":
            scenario["name"] = "Stealthy Persistence"
            scenario["max_steps"] = 30
            # Lower confidence of all alerts to trick defender
            for alert in scenario["initial_state"]["alerts"]:
                alert["confidence"] = round(alert["confidence"] * 0.5, 2)
                alert["severity"] = "low"
            return scenario
        
        scenario["name"] = "Standard Attack"
        return scenario
