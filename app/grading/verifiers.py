import re
from typing import Dict, List, Tuple

class IndependentVerifiers:
    def __init__(self, ground_truth: Dict):
        self.ground_truth = ground_truth

    def verify_actual_outcome(self, action: Dict, resulting_state: Dict) -> Tuple[float, str]:
        """Did the action ACTUALLY achieve its goal?"""
        action_type = action.get("action_type")
        target = action.get("target")

        if action_type == "triage_alert":
            alert_id = target
            claimed = action.get("parameters", {}).get("classification")
            
            # Find the alert in ground truth
            gt_alert = self.ground_truth.get("alerts", {}).get(alert_id, {})
            actual = gt_alert.get("is_true_positive", False)
            
            correct = (claimed == "true_positive") == actual
            
            if correct:
                return 1.0, "Correct classification"
            else:
                return 0.0, f"WRONG: claimed {claimed}, actually {'TP' if actual else 'FP'}"
        
        elif action_type == "isolate_host":
            host_id = target
            gt_host = self.ground_truth.get("hosts", {}).get(host_id, {})
            was_compromised = gt_host.get("compromised", False)
            
            if was_compromised:
                return 1.0, "Successfully isolated compromised host"
            else:
                return -0.5, "Isolated clean host - unnecessary business disruption"
        
        elif action_type == "escalate_to_human":
            complexity = self._calculate_complexity(resulting_state)
            
            if complexity > 0.8:
                return 0.3, "Valid escalation - complex scenario"
            else:
                return -0.2, "Unnecessary escalation - agent should handle this"
        
        return 0.0, "Unknown action"

    def verify_business_impact(self, action: Dict, resulting_state: Dict) -> Tuple[float, str]:
        """Was business disruption justified?"""
        services_disrupted = self._count_disrupted_services(resulting_state)
        threat_reduced = self._calculate_threat_reduction(action, resulting_state)
        
        disruption_cost = services_disrupted * 500  # $500/service
        threat_value = threat_reduced * 10000       # $10K/threat
        
        net_value = threat_value - disruption_cost
        
        if net_value > 0:
            return min(net_value / 10000, 1.0), f"Positive ROI: ${net_value}"
        else:
            return max(net_value / 10000, -1.0), f"Negative ROI: ${net_value}"

    def verify_evidence_validity(self, justification: Dict, observable_state: Dict) -> Tuple[float, str]:
        """Does cited evidence actually exist?"""
        score = 0.0
        issues = []
        evidence_list = justification.get("evidence", [])
        
        if not evidence_list:
            return -0.5, "No evidence provided in justification"

        for evidence in evidence_list:
            source = evidence.get("source", "")
            
            if source.startswith("ALT-"):
                if not any(a.get("alert_id") == source for a in observable_state.get("alerts", [])):
                    issues.append(f"{source} is FABRICATED (non-existent alert)")
                    score -= 1.0  # Stronger penalty for fabrication
                else:
                    score += 0.2
            
            elif source.startswith("HOST-") or source.startswith("ci-"):
                # Also handling standard host names or IDs
                if not any(h.get("host_id") == source for h in observable_state.get("hosts", [])):
                    issues.append(f"{source} is FABRICATED (non-existent host)")
                    score -= 1.0
                elif source not in observable_state.get("query_history", []):
                    issues.append(f"{source} cited but NEVER QUERIED - agent is hallucinating internal data")
                    score -= 0.8  # Stronger penalty for unqueried hosts
                else:
                    score += 0.2
        
        return max(-1.0, min(1.0, score)), " | ".join(issues) if issues else "Valid evidence"

    def verify_action_sequence_logic(self, action_history: List[Dict]) -> Tuple[float, str]:
        """Does action sequence make sense?"""
        if len(action_history) > 5:
            action_types = [a.get("action_type") for a in action_history]
            if len(set(action_types)) == 1:
                return 0.0, "Repetitive - using same action repeatedly"
        
        for i, action in enumerate(action_history):
            if action.get("action_type") == "remove_persistence":
                prior_contains = [a for a in action_history[:i] 
                                if a.get("action_type") in ["isolate_host", "block_ip"]]
                if not prior_contains:
                    return -0.5, "Invalid: eradicating before containment"
        
        return 1.0, "Logical sequence"

    def verify_justification_uniqueness(self, current: str, history: List[str]) -> Tuple[float, str]:
        """Is agent copy-pasting justifications?"""
        if not history:
            return 1.0, "First action"
        
        similarities = [self._jaccard_similarity(current, prev) for prev in history[-5:]]
        max_similarity = max(similarities) if similarities else 0
        
        if max_similarity > 0.9:
            return 0.0, "Copy-paste detected - 90%+ identical"
        elif max_similarity > 0.7:
            return 0.3, "Too similar to previous justification"
        else:
            return 1.0, "Unique"

    def verify_justification_structure(self, justification: Dict) -> Tuple[float, str]:
        """Check if justification has proper length and required fields."""
        reasoning = justification.get("reasoning", "")
        if len(reasoning) < 50:
            return -0.3, "Justification too short - lacks depth"
        
        required_fields = ["reasoning", "evidence", "risk_assessment"]
        missing = [f for f in required_fields if f not in justification]
        
        if missing:
            return -0.4, f"Missing required fields: {', '.join(missing)}"
            
        return 1.0, "Properly structured justification"

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with", "as", "by", "at", "this", "that", "it"}
        
        def get_words(t):
            words = set(re.findall(r'\w+', t.lower()))
            return words - stop_words
            
        words1 = get_words(text1)
        words2 = get_words(text2)
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def verify_specificity(self, justification: Dict, action: Dict) -> Tuple[float, str]:
        """Does justification cite specific details?"""
        reasoning = justification.get("reasoning", "")
        
        alert_ids = re.findall(r'ALT-\d+', reasoning)
        host_ids = re.findall(r'HOST-\d+', reasoning) + re.findall(r'ci-runner-\d+', reasoning)
        ip_addresses = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', reasoning)
        mitre_ids = re.findall(r'T\d{4}', reasoning)
        
        specificity_score = 0.0
        if alert_ids: specificity_score += 0.25
        if host_ids: specificity_score += 0.25
        if ip_addresses: specificity_score += 0.25
        if mitre_ids: specificity_score += 0.25
        
        generic_phrases = [
            "potential compromise detected",
            "suspicious activity observed",
            "requires further investigation"
        ]
        generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in reasoning.lower())
        if generic_count > 2:
            specificity_score *= 0.5
        
        if specificity_score < 0.3:
            return 0.0, "Generic - lacks specific IDs/IPs"
        else:
            return specificity_score, f"Cited {len(alert_ids)} alerts, {len(host_ids)} hosts"

    def verify_consistency(self, justification: Dict, action: Dict) -> Tuple[float, str]:
        """Does risk assessment match action severity?"""
        threat_level = justification.get("risk_assessment", {}).get("threat_level", "UNKNOWN")
        
        severity_map = {
            "CRITICAL": ["isolate_host", "block_ip", "disable_account"],
            "HIGH": ["isolate_host", "block_ip", "triage_alert"],
            "MEDIUM": ["query_host", "triage_alert"],
            "LOW": ["query_host"],
        }
        
        expected_actions = severity_map.get(threat_level, [])
        action_type = action.get("action_type")
        
        if action_type in expected_actions:
            return 1.0, "Action matches threat level"
        elif threat_level == "CRITICAL" and action_type == "escalate_to_human":
            return 0.5, "Critical but only escalating"
        elif threat_level == "LOW" and action_type == "isolate_host":
            return -0.5, "Over-reaction - isolating for LOW threat"
        else:
            return 0.3, "Inconsistent threat level vs action"

    def _calculate_complexity(self, state: Dict) -> float:
        # Mock complexity calculation
        alerts = len(state.get("alerts", []))
        return min(alerts / 10.0, 1.0)

    def _count_disrupted_services(self, state: Dict) -> int:
        isolated = [h for h in state.get("hosts", []) if h.get("status") == "isolated"]
        return len(isolated)

    def _calculate_threat_reduction(self, action: Dict, state: Dict) -> int:
        if action.get("action_type") == "isolate_host":
            return 1
        return 0
