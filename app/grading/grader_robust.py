from typing import Dict, List
from app.grading.verifiers import IndependentVerifiers

class RobustGrader:
    def __init__(self):
        self.verifiers = None
        self.suspicion_flags = []
        self.justification_history = []
    
    def grade_action(self, action: Dict, justification: Dict, observable_state: Dict, 
                    resulting_state: Dict, ground_truth: Dict, action_history: List[Dict]) -> Dict:
        
        if self.verifiers is None:
            self.verifiers = IndependentVerifiers(ground_truth)
        else:
            # Update ground truth in case it changed
            self.verifiers.ground_truth = ground_truth
            
        # Run all verifiers
        v1_score, v1_msg = self.verifiers.verify_actual_outcome(action, resulting_state)
        v2_score, v2_msg = self.verifiers.verify_business_impact(action, resulting_state)
        v3_score, v3_msg = self.verifiers.verify_evidence_validity(justification, observable_state)
        v4_score, v4_msg = self.verifiers.verify_action_sequence_logic(action_history)
        v5_score, v5_msg = self.verifiers.verify_justification_uniqueness(
            justification.get("reasoning", ""), self.justification_history
        )
        v6_score, v6_msg = self.verifiers.verify_specificity(justification, action)
        v7_score, v7_msg = self.verifiers.verify_consistency(justification, action)
        v8_score, v8_msg = self.verifiers.verify_justification_structure(justification)
        
        # Store for next check
        self.justification_history.append(justification.get("reasoning", ""))
        
        # Detect cheating and suspicious patterns
        cheating_detected = False
        
        if v3_score < 0:  # Fabricated evidence
            cheating_detected = True
            self.suspicion_flags.append("Fabricated evidence")
        
        if v5_score == 0:  # Copy-paste
            cheating_detected = True
            self.suspicion_flags.append("Copy-paste justification")
            
        if v8_score < 0:  # Poor structure
            self.suspicion_flags.append("Poor justification structure")

        # Add "over_escalation" and "isolation_spam" as suspicion flags
        if action.get("action_type") == "escalate_to_human" and v1_score < 0:
            self.suspicion_flags.append("over_escalation")
            
        if action.get("action_type") == "isolate_host" and v1_score < 0:
            # Check if this is the Nth bad isolation
            bad_isolations = [f for f in self.suspicion_flags if f == "unnecessary_isolation"]
            self.suspicion_flags.append("unnecessary_isolation")
            if len(bad_isolations) >= 2:
                self.suspicion_flags.append("isolation_spam")
                cheating_detected = True

        if v4_score < 0:  # Invalid sequence
            cheating_detected = True
            self.suspicion_flags.append("Invalid action sequence")
        
        # Calculate final score
        if cheating_detected:
            # Severe penalty
            final_score = -0.5
        else:
            # Weighted average
            final_score = (
                v1_score * 0.30 +  # Actual outcome
                v2_score * 0.15 +  # Business impact
                v3_score * 0.15 +  # Evidence validity
                v4_score * 0.10 +  # Sequence logic
                v5_score * 0.05 +  # Uniqueness
                v6_score * 0.10 +  # Specificity
                v7_score * 0.05 +  # Consistency
                v8_score * 0.10    # Structure
            )
        
        return {
            "final_score": final_score,
            "cheating_detected": cheating_detected,
            "verifier_breakdown": {
                "actual_outcome": {"score": v1_score, "message": v1_msg},
                "business_impact": {"score": v2_score, "message": v2_msg},
                "evidence_validity": {"score": v3_score, "message": v3_msg},
                "sequence_logic": {"score": v4_score, "message": v4_msg},
                "uniqueness": {"score": v5_score, "message": v5_msg},
                "specificity": {"score": v6_score, "message": v6_msg},
                "consistency": {"score": v7_score, "message": v7_msg},
                "structure": {"score": v8_score, "message": v8_msg}
            },
            "suspicion_flags": list(set(self.suspicion_flags[-10:]))
        }
