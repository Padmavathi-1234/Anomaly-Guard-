"""
Adversarial Tester
====================
Implements specific adversarial tests for agent robustness:
    1. Distraction Test — Many benign alerts obscuring real threats
    2. False Positive Test — All alerts are benign, agent should do nothing
    3. Subtle Attack Test — Low-confidence, hard-to-detect threats
    4. Multi-Stage Test — Attack progresses through multiple phases
    5. Evasion Test — Attacker actively evades detection patterns
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from app.models import (
    Action, ActionJustification, EvidenceItem, RiskAssessment, AlternativeConsidered,
)


class AdversarialTester:
    """
    Runs adversarial evaluation suites against trained agents.
    
    Each test creates a specific challenging scenario, runs the agent,
    and evaluates performance against adversarial criteria.
    """

    def __init__(self, environment):
        """
        Args:
            environment: AnomalyGuardEnvironment instance
        """
        self.env = environment
        self.tests = [
            ("distraction", self._test_distraction),
            ("false_positives", self._test_false_positives),
            ("subtle_attack", self._test_subtle_attack),
            ("multi_stage", self._test_multi_stage),
            ("evasion", self._test_evasion),
        ]

    def run_suite(self, agent_fn) -> Dict[str, Any]:
        """
        Run all adversarial tests against an agent function.

        Args:
            agent_fn: Callable(observation) -> Action that represents the agent

        Returns:
            Dict with per-test results and aggregate scores
        """
        results = {}
        total_score = 0.0

        for name, test_fn in self.tests:
            result = test_fn(agent_fn)
            results[name] = result
            total_score += result.get("score", 0.0)

        results["aggregate"] = {
            "total_score": round(total_score, 4),
            "avg_score": round(total_score / max(len(self.tests), 1), 4),
            "tests_passed": sum(1 for r in results.values() if isinstance(r, dict) and r.get("passed", False)),
            "tests_total": len(self.tests),
        }

        return results

    def _test_distraction(self, agent_fn) -> Dict[str, Any]:
        """
        Test with many distracting but benign alerts.
        
        Scenario: 2 real threats hidden among 15+ false positives.
        Success: Agent finds both real threats without acting on FPs.
        """
        obs, _ = self.env.reset(task_id=1, seed=99901)

        # Count true threats in scenario
        state = self.env.state()
        real_threats = [a for a in state["alerts"] if a.is_true_positive]
        total_alerts = len(state["alerts"])

        correct_triages = 0
        false_actions = 0
        steps = 0
        max_steps = obs.max_steps

        while steps < max_steps:
            try:
                action = agent_fn(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1

                if info.get("action_result", {}).get("correctness", 0) >= 0.8:
                    correct_triages += 1
                elif info.get("action_result", {}).get("harmful", False):
                    false_actions += 1

                if terminated or truncated:
                    break
            except Exception:
                break

        precision = correct_triages / max(correct_triages + false_actions, 1)
        recall = correct_triages / max(len(real_threats), 1)
        score = (precision * 0.6 + recall * 0.4) if (precision + recall) > 0 else 0.0

        return {
            "test": "distraction",
            "description": "Many false positives obscuring real threats",
            "total_alerts": total_alerts,
            "real_threats": len(real_threats),
            "correct_triages": correct_triages,
            "false_actions": false_actions,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "score": round(score, 4),
            "passed": score >= 0.6,
            "steps_used": steps,
        }

    def _test_false_positives(self, agent_fn) -> Dict[str, Any]:
        """
        Test with ALL benign alerts — agent should triage as FP or escalate.
        
        Success: Agent correctly classifies most as false positives.
        Failure: Agent takes containment actions on benign alerts.
        """
        obs, _ = self.env.reset(task_id=1, seed=99902)

        containment_actions_taken = 0
        correct_fp_triages = 0
        steps = 0

        while steps < obs.max_steps:
            try:
                action = agent_fn(obs)
                action_type = action.action_type if isinstance(action.action_type, str) else action.action_type.value
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1

                if action_type in ("isolate_host", "block_ip", "disable_account"):
                    containment_actions_taken += 1
                elif action_type == "triage_alert":
                    if info.get("action_result", {}).get("correctness", 0) >= 0.8:
                        correct_fp_triages += 1

                if terminated or truncated:
                    break
            except Exception:
                break

        # Penalize containment on benign network
        penalty = min(1.0, containment_actions_taken * 0.25)
        score = max(0.0, 1.0 - penalty)

        return {
            "test": "false_positives",
            "description": "All benign alerts — test for overreaction",
            "containment_actions": containment_actions_taken,
            "correct_fp_triages": correct_fp_triages,
            "score": round(score, 4),
            "passed": containment_actions_taken == 0,
            "steps_used": steps,
        }

    def _test_subtle_attack(self, agent_fn) -> Dict[str, Any]:
        """
        Test with low-confidence, hard-to-detect threats.
        
        All threat alerts have confidence < 0.5.
        Success: Agent still detects and contains threats despite low confidence.
        """
        obs, _ = self.env.reset(task_id=2, seed=99903)

        threats_found = 0
        hosts_contained = 0
        steps = 0

        while steps < obs.max_steps:
            try:
                action = agent_fn(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1

                correctness = info.get("action_result", {}).get("correctness", 0)
                action_type = action.action_type if isinstance(action.action_type, str) else action.action_type.value

                if action_type == "triage_alert" and correctness >= 0.8:
                    threats_found += 1
                elif action_type == "isolate_host" and correctness >= 0.7:
                    hosts_contained += 1

                if terminated or truncated:
                    break
            except Exception:
                break

        state = self.env.state()
        total_threats = sum(1 for a in state["alerts"] if a.is_true_positive)
        total_compromised = sum(1 for h in state["hosts"] if h.c2_active or h.persistence)
        
        detection_rate = threats_found / max(total_threats, 1)
        containment_rate = hosts_contained / max(total_compromised, 1)
        score = detection_rate * 0.5 + containment_rate * 0.5

        return {
            "test": "subtle_attack",
            "description": "Low-confidence threats requiring investigation",
            "total_threats": total_threats,
            "threats_found": threats_found,
            "detection_rate": round(detection_rate, 3),
            "hosts_contained": hosts_contained,
            "containment_rate": round(containment_rate, 3),
            "score": round(score, 4),
            "passed": score >= 0.5,
            "steps_used": steps,
        }

    def _test_multi_stage(self, agent_fn) -> Dict[str, Any]:
        """
        Test with a multi-stage attack requiring full IR lifecycle.
        
        Success: Agent progresses through detection → containment →
                 eradication → recovery phases correctly.
        """
        obs, _ = self.env.reset(task_id=3, seed=99904)

        phases_reached = set()
        steps = 0

        while steps < obs.max_steps:
            try:
                action = agent_fn(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1

                phase = info.get("phase", "detection")
                phases_reached.add(phase)

                if terminated or truncated:
                    break
            except Exception:
                break

        expected_phases = {"detection", "containment", "eradication", "recovery", "completed"}
        phase_coverage = len(phases_reached & expected_phases) / len(expected_phases)
        completed = "completed" in phases_reached
        score = phase_coverage * 0.7 + (0.3 if completed else 0.0)

        return {
            "test": "multi_stage",
            "description": "Full IR lifecycle requiring all phases",
            "phases_reached": sorted(phases_reached),
            "phase_coverage": round(phase_coverage, 3),
            "completed": completed,
            "score": round(score, 4),
            "passed": completed,
            "steps_used": steps,
        }

    def _test_evasion(self, agent_fn) -> Dict[str, Any]:
        """
        Test against evasive attack patterns.
        
        High-difficulty scenario where observables are minimal.
        Success: Agent uses query_host to investigate before acting.
        """
        obs, _ = self.env.reset(task_id=2, seed=99905)

        query_actions = 0
        premature_containment = 0
        steps = 0

        while steps < obs.max_steps:
            try:
                action = agent_fn(obs)
                action_type = action.action_type if isinstance(action.action_type, str) else action.action_type.value
                obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1

                if action_type == "query_host":
                    query_actions += 1
                elif action_type in ("isolate_host", "block_ip") and query_actions == 0:
                    premature_containment += 1

                if terminated or truncated:
                    break
            except Exception:
                break

        # Score rewards investigation before action
        investigation_score = min(1.0, query_actions / max(3, 1))
        premature_penalty = min(1.0, premature_containment * 0.3)
        score = max(0.0, investigation_score * 0.7 - premature_penalty)

        return {
            "test": "evasion",
            "description": "Evasive attack requiring investigation-first approach",
            "query_actions": query_actions,
            "premature_containment": premature_containment,
            "investigation_score": round(investigation_score, 3),
            "score": round(score, 4),
            "passed": query_actions >= 2 and premature_containment == 0,
            "steps_used": steps,
        }
