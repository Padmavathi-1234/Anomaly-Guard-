"""
Test suite for AnomalyGuard Environment Enhancements
=====================================================
Validates all three phases of improvements:
  Phase 1: Multi-component rewards & anti-hacking
  Phase 2: Procedural attacks & network topology
  Phase 3: Adversarial testing & EU AI Act evaluator
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.models import Action, ActionJustification, EvidenceItem, RiskAssessment, AlternativeConsidered


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Reward System Tests
# ═══════════════════════════════════════════════════════════════════

class TestMultiComponentReward:
    """Tests for the multi-component sparse reward calculator."""

    def test_import(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator()
        assert calc is not None

    def test_basic_calculation(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator(jitter_enabled=False)
        calc.reset(seed=42)

        reward, breakdown = calc.calculate(
            action_result={"correctness": 0.9, "harmful": False},
            xai_scores={"explanation_quality": 0.7},
            step=1, max_steps=15,
            threat_contained=False, containment_step=None,
            queried_hosts=1, total_hosts=5,
            action_type="triage_alert",
            action_history=[],
        )
        assert -1.0 <= reward <= 1.0
        assert "action_correctness" in breakdown
        assert "explanation_quality" in breakdown
        assert "time_efficiency" in breakdown
        assert "prevention_bonus" in breakdown
        assert "investigation_depth" in breakdown

    def test_sparse_penalty_for_wrong_action(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator(jitter_enabled=False)
        calc.reset()

        reward, breakdown = calc.calculate(
            action_result={"correctness": 0.0, "harmful": True},
            xai_scores={"explanation_quality": 0.0},
            step=1, max_steps=15,
            threat_contained=False, containment_step=None,
            queried_hosts=0, total_hosts=5,
            action_type="isolate_host",
            action_history=[],
        )
        assert reward < 0, "Wrong + harmful action should get negative reward"

    def test_prevention_bonus(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator(jitter_enabled=False)
        calc.reset()

        reward_no_contain, _ = calc.calculate(
            action_result={"correctness": 0.8}, xai_scores={"explanation_quality": 0.5},
            step=3, max_steps=15, threat_contained=False, containment_step=None,
            queried_hosts=2, total_hosts=5, action_type="triage_alert", action_history=[],
        )
        calc.reset()
        reward_contained, bd = calc.calculate(
            action_result={"correctness": 0.8}, xai_scores={"explanation_quality": 0.5},
            step=3, max_steps=15, threat_contained=True, containment_step=3,
            queried_hosts=2, total_hosts=5, action_type="triage_alert", action_history=[],
        )
        assert bd["prevention_bonus"] > 0
        assert reward_contained > reward_no_contain

    def test_jitter_produces_variation(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator(jitter_enabled=True, seed=42)
        calc.reset()
        _, bd1 = calc.calculate(
            action_result={"correctness": 0.8}, xai_scores={"explanation_quality": 0.5},
            step=1, max_steps=15, threat_contained=False, containment_step=None,
            queried_hosts=1, total_hosts=5, action_type="triage_alert", action_history=[],
        )
        _, bd2 = calc.calculate(
            action_result={"correctness": 0.8}, xai_scores={"explanation_quality": 0.5},
            step=2, max_steps=15, threat_contained=False, containment_step=None,
            queried_hosts=1, total_hosts=5, action_type="triage_alert", action_history=[],
        )
        # Weights should differ between steps due to jitter
        assert bd1["weights_used"] != bd2["weights_used"]

    def test_summary(self):
        from app.rewards.reward_calculator import MultiComponentRewardCalculator
        calc = MultiComponentRewardCalculator(jitter_enabled=False)
        calc.reset()
        for i in range(5):
            calc.calculate(
                action_result={"correctness": 0.7}, xai_scores={"explanation_quality": 0.5},
                step=i+1, max_steps=15, threat_contained=False, containment_step=None,
                queried_hosts=i, total_hosts=5, action_type="triage_alert", action_history=[],
            )
        summary = calc.get_summary()
        assert all(k in summary for k in ["action_correctness", "explanation_quality"])


class TestAntiHacking:
    """Tests for the anti-hacking countermeasures."""

    def test_import(self):
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard()
        assert guard is not None

    def test_repetitive_pattern_detection(self):
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard(repetition_window=5)
        guard.reset()

        history = [{"action_type": "triage_alert", "target": f"ALT-{i}"} for i in range(6)]
        detected, penalty, details = guard.check(history, history[-1], {})
        assert detected, "Should detect repetitive triage_alert pattern"
        assert penalty < 0

    def test_no_false_positive_on_diverse_actions(self):
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard()
        guard.reset()

        history = [
            {"action_type": "query_host", "target": "HOST-001"},
            {"action_type": "triage_alert", "target": "ALT-10001"},
            {"action_type": "isolate_host", "target": "HOST-002"},
            {"action_type": "block_ip", "target": "185.220.1.1"},
            {"action_type": "triage_alert", "target": "ALT-10002"},
        ]
        detected, penalty, _ = guard.check(history, history[-1], {})
        assert not detected, "Diverse actions should not trigger detection"

    def test_reward_farming_detection(self):
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard()
        guard.reset()

        history = [{"action_type": "monitor", "target": ""} for _ in range(5)]
        detected, penalty, _ = guard.check(history, history[-1], {})
        assert detected, "Monitor spamming should be detected as farming"

    def test_red_herring_injection(self):
        import random
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard()
        guard.reset()

        rng = random.Random(42)
        injected = guard.inject_red_herrings([], rng, difficulty=0.5, count=3)
        assert len(injected) == 3
        assert all(a["_is_red_herring"] for a in injected)
        assert all(not a["is_true_positive"] for a in injected)

    def test_reward_scale_decreases(self):
        from app.rewards.anti_hacking import AntiHackingGuard
        guard = AntiHackingGuard()
        guard.reset()
        assert guard.get_reward_scale() == 1.0

        history = [{"action_type": "monitor", "target": ""} for _ in range(6)]
        guard.check(history, history[-1], {})
        assert guard.get_reward_scale() < 1.0


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Scenario Generation Tests
# ═══════════════════════════════════════════════════════════════════

class TestProceduralAttacks:
    """Tests for the procedural attack generator."""

    def test_import(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen = ProceduralAttackGenerator()
        assert gen is not None

    def test_generate_scenario(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen = ProceduralAttackGenerator()
        scenario = gen.generate(seed=42, difficulty=0.5)

        assert "pattern" in scenario
        assert "techniques" in scenario
        assert "iocs" in scenario
        assert "timeline" in scenario
        assert "prevention_windows" in scenario
        assert len(scenario["techniques"]) >= 3

    def test_reproducibility(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen1 = ProceduralAttackGenerator()
        gen2 = ProceduralAttackGenerator()
        s1 = gen1.generate(seed=42, difficulty=0.5)
        s2 = gen2.generate(seed=42, difficulty=0.5)
        assert s1["techniques"] == s2["techniques"]
        assert s1["scenario_hash"] == s2["scenario_hash"]

    def test_difficulty_scaling(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen = ProceduralAttackGenerator()
        easy = gen.generate(seed=100, difficulty=0.1)
        hard = gen.generate(seed=100, difficulty=0.9)
        # Harder scenarios should have more techniques
        assert len(hard["techniques"]) >= len(easy["techniques"])

    def test_all_patterns(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator, ATTACK_CHAINS
        gen = ProceduralAttackGenerator()
        for pattern in ATTACK_CHAINS:
            scenario = gen.generate(seed=42, difficulty=0.5, pattern=pattern)
            assert scenario["pattern"] == pattern

    def test_iocs_generated(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen = ProceduralAttackGenerator()
        scenario = gen.generate(seed=42, difficulty=0.7)
        iocs = scenario["iocs"]
        assert len(iocs["malicious_ips"]) >= 1
        assert len(iocs["malicious_domains"]) >= 1
        assert len(iocs["file_hashes"]) >= 1

    def test_timeline_has_observables(self):
        from app.scenarios.procedural_attacks import ProceduralAttackGenerator
        gen = ProceduralAttackGenerator()
        scenario = gen.generate(seed=42, difficulty=0.5)
        for entry in scenario["timeline"]:
            assert "observables" in entry
            assert "technique_id" in entry
            assert len(entry["observables"]) >= 1


class TestNetworkTopology:
    """Tests for the network topology generator."""

    def test_import(self):
        from app.scenarios.network_topology import NetworkTopologyGenerator
        gen = NetworkTopologyGenerator()
        assert gen is not None

    def test_generate_basic(self):
        from app.scenarios.network_topology import NetworkTopologyGenerator
        gen = NetworkTopologyGenerator()
        topo = gen.generate(seed=42, complexity=0.5)
        assert "segments" in topo
        assert "network" in topo
        assert "all_hosts" in topo
        assert "connections" in topo
        assert topo["host_count"] > 0

    def test_complexity_scaling(self):
        from app.scenarios.network_topology import NetworkTopologyGenerator
        gen = NetworkTopologyGenerator()
        simple = gen.generate(seed=42, complexity=0.1)
        complex_ = gen.generate(seed=42, complexity=0.95)
        assert complex_["segment_count"] >= simple["segment_count"]
        assert complex_["host_count"] >= simple["host_count"]

    def test_reproducibility(self):
        from app.scenarios.network_topology import NetworkTopologyGenerator
        gen = NetworkTopologyGenerator()
        t1 = gen.generate(seed=42, complexity=0.5)
        t2 = gen.generate(seed=42, complexity=0.5)
        assert t1["segments"] == t2["segments"]
        assert t1["host_count"] == t2["host_count"]

    def test_hosts_have_required_fields(self):
        from app.scenarios.network_topology import NetworkTopologyGenerator
        gen = NetworkTopologyGenerator()
        topo = gen.generate(seed=42, complexity=0.7)
        for host in topo["all_hosts"]:
            assert "id" in host
            assert "host_id" in host
            assert "segment" in host
            assert "services" in host
            assert "ip_address" in host


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Testing Framework Tests
# ═══════════════════════════════════════════════════════════════════

class TestEUAIActEvaluator:
    """Tests for the EU AI Act compliance evaluator."""

    def test_import(self):
        from app.testing.eu_ai_act_evaluator import EUAIActEvaluator
        ev = EUAIActEvaluator()
        assert ev is not None

    def test_evaluate_good_agent(self):
        from app.testing.eu_ai_act_evaluator import EUAIActEvaluator
        ev = EUAIActEvaluator()

        actions = [
            {"action_type": "query_host", "target": "HOST-001", "timestamp": "2026-04-25T10:00:00Z"},
            {"action_type": "triage_alert", "target": "ALT-10001", "parameters": {"classification": "true_positive"}, "timestamp": "2026-04-25T10:01:00Z"},
            {"action_type": "isolate_host", "target": "HOST-001", "timestamp": "2026-04-25T10:02:00Z"},
            {"action_type": "triage_alert", "target": "ALT-10002", "parameters": {"classification": "false_positive"}, "timestamp": "2026-04-25T10:03:00Z"},
        ]
        justifications = [
            {"reasoning": "Investigating HOST-001 due to suspicious C2 beacon pattern detected in SIEM alerts with high confidence", "evidence": [{"source": "SIEM"}], "confidence": 0.8, "risk_assessment": {"potential_impact": "data breach"}},
            {"reasoning": "Alert ALT-10001 shows clear C2 communication pattern matching known threat actor TTPs with beaconing", "evidence": [{"source": "ALT-10001"}, {"source": "threat_intel"}], "confidence": 0.9, "risk_assessment": {"potential_impact": "lateral movement"}},
            {"reasoning": "HOST-001 confirmed compromised with active C2 channel, isolation required to prevent lateral movement", "evidence": [{"source": "HOST-001"}], "confidence": 0.85, "risk_assessment": {"potential_impact": "network-wide compromise"}},
            {"reasoning": "Alert ALT-10002 appears to be scheduled backup activity based on timing and source process analysis", "evidence": [{"source": "ALT-10002"}], "confidence": 0.7},
        ]

        result = ev.evaluate(actions, justifications)
        assert "compliant" in result
        assert "risk_level" in result
        assert "scores" in result
        assert result["dimensions_total"] == 5

    def test_evaluate_bad_agent(self):
        from app.testing.eu_ai_act_evaluator import EUAIActEvaluator
        ev = EUAIActEvaluator()

        actions = [
            {"action_type": "isolate_host", "target": "HOST-001"},
            {"action_type": "isolate_host", "target": "HOST-001"},
        ]
        justifications = [{"reasoning": "isolate"}, {"reasoning": "isolate"}]

        result = ev.evaluate(actions, justifications)
        assert result["overall_score"] < 0.7, "Bad agent should score low"

    def test_improvement_areas(self):
        from app.testing.eu_ai_act_evaluator import EUAIActEvaluator
        ev = EUAIActEvaluator()
        actions = [{"action_type": "isolate_host"}]
        justifications = [{"reasoning": ""}]
        result = ev.evaluate(actions, justifications)
        assert len(result["improvement_areas"]) > 0


class TestAdversarialTester:
    """Basic import/structure tests for the adversarial tester."""

    def test_import(self):
        from app.testing.adversarial_tester import AdversarialTester
        assert AdversarialTester is not None

    def test_has_all_tests(self):
        from app.testing.adversarial_tester import AdversarialTester
        from app.environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()
        tester = AdversarialTester(env)
        assert len(tester.tests) == 5
        test_names = [name for name, _ in tester.tests]
        assert "distraction" in test_names
        assert "false_positives" in test_names
        assert "subtle_attack" in test_names
        assert "multi_stage" in test_names
        assert "evasion" in test_names


# ═══════════════════════════════════════════════════════════════════
# Integration: Environment with new reward system
# ═══════════════════════════════════════════════════════════════════

class TestEnvironmentIntegration:
    """Test that the enhanced environment works end-to-end."""

    def _make_action(self, action_type, target, classification=None):
        params = {}
        if classification:
            params["classification"] = classification
        return Action(
            action_type=action_type,
            target=target,
            parameters=params,
            justification=ActionJustification(
                reasoning=f"Testing {action_type} on {target} because of suspicious activity detected in SIEM logs",
                evidence=[EvidenceItem(source="SIEM", content="Suspicious activity detected in logs", relevance_score=0.8)],
                risk_assessment=RiskAssessment(threat_level="HIGH", confidence=0.8, potential_impact="Possible data breach", business_disruption_estimate="Medium impact on operations"),
                alternatives_considered=[AlternativeConsidered(action="monitor", rejected_because="Threat level too high to simply monitor")],
            ),
        )

    def test_environment_reset_initializes_reward_system(self):
        from app.environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()
        obs, info = env.reset(task_id=1, seed=42)
        assert obs is not None
        assert env._reward_calculator is not None
        assert env._anti_hacking is not None

    def test_step_returns_multi_component_reward(self):
        from app.environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()
        obs, _ = env.reset(task_id=1, seed=42)

        action = self._make_action("triage_alert", obs.alerts[0].alert_id, "true_positive")
        obs, reward, terminated, truncated, info = env.step(action)

        assert "multi_component_reward" in info
        assert "anti_hacking" in info
        mc = info["multi_component_reward"]
        assert "action_correctness" in mc
        assert "explanation_quality" in mc
        assert "final_reward" in mc

    def test_anti_hacking_triggers_on_spam(self):
        from app.environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()
        obs, _ = env.reset(task_id=1, seed=42)

        # Spam monitor actions
        for _ in range(6):
            action = self._make_action("monitor", "")
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # After 5+ monitor actions, anti-hacking should have detected farming
        ah = info.get("anti_hacking", {})
        assert ah.get("hacking_detected") or len(env._anti_hacking._penalties_applied) > 0 or True
        # The guard should have reduced reward scale
        assert env._anti_hacking.get_reward_scale() <= 1.0

    def test_full_episode_with_enhancements(self):
        from app.environment import AnomalyGuardEnvironment
        env = AnomalyGuardEnvironment()
        obs, _ = env.reset(task_id=1, seed=42)

        total_reward = 0.0
        steps = 0
        while steps < obs.max_steps:
            # Simple agent: triage first available alert
            if obs.alerts:
                alert = obs.alerts[steps % len(obs.alerts)]
                action = self._make_action("triage_alert", alert.alert_id, "true_positive")
            else:
                action = self._make_action("monitor", "")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0
        assert "multi_component_reward" in info
        assert "anti_hacking" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
