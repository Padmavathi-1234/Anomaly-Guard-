"""
Multi-Agent Anomaly Guard Environment
=======================================
Extends AnomalyGuardBase with:
- Multi-agent roles (Triage, Containment, Threat Hunter, Forensics)
- Role-based permissions and filtered observations
- Adversarial attacker integration
- Self-evolving curriculum manager (8 levels, long-horizon)
- EU AI Act compliance scoring on every action
- Coordination bonuses for inter-agent synergy
"""

import copy
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
from app.core.environment_base import AnomalyGuardBase
from app.agents.adversarial_attacker import AdversarialAttacker
from app.core.curriculum_manager import CurriculumManager
from app.compliance.eu_ai_act_engine import EUAIActComplianceEngine
from app.agents.multi_agent_coordinator import CoordinationTracker


class AgentRole(Enum):
    TRIAGE = "triage"
    CONTAINMENT = "containment"
    THREAT_HUNTER = "threat_hunter"
    FORENSICS = "forensics"


class MultiAgentAnomalyGuard(AnomalyGuardBase):
    """
    Production multi-agent cybersecurity environment with:
    - Adaptive curriculum (auto-adjusts difficulty based on performance)
    - Long-horizon mode (20–80 steps depending on level)
    - EU AI Act compliance scoring on every action
    - Adversarial attacker support
    """

    def __init__(
        self,
        use_adversarial: bool = False,
        use_realistic: bool = False,
        use_live_intel: bool = False,
        curriculum_start_level: int = 1,
    ):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.use_realistic = use_realistic
        self.use_live_intel = use_live_intel

        # Adversarial attacker
        self.attacker: Optional[AdversarialAttacker] = None
        if use_adversarial:
            self.attacker = AdversarialAttacker()

        # Curriculum manager (self-evolving difficulty)
        self.curriculum = CurriculumManager(
            window_size=100, start_level=curriculum_start_level
        )

        # EU AI Act compliance engine
        self.compliance_engine = EUAIActComplianceEngine()

        # Advanced coordination tracker
        self.coordinator = CoordinationTracker()

        # Role-based permissions
        self.permissions: Dict[AgentRole, List[str]] = {
            AgentRole.TRIAGE: [
                "triage_alert", "query_host", "escalate_to_human", "share_intel",
            ],
            AgentRole.CONTAINMENT: [
                "isolate_host", "block_ip", "disable_account", "query_host",
            ],
            AgentRole.THREAT_HUNTER: [
                "query_host", "search_iocs", "analyze_logs", "share_intel",
            ],
            AgentRole.FORENSICS: [
                "query_host", "collect_evidence", "trace_timeline",
            ],
        }

        # Episode-level trackers
        self.isolated_hosts_count: int = 0
        self.classified_alerts: set = set()
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: int = 1,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        options = options or {}

        # Apply curriculum-level settings
        curriculum_params = self.curriculum.get_scenario_params()

        if self.use_realistic:
            options["use_realistic"] = True
            options["curriculum_level"] = curriculum_params["curriculum_level"]
            options["difficulty"] = curriculum_params["difficulty"]

        if self.use_adversarial and self.attacker is not None:
            defender_perf = self._get_defender_performance()
            attack_scenario = self.attacker.generate_attack(defender_perf)
            self.current_state = attack_scenario["initial_state"]
            self.ground_truth = attack_scenario["ground_truth"]
            self.action_history = []
            self.query_history = []
            self.step_count = 0
            # Use curriculum max_steps for long-horizon support
            self.max_steps = self.curriculum.max_steps
            base_obs = self._get_masked_observation()
            info: Dict[str, Any] = {
                "task_id": task_id,
                "seed": seed,
                "adversarial_attack": attack_scenario["name"],
            }
        else:
            base_obs, info = super().reset(task_id, seed, options)
            # Override max_steps with curriculum-controlled value
            self.max_steps = self.curriculum.max_steps

        # Reset episode-level trackers
        self.isolated_hosts_count = 0
        self.classified_alerts = set()
        self._episode_rewards = []

        # Reset coordination tracker for new episode
        self.coordinator.reset()

        # Build per-agent observations
        agent_observations = {
            AgentRole.TRIAGE: self._filter_for_triage(base_obs),
            AgentRole.CONTAINMENT: self._filter_for_containment(base_obs),
            AgentRole.THREAT_HUNTER: self._filter_for_hunter(base_obs),
            AgentRole.FORENSICS: self._filter_for_forensics(base_obs),
        }

        # Inject live threat intel if enabled
        if self.use_live_intel:
            try:
                from app.scenarios.threat_intel_live import LiveThreatIntel
                intel_provider = LiveThreatIntel()
                intel_data = intel_provider.fetch_latest()
                info["live_threat_intel"] = {
                    "ips": [i["ip"] for i in intel_data.get("malicious_ips", [])[:3]],
                    "domains": [d["domain"] for d in intel_data.get("malicious_domains", [])[:3]],
                }
            except Exception:
                info["live_threat_intel"] = {"ips": [], "domains": [], "error": "unavailable"}

        # Add curriculum info to reset info
        info["curriculum"] = self.curriculum.status()

        return agent_observations, info

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> Dict:
        base_obs = super().state
        return {
            AgentRole.TRIAGE: self._filter_for_triage(base_obs),
            AgentRole.CONTAINMENT: self._filter_for_containment(base_obs),
            AgentRole.THREAT_HUNTER: self._filter_for_hunter(base_obs),
            AgentRole.FORENSICS: self._filter_for_forensics(base_obs),
        }

    # ------------------------------------------------------------------
    # Observation filters
    # ------------------------------------------------------------------

    def _filter_for_triage(self, base_obs: Dict) -> Dict:
        """TRIAGE sees all alerts."""
        return copy.deepcopy(base_obs)

    def _filter_for_containment(self, base_obs: Dict) -> Dict:
        """CONTAINMENT sees only classified alerts."""
        obs = copy.deepcopy(base_obs)
        obs["alerts"] = [
            a for a in obs.get("alerts", [])
            if a.get("alert_id") in self.classified_alerts
        ]
        return obs

    def _filter_for_hunter(self, base_obs: Dict) -> Dict:
        """HUNTER sees host network topology."""
        obs = copy.deepcopy(base_obs)
        obs["network_topology"] = "Mock Network Topology View"
        return obs

    def _filter_for_forensics(self, base_obs: Dict) -> Dict:
        """FORENSICS sees timeline of events."""
        obs = copy.deepcopy(base_obs)
        obs["timeline"] = list(self.action_history)
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[AgentRole, Dict]
    ) -> Tuple[Dict, Dict[AgentRole, float], bool, bool, Dict]:
        rewards: Dict[AgentRole, float] = {role: 0.0 for role in AgentRole}
        info: Dict[str, Any] = {
            "executed_actions": {},
            "failures": {},
            "grading": {},
            "compliance": {},
        }

        # Advance environment step
        self.step_count += 1

        # ---- Permission & resource validation ----
        valid_actions: Dict[AgentRole, Dict] = {}
        for role, action in actions.items():
            action_type = action.get("action_type", "")

            if action_type not in self.permissions.get(role, []):
                info["failures"][role] = (
                    f"Action {action_type} not permitted for role {role}"
                )
                continue

            if action_type == "isolate_host" and self.isolated_hosts_count >= 3:
                info["failures"][role] = "No isolation slots"
                continue

            valid_actions[role] = action

        # ---- Apply state changes (simultaneous) ----
        for role, action in valid_actions.items():
            action_type = action.get("action_type", "")
            target_id = action.get("target_id")

            self.action_history.append(action)

            if action_type == "query_host":
                if target_id and target_id not in self.query_history:
                    self.query_history.append(target_id)

            elif action_type == "isolate_host":
                for host in self.current_state.get("hosts", []):
                    if host.get("host_id") == target_id:
                        if host.get("status") != "isolated":
                            host["status"] = "isolated"
                            self.isolated_hosts_count += 1
                        break

            elif action_type == "triage_alert":
                if target_id:
                    self.classified_alerts.add(target_id)

        # ---- Grade & compliance-check each action ----
        for role, action in valid_actions.items():
            justification = action.get("justification", {})
            obs = self._get_masked_observation()

            # Standard grading
            grading_result = self.grader.grade_action(
                action=action,
                justification=justification,
                observable_state=obs,
                resulting_state=self.current_state,
                ground_truth=self.ground_truth,
                action_history=self.action_history,
            )
            rewards[role] = grading_result["final_score"]
            info["grading"][role] = grading_result
            info["executed_actions"][role] = action.get("action_type")

            # EU AI Act compliance evaluation
            compliance_record = self.compliance_engine.evaluate_action(
                action=action,
                justification=justification,
                observable_state=obs,
                action_history=self.action_history,
            )
            info["compliance"][role] = {
                "action_id": compliance_record.action_id,
                "overall_score": compliance_record.overall_score,
                "compliant": compliance_record.compliant,
                "risk_level": compliance_record.risk_level,
            }

            # Coordination bonus for sharing intel
            if action.get("action_type") == "share_intel":
                self.coordinator.log_communication(
                    from_agent=role.value,
                    to_agent=action.get("to_agent", "team"),
                    message=action.get("message", {}),
                    led_to_action=True,
                )
                rewards[role] += 0.15

        # ---- Post-process coordination bonuses ----
        for role, action in valid_actions.items():
            action_type = action.get("action_type", "")
            if role == AgentRole.CONTAINMENT and action_type in (
                "isolate_host", "block_ip"
            ):
                target_id = action.get("target_id")
                host_alerts = [
                    a for a in self.current_state.get("alerts", [])
                    if a.get("source_host") == target_id
                ]
                if any(
                    a.get("alert_id") in self.classified_alerts
                    for a in host_alerts
                ):
                    rewards[AgentRole.CONTAINMENT] += 0.2
                    # Log a successful handoff from TRIAGE -> CONTAINMENT
                    self.coordinator.log_handoff(
                        from_agent=AgentRole.TRIAGE.value,
                        to_agent=AgentRole.CONTAINMENT.value,
                        target=str(target_id),
                        success=True,
                    )
                    rewards[AgentRole.TRIAGE] += 0.2

        # Track episode rewards for curriculum
        active_rewards = [r for role, r in rewards.items() if role in valid_actions]
        step_avg = sum(active_rewards) / max(1, len(active_rewards)) if active_rewards else 0.0
        self._episode_rewards.append(step_avg)

        # Clamp step_count to max_steps for safety in long-horizon mode
        terminated = False
        truncated = self.step_count >= self.max_steps

        agent_observations = self.state

        # On episode end, record reward for curriculum manager
        if truncated or terminated:
            episode_total = (
                sum(self._episode_rewards) / max(1, len(self._episode_rewards))
            )
            transition = self.curriculum.record_episode(episode_total)
            info["curriculum_transition"] = transition
            info["coordination_report"] = self.coordinator.get_coordination_report()
            info["coordination_score"] = self.coordinator.calculate_coordination_score()

        # Inject live threat intel
        if self.use_live_intel:
            try:
                from app.scenarios.threat_intel_live import LiveThreatIntel
                intel_provider = LiveThreatIntel()
                intel_data = intel_provider.fetch_latest()
                info["live_threat_intel"] = {
                    "ips": [i["ip"] for i in intel_data.get("malicious_ips", [])[:3]],
                    "domains": [
                        d["domain"] for d in intel_data.get("malicious_domains", [])[:3]
                    ],
                }
            except Exception:
                info["live_threat_intel"] = {"ips": [], "domains": [], "error": "unavailable"}

        return agent_observations, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_defender_performance(self) -> Dict:
        """Aggregate defender performance metrics."""
        return {
            "detection_speed": 12,
            "false_negative_rate": 0.35,
        }
