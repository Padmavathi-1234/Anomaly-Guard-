import copy
from typing import Tuple, Dict, Optional, Any
from openenv.core import Environment as Env
from app.scenarios.scenario_base import generate_basic_scenario
from app.grading.grader_robust import RobustGrader

class AnomalyGuardBase(Env):
    def __init__(self):
        super().__init__()
        self.grader = RobustGrader()
        self.current_state = {}
        self.ground_truth = {}
        self.action_history = []
        self.query_history = []
        self.step_count = 0
        self.max_steps = 15

    @property
    def state(self) -> Dict:
        return self._get_masked_observation()

    def reset(self, task_id: int = 1, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if options and options.get("use_realistic", False):
            from app.scenarios.realistic_attacks import RealisticScenarioGenerator
            generator = RealisticScenarioGenerator()
            # Pass curriculum-level params if provided
            difficulty = options.get("difficulty", 0.85)
            curriculum_level = options.get("curriculum_level", 1)
            scenario = generator.generate(
                difficulty=difficulty, seed=seed,
                curriculum_level=curriculum_level,
            )
        else:
            scenario = generate_basic_scenario(task_id, seed)
        
        if options and options.get("use_live_intel", False):
            from app.scenarios.threat_intel_live import LiveThreatIntel
            intel_provider = LiveThreatIntel()
            scenario = intel_provider.inject_into_scenario(scenario)

        # Set state
        self.current_state = copy.deepcopy(scenario["initial_state"])
        self.ground_truth = copy.deepcopy(scenario["ground_truth"])
        self.action_history = []
        self.query_history = []
        self.step_count = 0
        self.max_steps = scenario.get("max_steps", 15)
        
        info = {
            "task_id": task_id,
            "seed": seed
        }
        
        return self._get_masked_observation(), info

    def _get_masked_observation(self) -> Dict:
        # Hide ground truth from agent
        obs = {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "alerts": copy.deepcopy(self.current_state.get("alerts", [])),
            "hosts": copy.deepcopy(self.current_state.get("hosts", [])),
            "query_history": list(self.query_history)
        }
        
        # Ensure ground truth is not leaked for hosts not in query history
        for host in obs["hosts"]:
            host_id = host["host_id"]
            if host_id not in self.query_history:
                # Remove compromised and c2_active if they somehow got in
                host.pop("compromised", None)
                host.pop("c2_active", None)
            else:
                # Add ground truth if queried
                gt_host = self.ground_truth["hosts"].get(host_id, {})
                host["compromised"] = gt_host.get("compromised", False)
                host["c2_active"] = gt_host.get("c2_active", False)
                
        # Make sure no alert truth is leaked
        for alert in obs["alerts"]:
            alert.pop("is_true_positive", None)
            alert.pop("mitre_technique", None)
            
        return obs

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        action_type = action.get("action_type")
        target_id = action.get("target_id")
        
        self.action_history.append(action)
        self.step_count += 1
        
        if action_type == "query_host":
            host_id = target_id
            if host_id not in self.query_history:
                self.query_history.append(host_id)
                
        elif action_type == "isolate_host":
            host_id = target_id
            for host in self.current_state["hosts"]:
                if host["host_id"] == host_id:
                    host["status"] = "isolated"
                    break
                    
        grading_result = self.grader.grade_action(
            action=action,
            justification=action.get("justification", {}),
            observable_state=self._get_masked_observation(),
            resulting_state=self.current_state,
            ground_truth=self.ground_truth,
            action_history=self.action_history
        )
        
        reward = grading_result["final_score"]
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        info = {
            "action_executed": action_type,
            "action_history": list(self.action_history),
            "grading": grading_result
        }
        
        return self._get_masked_observation(), reward, terminated, truncated, info
