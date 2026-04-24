"""
Self-Evolving Curriculum Manager
=================================
Implements an 8-level adaptive curriculum system that:
- Automatically promotes/demotes the agent based on rolling reward averages
- Controls episode length (long-horizon mode: 20 → 80 steps)
- Adjusts attack complexity per level
- Provides serializable state for checkpointing and API exposure

Level progression:
  Level 1 (Beginner)      → 15–20 steps, simple 3-phase attacks
  Level 2 (Novice)        → 18–25 steps
  Level 3 (Intermediate)  → 22–30 steps
  Level 4 (Skilled)       → 28–40 steps
  Level 5 (Advanced)      → 32–45 steps
  Level 6 (Expert)        → 38–55 steps
  Level 7 (Master)        → 45–65 steps
  Level 8 (Grandmaster)   → 55–80 steps, full 7-phase multi-vector attacks
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


# ---------------------------------------------------------------------------
# Level definitions
# ---------------------------------------------------------------------------

@dataclass
class CurriculumLevel:
    """Configuration for a single curriculum level."""
    level: int
    name: str
    min_steps: int
    max_steps: int
    min_phases: int
    max_phases: int
    difficulty: float           # Base difficulty for RealisticScenarioGenerator
    promote_threshold: float    # Avg reward above this → promote
    demote_threshold: float     # Avg reward below this → demote
    description: str = ""


CURRICULUM_LEVELS: Dict[int, CurriculumLevel] = {
    1: CurriculumLevel(
        level=1, name="Beginner", min_steps=15, max_steps=20,
        min_phases=3, max_phases=4, difficulty=0.30,
        promote_threshold=0.75, demote_threshold=0.0,
        description="Basic alert triage with simple single-vector attacks",
    ),
    2: CurriculumLevel(
        level=2, name="Novice", min_steps=18, max_steps=25,
        min_phases=3, max_phases=5, difficulty=0.40,
        promote_threshold=0.75, demote_threshold=0.40,
        description="Multi-phase attacks with moderate observables",
    ),
    3: CurriculumLevel(
        level=3, name="Intermediate", min_steps=22, max_steps=30,
        min_phases=4, max_phases=5, difficulty=0.50,
        promote_threshold=0.75, demote_threshold=0.40,
        description="Contextual observables and false-positive noise",
    ),
    4: CurriculumLevel(
        level=4, name="Skilled", min_steps=28, max_steps=40,
        min_phases=4, max_phases=6, difficulty=0.60,
        promote_threshold=0.75, demote_threshold=0.40,
        description="Long-horizon episodes with coordination requirements",
    ),
    5: CurriculumLevel(
        level=5, name="Advanced", min_steps=32, max_steps=45,
        min_phases=5, max_phases=6, difficulty=0.70,
        promote_threshold=0.75, demote_threshold=0.40,
        description="Complex multi-vector attacks with limited prevention windows",
    ),
    6: CurriculumLevel(
        level=6, name="Expert", min_steps=38, max_steps=55,
        min_phases=5, max_phases=7, difficulty=0.80,
        promote_threshold=0.75, demote_threshold=0.40,
        description="Advanced evasion techniques with minimal alerts",
    ),
    7: CurriculumLevel(
        level=7, name="Master", min_steps=45, max_steps=65,
        min_phases=6, max_phases=7, difficulty=0.90,
        promote_threshold=0.75, demote_threshold=0.40,
        description="APT-level scenarios with lateral movement chains",
    ),
    8: CurriculumLevel(
        level=8, name="Grandmaster", min_steps=55, max_steps=80,
        min_phases=6, max_phases=7, difficulty=0.95,
        promote_threshold=1.0,  # Cannot promote beyond level 8
        demote_threshold=0.40,
        description="Full-spectrum multi-vector attacks with maximum complexity",
    ),
}


# ---------------------------------------------------------------------------
# Curriculum Manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """
    Adaptive curriculum controller for RL training.

    Tracks a rolling window of episode rewards and automatically adjusts
    the difficulty level.  Integrates with RealisticScenarioGenerator and
    the environment's max_steps parameter.
    """

    def __init__(self, window_size: int = 100, start_level: int = 1):
        """
        Args:
            window_size: Number of recent episodes to average over.
            start_level: Starting curriculum level (1–8).
        """
        self.window_size = window_size
        self.current_level = max(1, min(8, start_level))
        self.reward_history: deque = deque(maxlen=window_size)
        self.level_history: List[Dict] = []   # Audit log of level changes
        self.total_episodes = 0
        self.promotions = 0
        self.demotions = 0
        self._created_at = time.time()

        # Record initial state
        self._log_transition("init", 0, self.current_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def level_config(self) -> CurriculumLevel:
        """Return the configuration for the current level."""
        return CURRICULUM_LEVELS[self.current_level]

    @property
    def avg_reward(self) -> float:
        """Rolling average reward over the last `window_size` episodes."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def max_steps(self) -> int:
        """Suggested max steps for the current level (uses the level max)."""
        cfg = self.level_config
        return cfg.max_steps

    @property
    def difficulty(self) -> float:
        """Base difficulty for the current level."""
        return self.level_config.difficulty

    def record_episode(self, episode_reward: float) -> Dict:
        """
        Record a completed episode's reward and trigger level adjustment.

        Args:
            episode_reward: The total/average reward for the episode.

        Returns:
            Dict with level transition info (if any).
        """
        self.reward_history.append(episode_reward)
        self.total_episodes += 1

        # Only evaluate after enough episodes
        if len(self.reward_history) < min(10, self.window_size):
            return {"action": "collecting", "current_level": self.current_level,
                    "episodes_until_eval": min(10, self.window_size) - len(self.reward_history)}

        avg = self.avg_reward
        old_level = self.current_level
        cfg = self.level_config

        if avg > cfg.promote_threshold and self.current_level < 8:
            self.current_level += 1
            self.promotions += 1
            self.reward_history.clear()  # Reset window after transition
            self._log_transition("promote", old_level, self.current_level, avg)
            return {"action": "promoted", "from": old_level, "to": self.current_level,
                    "avg_reward": round(avg, 4)}

        if avg < cfg.demote_threshold and self.current_level > 1:
            self.current_level -= 1
            self.demotions += 1
            self.reward_history.clear()
            self._log_transition("demote", old_level, self.current_level, avg)
            return {"action": "demoted", "from": old_level, "to": self.current_level,
                    "avg_reward": round(avg, 4)}

        return {"action": "hold", "current_level": self.current_level,
                "avg_reward": round(avg, 4)}

    def get_scenario_params(self) -> Dict:
        """
        Return the parameters to pass to RealisticScenarioGenerator.generate().

        Returns:
            Dict with difficulty, curriculum_level, and max_steps.
        """
        cfg = self.level_config
        return {
            "difficulty": cfg.difficulty,
            "curriculum_level": self.current_level,
        }

    def reset(self, start_level: int = 1) -> Dict:
        """Reset curriculum state to a fresh start."""
        self.current_level = max(1, min(8, start_level))
        self.reward_history.clear()
        self.level_history.clear()
        self.total_episodes = 0
        self.promotions = 0
        self.demotions = 0
        self._log_transition("reset", 0, self.current_level)
        return self.status()

    def status(self) -> Dict:
        """Return full curriculum status for API exposure."""
        cfg = self.level_config
        return {
            "current_level": self.current_level,
            "level_name": cfg.name,
            "level_description": cfg.description,
            "difficulty": cfg.difficulty,
            "max_steps": cfg.max_steps,
            "step_range": [cfg.min_steps, cfg.max_steps],
            "promote_threshold": cfg.promote_threshold,
            "demote_threshold": cfg.demote_threshold,
            "avg_reward": round(self.avg_reward, 4),
            "episodes_in_window": len(self.reward_history),
            "window_size": self.window_size,
            "total_episodes": self.total_episodes,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "level_history": self.level_history[-20:],  # Last 20 transitions
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_transition(self, action: str, from_level: int, to_level: int,
                        avg_reward: float = 0.0):
        self.level_history.append({
            "action": action,
            "from_level": from_level,
            "to_level": to_level,
            "avg_reward": round(avg_reward, 4),
            "episode": self.total_episodes,
            "timestamp": time.time(),
        })
