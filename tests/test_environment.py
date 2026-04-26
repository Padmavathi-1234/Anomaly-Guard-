"""
AnomalyGuard — Core Environment Tests
Validates OpenEnv compliance, reproducibility, and reward logic.
"""
import pytest
from app.environment import AnomalyGuardEnvironment
from app.models import Action


@pytest.fixture
def env():
    return AnomalyGuardEnvironment()


def test_reset_returns_tuple(env):
    """reset() must return (Observation, dict) tuple."""
    result = env.reset(task_id=1, seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert obs is not None


def test_step_returns_5_tuple(env):
    """step() must return OpenEnv-compliant 5-tuple."""
    env.reset(task_id=1, seed=42)
    result = env.step(Action(action_type="monitor", target=""))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_in_valid_range(env):
    """Reward must always be in [-1.0, 1.0]."""
    env.reset(task_id=1, seed=42)
    _, reward, _, _, _ = env.step(Action(action_type="monitor", target=""))
    assert -1.0 <= reward <= 1.0


def test_reproducibility(env):
    """Same seed must produce identical scenarios."""
    obs1, _ = env.reset(task_id=1, seed=42)
    obs2, _ = env.reset(task_id=1, seed=42)
    assert len(obs1.alerts) == len(obs2.alerts)
    assert obs1.alerts[0].alert_id == obs2.alerts[0].alert_id
    assert obs1.max_steps == obs2.max_steps


def test_partial_observability_starts_hidden(env):
    """All hosts must start unqueried with hidden details."""
    obs, _ = env.reset(task_id=1, seed=42)
    assert len(env._queried_hosts) == 0
    for host in obs.hosts:
        assert host.is_queried == False
        assert host.c2_active == False
        assert host.persistence == []


def test_query_host_reveals_details(env):
    """query_host must mark host as queried."""
    obs, _ = env.reset(task_id=1, seed=42)
    host_id = obs.hosts[0].host_id
    obs2, _, _, _, info = env.step(
        Action(action_type="query_host", target=host_id)
    )
    assert host_id in env._queried_hosts
    queried = next(h for h in obs2.hosts if h.host_id == host_id)
    assert queried.is_queried == True


def test_termination_reason_in_info(env):
    """info must always contain termination_reason."""
    env.reset(task_id=1, seed=42)
    _, _, _, _, info = env.step(Action(action_type="monitor", target=""))
    assert "termination_reason" in info
    assert info["termination_reason"] in [
        "in_progress", "task_complete",
        "network_shutdown", "total_compromise", "timeout"
    ]


def test_progress_bonus_in_info(env):
    """info must always contain progress_bonus."""
    env.reset(task_id=1, seed=42)
    _, _, _, _, info = env.step(Action(action_type="monitor", target=""))
    assert "progress_bonus" in info
    assert isinstance(info["progress_bonus"], float)


def test_state_method_exists(env):
    """state() method must exist and return full unmasked state."""
    env.reset(task_id=1, seed=42)
    s = env.state()
    assert s is not None
    assert "alerts" in s
    assert "hosts" in s
    assert "triaged" in s
    assert "isolated" in s


def test_different_seeds_give_different_scenarios(env):
    """Different seeds must produce different scenarios."""
    obs1, _ = env.reset(task_id=1, seed=42)
    obs2, _ = env.reset(task_id=1, seed=99)
    assert (
        obs1.alerts[0].alert_id != obs2.alerts[0].alert_id
        or len(obs1.alerts) != len(obs2.alerts)
    )


def test_episode_done_raises_on_step(env):
    """step() after episode end must raise RuntimeError."""
    env.reset(task_id=1, seed=42)
    action = Action(action_type="monitor", target="")
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    with pytest.raises(RuntimeError):
        env.step(action)


def test_curriculum_level_exists(env):
    """Curriculum level must be between 1 and 10."""
    assert 1 <= env._curriculum_level <= 10


def test_all_tasks_reset_correctly(env):
    """All 3 tasks must reset without errors."""
    for task_id in [1, 2, 3]:
        obs, info = env.reset(task_id=task_id, seed=42)
        assert obs is not None
        assert len(obs.alerts) > 0
        assert len(obs.hosts) > 0


def test_queried_hosts_reset_on_new_episode(env):
    """_queried_hosts must be empty after reset."""
    obs, _ = env.reset(task_id=1, seed=42)
    host_id = obs.hosts[0].host_id
    env.step(Action(action_type="query_host", target=host_id))
    assert len(env._queried_hosts) > 0

    # Reset and verify cleared
    env.reset(task_id=1, seed=42)
    assert len(env._queried_hosts) == 0


def test_reward_breakdown_in_info(env):
    """info must contain reward_breakdown with all components."""
    env.reset(task_id=1, seed=42)
    _, _, _, _, info = env.step(Action(action_type="monitor", target=""))
    assert "reward_breakdown" in info
    rb = info["reward_breakdown"]
    assert "action_correctness" in rb
    assert "explanation_quality" in rb
    assert "progress_bonus" in rb
    assert "final_reward" in rb
