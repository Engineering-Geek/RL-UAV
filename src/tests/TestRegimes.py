# Test _BaseRegime and TargetRegime, there may be more Regimes but all inherit from _BaseRegime
import pytest
from src.TrainingRegimes import TargetRegime


# region Declaring the fixtures
@pytest.fixture
def target_regime():
    return TargetRegime(
        tolerance_distance=0.1,
        max_time=20,
        reward_distance_coefficient=1,
        reward_distance_exp=1,
        reward_distance_max=100,
        reward_goal=100,
        reward_velocity_coefficient=1,
        reward_velocity_exp=1,
        reward_velocity_max=100,
        penalty_time=1,
        penalty_crash=100,
        dt=0.1,
        height=256,
        width=256,
        n_camera=0
    )
# endregion


# region Test TargetRegime
def test_reward_distance(target_regime):
    assert isinstance(target_regime.reward_distance, float)


def test_reward_time(target_regime):
    assert isinstance(target_regime.reward_time, float)


def test_reward_crash(target_regime):
    assert isinstance(target_regime.reward_crash, float)


def test_reward_goal(target_regime):
    assert isinstance(target_regime.reward_goal, float)


def test_reward_velocity(target_regime):
    assert isinstance(target_regime.reward_velocity, float)


def test_reward(target_regime):
    assert isinstance(target_regime.reward, float)


def test_metrics(target_regime):
    metrics = target_regime.metrics
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"distance", "goal_reached", "goal_velocity", "hit_ground", "time"}


def test_done(target_regime):
    assert isinstance(target_regime.done, bool)


def test_truncated(target_regime):
    assert isinstance(target_regime.truncated, bool)


def test_goal_reached(target_regime):
    assert isinstance(target_regime.goal_reached, bool)
# endregion
