from collections import deque

import gymnasium as gym
import pytest

import options_envs
from options_envs.envs.four_rooms import FourRoomsEnv


def test_four_rooms_direct_and_registered_creation():
    direct = FourRoomsEnv()
    registered = gym.make("OptionsEnv/FourRooms-v0")
    try:
        assert direct.n_states == 104
        assert direct.observation_space.n == 104
        assert direct.action_space.n == 4
        assert isinstance(registered.unwrapped, FourRoomsEnv)
    finally:
        direct.close()
        registered.close()


def test_four_rooms_reset_and_step_contract():
    env = FourRoomsEnv()
    try:
        observation, info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        result = env.step(env.action_space.sample())
        assert len(result) == 5
        observation, reward, terminated, truncated, info = result
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not truncated
    finally:
        env.close()


def test_four_rooms_seed_reproduces_stochastic_rollout():
    env_a = FourRoomsEnv()
    env_b = FourRoomsEnv()
    try:
        obs_a, _ = env_a.reset(seed=123)
        obs_b, _ = env_b.reset(seed=123)
        assert obs_a == obs_b

        for action in (0, 3, 1, 2, 3, 0):
            assert env_a.step(action) == env_b.step(action)
    finally:
        env_a.close()
        env_b.close()


def test_four_rooms_walls_block_intended_transition():
    env = FourRoomsEnv()
    try:
        upper_left = env.tostate[(1, 1)]
        assert env.transition(upper_left, env.UP) == upper_left
        assert env.transition(upper_left, env.LEFT) == upper_left
        assert env.tostate[(3, 5)] != env.tostate[(3, 6)]
        assert env.transition(env.tostate[(3, 5)], env.RIGHT) == env.tostate[(3, 6)]
    finally:
        env.close()


def test_four_rooms_has_four_connected_rooms_and_four_passages():
    env = FourRoomsEnv()
    try:
        representatives = [(1, 1), (8, 1), (1, 8), (8, 8)]
        for position in representatives:
            assert position in env.tostate

        passages = [
            ((3, 5), (3, 6)),
            ((6, 2), (7, 2)),
            ((6, 9), (7, 9)),
            ((10, 5), (10, 6)),
        ]
        for left, right in passages:
            assert left in env.tostate
            assert right in env.tostate

        start = env.tostate[representatives[0]]
        queue = deque([start])
        reached = {start}
        while queue:
            state = queue.popleft()
            for action in range(4):
                successor = env.transition(state, action)
                if successor not in reached:
                    reached.add(successor)
                    queue.append(successor)
        assert len(reached) == env.n_states
    finally:
        env.close()


def test_four_rooms_goal_and_initial_distribution():
    env = FourRoomsEnv()
    try:
        assert env.goal_state == 62
        assert env.goal_position == (7, 9)
        assert len(env.initial_states) == 103
        assert env.goal_state not in env.initial_states
    finally:
        env.close()


def test_four_rooms_reaching_goal_terminates_with_reward(monkeypatch):
    env = FourRoomsEnv()
    try:
        env.reset(seed=0)
        predecessor = env.tostate[(6, 9)]
        env._state = predecessor
        env.current_cell = env.tocell[predecessor]
        monkeypatch.setattr(env, "_sample_available_position", lambda position: env.goal_position)

        observation, reward, terminated, truncated, info = env.step(env.DOWN)
        assert observation == env.goal_state
        assert reward == 1.0
        assert terminated
        assert not truncated
        assert info["is_success"]
    finally:
        env.close()


def test_four_rooms_invalid_action_and_step_lifecycle():
    env = FourRoomsEnv()
    try:
        with pytest.raises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(4)
    finally:
        env.close()


def test_four_rooms_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(FourRoomsEnv(), skip_render_check=True)
