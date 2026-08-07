import gymnasium as gym
import pytest

from options_envs.envs.two_rooms import TwoRoomsEnv


def test_two_rooms_direct_and_registered_creation():
    direct = TwoRoomsEnv()
    registered = gym.make("OptionsEnv/TwoRooms-v0")
    try:
        assert direct.n_states == 73
        assert direct.observation_space.n == 73
        assert direct.action_space.n == 4
        assert isinstance(registered.unwrapped, TwoRoomsEnv)
    finally:
        direct.close()
        registered.close()


def test_two_rooms_reset_and_step_contract():
    env = TwoRoomsEnv()
    try:
        observation, info = env.reset(seed=7)
        assert observation == env.start_state
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        result = env.step(env.RIGHT)
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


def test_two_rooms_seed_is_reproducible_and_deterministic():
    env_a = TwoRoomsEnv()
    env_b = TwoRoomsEnv()
    try:
        assert env_a.reset(seed=1) == env_b.reset(seed=999)
        for action in (0, 3, 3, 1, 2, 0):
            assert env_a.step(action) == env_b.step(action)
    finally:
        env_a.close()
        env_b.close()


def test_two_rooms_wall_and_divider_block_movement():
    env = TwoRoomsEnv()
    try:
        left_top = env.tostate[(1, 1)]
        assert env.transition(left_top, env.UP) == left_top
        assert env.transition(left_top, env.LEFT) == left_top

        left_of_wall = env.tostate[(6, 2)]
        assert env.transition(left_of_wall, env.RIGHT) == left_of_wall
        assert env.transition(env.tostate[(6, 3)], env.RIGHT) == env.hallway_state
    finally:
        env.close()


def test_two_rooms_only_hallway_connects_rooms():
    env = TwoRoomsEnv()
    try:
        for y in (1, 2, 4, 5, 6):
            left = env.tostate[(6, y)]
            assert env.transition(left, env.RIGHT) == left
        assert env.transition(env.hallway_state, env.RIGHT) == env.tostate[(8, 3)]
        assert env.room_id(env.hallway_state) == "hallway"
        assert env.room_id(env.start_state) == "left"
        assert env.room_id(env.goal_state) == "right"
    finally:
        env.close()


def test_two_rooms_goal_definition_preserves_executable_reference():
    env = TwoRoomsEnv()
    try:
        assert env._id_state[env.goal_state] == (10, 6)
        assert env.goal_state == env.tostate[(10, 6)]
        assert env.goal_state not in env.gray_region
    finally:
        env.close()


def test_two_rooms_gray_region_reward_and_goal_reward():
    env = TwoRoomsEnv()
    try:
        penalized = env.tostate[(2, 1)]
        assert penalized in env.gray_region
        assert env.base_reward_for_next_state(penalized) == -1.0
        assert env.base_reward_for_next_state(env.goal_state) == 1.0

        env.reset(seed=0)
        env._state = env.tostate[(1, 1)]
        observation, reward, terminated, truncated, _ = env.step(env.RIGHT)
        assert observation == penalized
        assert reward == -1.0
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_two_rooms_known_trajectory_reaches_goal():
    env = TwoRoomsEnv()
    try:
        env.reset(seed=0)
        actions = [env.RIGHT] * 9 + [env.DOWN] * 3
        for action in actions[:-1]:
            _, _, terminated, _, _ = env.step(action)
            assert not terminated
        observation, reward, terminated, truncated, info = env.step(actions[-1])
        assert observation == env.goal_state
        assert reward == 1.0
        assert terminated
        assert not truncated
        assert info["is_success"]
        with pytest.raises(RuntimeError):
            env.step(env.UP)
    finally:
        env.close()


def test_two_rooms_render_ansi():
    env = TwoRoomsEnv(render_mode="ansi")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, str)
        assert "#" in frame
        assert "A" in frame
        assert "G" in frame
        assert "~" in frame
    finally:
        env.close()


def test_two_rooms_invalid_action_and_gymnasium_checker():
    env = TwoRoomsEnv()
    try:
        with pytest.raises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(4)
    finally:
        env.close()

    from gymnasium.utils.env_checker import check_env

    check_env(TwoRoomsEnv(), skip_render_check=True)
