# Pinball

| Action Space | `Discrete(5)` |
| --- | --- |
| Observation Space | `Box(4,)` (`float32`) |
| Import | `gymnasium.make("OptionsEnv/Pinball-v0")` |

## Description

Pinball is a continuous-state environment for hierarchical reinforcement
learning and options research. The agent applies impulses to a ball moving in a
bounded table containing polygonal obstacles and a target.

The default observation is the ball state. RGB frames are available through
`render()`, but images are not used as observations.

```python
import gymnasium as gym
import options_envs

env = gym.make("OptionsEnv/Pinball-v0")
observation, info = env.reset(seed=0)
```

The observation type can also be selected explicitly:

```python
env = gym.make("OptionsEnv/Pinball-v0", obs_type="state")
observation, info = env.reset(seed=0)
```

## Action Space

There are five discrete actions:

| Action | Effect |
|---:|---|
| 0 | thrust right |
| 1 | thrust down |
| 2 | thrust left |
| 3 | thrust up |
| 4 | no-op |

## Observation Space

The state observation is a `numpy.ndarray` with shape `(4,)` and dtype
`np.float32`:

```text
[x, y, vx, vy]
```

| Index | Field | Range |
|---:|---|---|
| 0 | `x` — horizontal position | `[0, 1]` |
| 1 | `y` — vertical position | `[0, 1]` |
| 2 | `vx` — horizontal velocity | `[-1, 1]` |
| 3 | `vy` — vertical velocity | `[-1, 1]` |

RGB rendering does not change the observation:

```python
env = gym.make("OptionsEnv/Pinball-v0", render_mode="rgb_array")
observation, info = env.reset(seed=0)
frame = env.render()  # (600, 800, 3), uint8
```

## Rewards

| Event | Reward |
|---|---:|
| Thrust action without reaching the target | `-5.0` |
| No-op action without reaching the target | `-1.0` |
| Reaching the target | `10000.0` |

The implementation uses the following reward constants:

| Constant | Value | Meaning |
|---|---:|---|
| `step_penalty` | `-1.0` | No-op step penalty. |
| `thrust_penalty` | `-5.0` | Thrust step penalty. |
| `success_reward` | `10000.0` | Reward for reaching the target. |

## Starting State

At reset, the ball is placed at one of the start positions defined by the
selected layout. The position is sampled with Gymnasium's seeded random
generator.

## Episode Termination

The episode terminates when the ball reaches the target. It is truncated when
the configured episode step limit is reached before success.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `task` | `"default-v0"` | Selects the task and layout. |
| `max_steps` | Task-dependent | Environment-level truncation horizon. |
| `obs_type` | `"state"` | Current observation representation. |
| `render_mode` | `None` | `None`, `"rgb_array"`, or `"human"`. |

Available tasks are `default-v0` and `hard-v0`.

| Task | Layout | `max_steps` |
|---|---|---:|
| `default-v0` | `default-v0.cfg` | `500` |
| `hard-v0` | `hard-v0.cfg` | `10000` |

## Version History

- `v0`: initial Pinball environment in `options-envs`.

## Notes

The environment supports the following render modes:

| Mode | Description |
|---|---|
| `None` | No rendering. |
| `"rgb_array"` | Returns a `(600, 800, 3)` RGB array. |
| `"human"` | Displays the table in a Pygame window. |

Layouts and task definitions are packaged under
`options_envs/envs/pinball/assets/layouts/` and
`options_envs/envs/pinball/tasks.py`.

## Useful Information

### Runnable Examples

```bash
python examples/pinball/random_agent.py
python examples/pinball/render.py
python examples/pinball/render.py outputs/my_run.mp4
```

### Layouts and Assets

`options_envs/envs/pinball/assets/layouts/`

### Main Files

| File | Description |
|---|---|
| `options_envs/envs/pinball/env.py` | Gymnasium Pinball environment. |
| `options_envs/envs/pinball/tasks.py` | Versioned task definitions. |
| `options_envs/envs/pinball/config.py` | Layout configuration loader. |
| `options_envs/envs/pinball/registration.py` | Registration of `OptionsEnv/Pinball-v0`. |
| `options_envs/envs/pinball/assets/layouts/` | Packaged environment layouts. |

## References

The Pinball environment is based on the classic Pinball domain used in
reinforcement learning and option-learning research.

- [Konidaris / Brown IRL Pinball domain](http://irl.cs.brown.edu/pinball/)
- [Pierre-Luc Bacon's Python RL implementation](https://github.com/amarack/python-rl/tree/master)

## Code Location

`options_envs/envs/pinball/`
