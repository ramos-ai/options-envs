# {Environment Name}

| Action Space | `{Action space}` |
| --- | --- |
| Observation Space | `{Observation space}` |
| Import | `gymnasium.make("{Environment ID}")` |

## Description

Describe the task, the environment's purpose, and its main dynamics.

```python
import gymnasium as gym
import options_envs

env = gym.make("{Environment ID}")
observation, info = env.reset(seed=0)
```

## Action Space

Describe the action space and list every action:

| Action | Meaning |
|---:|---|
| 0 | `{Meaning}` |

## Observation Space

Describe the observation returned by `reset()` and `step()`. Include its
shape, dtype, bounds, and the mapping between observations and environment
states when applicable.

## Layout

Document the map, topology, obstacles, rooms, passages, or other spatial
structure. Include a textual grid or a table when useful.

## Rewards

| Event | Reward |
|---|---:|
| `{Event}` | `{Reward}` |

Document reward priority when multiple conditions can apply.

## Starting State

Describe the initial state or initial-state distribution, including seed
behavior.

## Episode Termination

Describe the exact conditions for `terminated=True` and `truncated=True`.
State whether the environment has an internal horizon or relies on a
Gymnasium wrapper.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `render_mode` | `None` | `{Supported render modes}` |

List task-specific or environment-specific constructor arguments here.

## Version History

- `v0`: `{Initial version or compatibility note}`

## Notes

Document determinism, stochasticity, rendering, compatibility notes, known
reference divergences, or other details needed by users.

## References

- `{Paper, dataset, implementation, or other reference}`

## Code Location

`options_envs/envs/{environment_name}/`
