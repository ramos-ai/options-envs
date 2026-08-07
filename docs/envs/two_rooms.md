# TwoRooms

| Action Space | `Discrete(4)` |
| --- | --- |
| Observation Space | `Discrete(73)` |
| Import | `gymnasium.make("OptionsEnv/TwoRooms-v0")` |

## Description

TwoRooms is a deterministic tabular gridworld with two rooms joined by a
single opening in a vertical divider.

```python
import gymnasium as gym
import options_envs

env = gym.make("OptionsEnv/TwoRooms-v0")
observation, info = env.reset(seed=0)
```

## Action Space

There are four primitive actions:

| Action | Direction |
|---:|---|
| 0 | up |
| 1 | down |
| 2 | left |
| 3 | right |

## Observation Space

The observation is the integer ID of the current navigable cell. The grid is
8×15 and contains 73 navigable cells. IDs are assigned in row-major order over
`.` cells using `(x, y)` coordinates.

The mappings used by tabular planners are available as `env.tostate` and
`env.tocell`.

## Layout

`#` is a wall and `.` is a navigable cell. In the annotated view, `~` marks the
gray penalized region:

```text
###############
#.~~~~.#......#
#.~~~~.#......#
#.~~~~........#
#.~~~~.#......#
#.~~~~.#......#
#......#......#
###############
```

The static grid uses `.` for all navigable cells. The two 6×6 rooms are joined
only at `(x=7, y=3)`.

## Rewards

| Event | Reward |
|---|---:|
| Normal transition | `0.0` |
| Entering the gray region | `-1.0` |
| Entering the goal | `+1.0` |

The goal reward has priority if a state were ever to belong to both sets. In
the current layout, the goal is not part of the gray region.

## Starting State

The fixed initial position is `(x=1, y=3)`. The state is reset to this
position on every episode. The seed is accepted through the Gymnasium API, but
does not change the deterministic initial state or transitions.

## Episode Termination

The executable reference definition places the goal at `(x=10, y=6)`, state
`69`. Its prose comments suggest different coordinates; this implementation
preserves the executable definition and documents the divergence explicitly.

Entering the goal returns `terminated=True`. There is no environment-level
truncation horizon; `truncated` is always `False`.

Moves into walls or outside the grid leave the state unchanged. If `step()` is
called after termination, the environment raises `RuntimeError` and must be
reset before continuing.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `render_mode` | `None` | `None` or `"ansi"`. |

The `"ansi"` mode returns a textual grid with `A` for the agent, `G` for the
goal, and `~` for penalized cells.

## Version History

- `v0`: initial TwoRooms environment in `options-envs`.

## Notes

The transition function is deterministic and is available as
`env.transition(state, action)`. The gray region is defined by
`2 <= x <= 5` and `1 <= y <= 5`, excluding the start state.

## References

- Sutton, R. S., Machado, M. C., Holland, G. Z., Szepesvári, D., Timbers, F., Tanner, B., & White, A. (2023). **“Reward-respecting subtasks for model-based reinforcement learning.”** *Artificial Intelligence, 324*, 104001.

## Code Location

`options_envs/envs/two_rooms/`
