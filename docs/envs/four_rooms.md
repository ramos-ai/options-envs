# FourRooms

| Action Space | `Discrete(4)` |
| --- | --- |
| Observation Space | `Discrete(104)` |
| Import | `gymnasium.make("OptionsEnv/FourRooms-v0")` |

## Description

FourRooms is a tabular stochastic gridworld for hierarchical reinforcement
learning and options research. Its layout and transition semantics are adapted
from the Option-Critic reference implementation.

```python
import gymnasium as gym
import options_envs

env = gym.make("OptionsEnv/FourRooms-v0")
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
13×13 and contains 104 navigable cells. IDs are assigned in row-major order
over blank cells.

The mappings used by tabular planners are available as `env.tostate` and
`env.tocell`.

## Layout

`#` denotes a wall and a blank cell denotes a navigable state:

```text
#############
#     #     #
#     #     #
#           #
#     #     #
#     #     #
## ####     #
#     ### ###
#     #     #
#     #     #
#           #
#     #     #
#############
```

## Rewards

| Event | Reward |
|---|---:|
| Normal transition | `0.0` |
| Entering the goal | `+1.0` |

There is no intrinsic reward, shaping reward, option cost, or option policy
inside the environment.

## Starting State

The initial state is sampled uniformly from the 103 navigable states different
from the goal. Sampling uses `self.np_random` and is reproducible when a seed
is passed to `reset()`.

## Episode Termination

The goal is state `62`, at grid position `(row=7, column=9)`. Entering the goal
returns `terminated=True`. There is no environment-level truncation horizon;
`truncated` is always `False`.

Moves into walls or outside the grid leave the state unchanged. If `step()` is
called after termination, the environment raises `RuntimeError` and must be
reset before continuing.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `render_mode` | `None` | `None` or `"ansi"`. |

The `"ansi"` mode returns a textual grid with the agent and goal marked.

## Runnable Examples

```bash
python examples/four_rooms/random_agent.py
python examples/four_rooms/render.py
python examples/four_rooms/render.py --steps 20 --seed 0
```

## Version History

- `v0`: initial FourRooms environment in `options-envs`.

## Notes

When the intended destination is free, the intended action is taken with
probability `2/3`. With probability `1/3`, one of the currently available
neighbouring free cells is selected uniformly. If the intended destination is
a wall, the agent remains in place.

## References

- Sutton, R. S., Precup, D., & Singh, S. (1999). **“Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.”** *Artificial Intelligence, 112*(1–2), 181–211.
- [Option-Critic reference implementation — `jeanharb/option_critic`](https://github.com/jeanharb/option_critic/blob/master/fourrooms/fourrooms.py)

## Code Location

`options_envs/envs/four_rooms/`
