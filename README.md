# options-envs

A collection of Gymnasium-compatible environments for hierarchical reinforcement learning and options research.

## Installation

### Runtime only

From a local clone:

```bash
pip install .
```

For editable development from a local clone:

```bash
pip install -e .
```

From GitHub:

```bash
pip install "git+https://github.com/ramos-ai/options-envs.git"
```

To install a specific branch, tag, or commit:

```bash
pip install "git+https://github.com/ramos-ai/options-envs.git@main"
```

### Development and tests

```bash
pip install -e ".[dev]"
python -m pytest
python examples/pinball/random_agent.py
```

### Render example

The render example writes an `.mp4` file and uses the `dev` extra.

```bash
python examples/pinball/render.py
python examples/pinball/render.py outputs/my_run.mp4
```

## Dependency policy

This project is a library. Compatible dependencies live in `pyproject.toml`. We do not use `requirements.txt` as the primary source of dependencies. For applications, experiments, or deployable environments that need exact reproducibility, use a lockfile or a dedicated `requirements` file outside the library.

## Quick start

```python
import gymnasium as gym
import options_envs

env = gym.make("OptionsEnv/Pinball-v0")
obs, info = env.reset(seed=0)
```

## Available environment

- `OptionsEnv/Pinball-v0`

## Testing

```bash
python -m pytest
```

## Documentation

See `docs/`:

- `docs/adding_environment.md`
- `docs/versioning.md`
- `docs/envs/`

## Structure

Environments live in `options_envs/envs/`, wrappers live in `options_envs/wrappers/`, and assets are packaged alongside each environment.
