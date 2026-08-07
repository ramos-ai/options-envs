from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces


class FourRoomsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    _LAYOUT = (
        "#############",
        "#     #     #",
        "#     #     #",
        "#           #",
        "#     #     #",
        "#     #     #",
        "## ####     #",
        "#     ### ###",
        "#     #     #",
        "#     #     #",
        "#           #",
        "#     #     #",
        "#############",
    )
    _DIRECTIONS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.height = len(self._LAYOUT)
        self.width = len(self._LAYOUT[0])

        self._state_id: Dict[Tuple[int, int], int] = {}
        self._id_state: Dict[int, Tuple[int, int]] = {}
        for row, line in enumerate(self._LAYOUT):
            for column, cell in enumerate(line):
                if cell == " ":
                    state = len(self._state_id)
                    self._state_id[(row, column)] = state
                    self._id_state[state] = (row, column)

        self.n_states = len(self._state_id)
        self.n_actions = 4
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.tostate = self._state_id
        self.tocell = self._id_state
        self.directions = tuple(self._DIRECTIONS[action] for action in range(4))

        self.goal_state = 62
        self.goal_position = self._id_state[self.goal_state]
        self.initial_states = tuple(state for state in range(self.n_states) if state != self.goal_state)
        self.init_states = list(self.initial_states)
        self.goal = self.goal_state

        self._state = self.goal_state
        self.current_cell = self.goal_position
        self._has_reset = False
        self._terminated = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self._state = int(self.np_random.choice(self.initial_states))
        self.current_cell = self._id_state[self._state]
        self._has_reset = True
        self._terminated = False
        return self._state, self._get_info()

    def step(self, action: int):
        self._check_ready()
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        intended_action = int(action)
        row, column = self._id_state[self._state]
        next_row = row + self._DIRECTIONS[intended_action][0]
        next_column = column + self._DIRECTIONS[intended_action][1]
        intended_position = (next_row, next_column)

        if intended_position in self._state_id:
            if self.np_random.random() < (1.0 / 3.0):
                next_position = self._sample_available_position((row, column))
            else:
                next_position = intended_position
            self._state = self._state_id[next_position]
            self.current_cell = next_position

        self._terminated = self._state == self.goal_state
        reward = 1.0 if self._terminated else 0.0
        return self._state, reward, self._terminated, False, self._get_info()

    def transition(self, state: int, action: int) -> int:
        self._validate_state(state)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        row, column = self._id_state[int(state)]
        delta_row, delta_column = self._DIRECTIONS[int(action)]
        position = (row + delta_row, column + delta_column)
        return self._state_id.get(position, int(state))

    def available_states(self, state: int) -> Tuple[int, ...]:
        self._validate_state(state)
        position = self._id_state[int(state)]
        return tuple(self._state_id[cell] for cell in self._available_positions(position))

    def check_available_cells(self, cell: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        if cell not in self._state_id:
            raise ValueError(f"Invalid cell: {cell}")
        return self._available_positions(cell)

    def render(self):
        if self.render_mode != "ansi":
            return None

        grid = [list(line) for line in self._LAYOUT]
        goal_row, goal_column = self.goal_position
        grid[goal_row][goal_column] = "G"
        if self._has_reset:
            row, column = self._id_state[self._state]
            grid[row][column] = "A"
        return "\n".join("".join(line) for line in grid) + "\n"

    def close(self):
        return None

    def _sample_available_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        available = self._available_positions(position)
        return available[int(self.np_random.integers(len(available)))]

    def _available_positions(self, position: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        row, column = position
        return tuple(
            (row + delta_row, column + delta_column)
            for delta_row, delta_column in self._DIRECTIONS.values()
            if (row + delta_row, column + delta_column) in self._state_id
        )

    def _check_ready(self):
        if not self._has_reset:
            raise RuntimeError("FourRoomsEnv.step() called before reset().")
        if self._terminated:
            raise RuntimeError("FourRoomsEnv.step() called after termination.")

    def _validate_state(self, state: int):
        if not self.observation_space.contains(state):
            raise ValueError(f"Invalid state: {state}")

    def _get_info(self):
        return {
            "position": self._id_state[self._state] if self._has_reset else None,
            "goal": self.goal_position,
            "is_success": self._terminated,
            "terminal_reason": "goal" if self._terminated else None,
        }
