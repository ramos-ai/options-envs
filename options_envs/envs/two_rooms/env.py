from __future__ import annotations

from typing import Dict, Set, Tuple

import gymnasium as gym
from gymnasium import spaces


class TwoRoomsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    _GRID = (
        "###############",
        "#......#......#",
        "#......#......#",
        "#.............#",
        "#......#......#",
        "#......#......#",
        "#......#......#",
        "###############",
    )
    _DIRECTIONS = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
    }

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.height = len(self._GRID)
        self.width = len(self._GRID[0])
        self._state_id: Dict[Tuple[int, int], int] = {}
        self._id_state: Dict[int, Tuple[int, int]] = {}
        for y, line in enumerate(self._GRID):
            for x, cell in enumerate(line):
                if cell == ".":
                    state = len(self._state_id)
                    self._state_id[(x, y)] = state
                    self._id_state[state] = (x, y)

        self.n_states = len(self._state_id)
        self.n_actions = 4
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.tostate = self._state_id
        self.tocell = self._id_state

        self.start_state = self._state_id[(1, 3)]
        self.hallway_state = self._state_id[(7, 3)]
        self.goal_state = self._state_id[(10, 6)]

        self.gray_region: Set[int] = {
            state
            for (x, y), state in self._state_id.items()
            if 2 <= x <= 5 and 1 <= y <= 5
        }
        self.gray_region.discard(self.start_state)

        self._state = self.start_state
        self._has_reset = False
        self._terminated = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self._state = self.start_state
        self._has_reset = True
        self._terminated = False
        return self._state, self._get_info()

    def step(self, action: int):
        self._check_ready()
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self._state = self.transition(self._state, int(action))
        self._terminated = self._state == self.goal_state
        reward = self.base_reward_for_next_state(self._state)
        return self._state, reward, self._terminated, False, self._get_info()

    def transition(self, state: int, action: int) -> int:
        self._validate_state(state)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        x, y = self._id_state[int(state)]
        delta_x, delta_y = self._DIRECTIONS[int(action)]
        position = (x + delta_x, y + delta_y)
        return self._state_id.get(position, int(state))

    def base_reward_for_next_state(self, next_state: int) -> float:
        self._validate_state(next_state)
        if int(next_state) == self.goal_state:
            return 1.0
        if int(next_state) in self.gray_region:
            return -1.0
        return 0.0

    def render(self):
        if self.render_mode != "ansi":
            return None

        grid = [list(line) for line in self._GRID]
        for state in self.gray_region:
            x, y = self._id_state[state]
            grid[y][x] = "~"
        goal_x, goal_y = self._id_state[self.goal_state]
        grid[goal_y][goal_x] = "G"
        if self._has_reset:
            x, y = self._id_state[self._state]
            grid[y][x] = "A"
        return "\n".join("".join(line) for line in grid) + "\n"

    def close(self):
        return None

    def _check_ready(self):
        if not self._has_reset:
            raise RuntimeError("TwoRoomsEnv.step() called before reset().")
        if self._terminated:
            raise RuntimeError("TwoRoomsEnv.step() called after termination.")

    def _validate_state(self, state: int):
        if not self.observation_space.contains(state):
            raise ValueError(f"Invalid state: {state}")

    def _get_info(self):
        return {
            "position": self._id_state[self._state] if self._has_reset else None,
            "goal": self._id_state[self.goal_state],
            "room_id": self.room_id(self._state) if self._has_reset else None,
            "is_success": self._terminated,
            "terminal_reason": "goal" if self._terminated else None,
        }

    def room_id(self, state: int) -> str:
        self._validate_state(state)
        x, _ = self._id_state[int(state)]
        if x <= 6:
            return "left"
        if x == 7:
            return "hallway"
        return "right"
