# agent.py
import random
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Condition-action reflex agent with no internal history."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Suck'
        if percept.get('wall_ahead'):
            return 'Left'
        return 'Forward'


class ModelBasedAgent:
    """Reflex agent with internal state for escaping local loops."""

    def __init__(self):
        self.position = (0, 0)
        self.facing = 'Up'
        self.visited_cells = {self.position}
        self.last_action = None
        self.last_percept = None
        self.wall_block_count = 0

    @staticmethod
    def _turn_left(facing: str) -> str:
        order = ['Up', 'Left', 'Down', 'Right']
        return order[(order.index(facing) + 1) % 4]

    @staticmethod
    def _turn_right(facing: str) -> str:
        order = ['Up', 'Right', 'Down', 'Left']
        return order[(order.index(facing) + 1) % 4]

    @staticmethod
    def _move(position, facing):
        x, y = position
        if facing == 'Up':
            return x, y + 1
        if facing == 'Down':
            return x, y - 1
        if facing == 'Left':
            return x - 1, y
        return x + 1, y

    def _relative_direction(self, turn: str) -> str:
        if turn == 'left':
            return self._turn_left(self.facing)
        if turn == 'right':
            return self._turn_right(self.facing)
        return self.facing

    def _update_state(self, percept: dict) -> None:
        if self.last_action == 'Forward' and self.last_percept and not self.last_percept.get('wall_ahead'):
            self.position = self._move(self.position, self.facing)
        elif self.last_action == 'TurnLeft':
            self.facing = self._turn_left(self.facing)
        elif self.last_action == 'TurnRight':
            self.facing = self._turn_right(self.facing)

        self.visited_cells.add(self.position)
        self.last_percept = dict(percept)
        if percept.get('wall_ahead'):
            self.wall_block_count += 1
        else:
            self.wall_block_count = 0

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)

        if percept.get('food_here'):
            action = 'Suck'
        elif percept.get('wall_ahead'):
            left_cell = self._move(self.position, self._relative_direction('left'))
            right_cell = self._move(self.position, self._relative_direction('right'))

            if self.last_action in ('TurnLeft', 'TurnRight') and self.wall_block_count > 0:
                action = 'TurnRight' if self.last_action == 'TurnLeft' else 'TurnLeft'
            elif right_cell not in self.visited_cells:
                action = 'TurnRight'
            elif left_cell not in self.visited_cells:
                action = 'TurnLeft'
            elif self.wall_block_count % 2 == 0:
                action = 'TurnLeft'
            else:
                action = 'TurnRight'
        else:
            forward_cell = self._move(self.position, self.facing)
            left_cell = self._move(self.position, self._relative_direction('left'))
            right_cell = self._move(self.position, self._relative_direction('right'))

            if forward_cell not in self.visited_cells:
                action = 'Forward'
            elif left_cell not in self.visited_cells:
                action = 'TurnLeft'
            elif right_cell not in self.visited_cells:
                action = 'TurnRight'
            else:
                action = 'Forward'

        self.last_action = action
        return action


class SearchAgent:
    """Breadth-first search agent for the static grid tests."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        blocked = set(walls)
        frontier = deque([(start_pos, [])])
        visited = {start_pos}
        moves = [
            ('Up', (0, 1)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
            ('Right', (1, 0)),
        ]

        while frontier:
            position, path = frontier.popleft()
            if position == goal_pos:
                return path

            for action, (dx, dy) in moves:
                next_pos = (position[0] + dx, position[1] + dy)
                if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    continue
                if next_pos in blocked or next_pos in visited:
                    continue

                visited.add(next_pos)
                frontier.append((next_pos, path + [action]))

        return None