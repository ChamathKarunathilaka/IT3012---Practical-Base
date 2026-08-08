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
            # Turn in-place when a wall is detected ahead (no internal memory)
            return 'TurnLeft'
        return 'Forward'


class ModelBasedAgent:
    """Simple model-based agent that records repeated percepts and breaks loops.

    This implementation intentionally keeps a lightweight memory (no global coords)
    suitable for partially observable worlds: it tracks the last percept and the
    number of times the same percept was seen consecutively and switches actions
    when stuck.
    """

    def __init__(self):
        self.last_percept = None
        self.last_action = None
        self.repeat_count = 0

    def sense_and_act(self, percept: dict) -> str:
        # Update repeat_count for identical percepts (detect being stuck)
        if self.last_percept is not None and percept == self.last_percept:
            self.repeat_count += 1
        else:
            self.repeat_count = 0

        if percept.get('food_here'):
            action = 'Suck'
        elif percept.get('wall_ahead'):
            # If we've seen the same wall percept repeatedly, try to change behaviour
            if self.repeat_count >= 1:
                # Alternate between turning and attempting to move forward
                if self.last_action == 'TurnLeft':
                    action = 'Forward'
                else:
                    action = 'TurnLeft'
            else:
                action = 'TurnLeft'
        else:
            action = 'Forward'

        self.last_percept = dict(percept)
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