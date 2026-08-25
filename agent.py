# agent.py
import heapq
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
    """Graph-search agent that plans a route to the nearest food pellet."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    @staticmethod
    def _moves():
        return [
            ('Up', (0, 1)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
            ('Right', (1, 0)),
        ]

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        blocked = set(walls)
        frontier = deque([(start_pos, [])])
        reached = {start_pos}

        while frontier:
            position, path = frontier.popleft()
            if position == goal_pos:
                return path

            for action, (dx, dy) in self._moves():
                next_pos = (position[0] + dx, position[1] + dy)
                if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    continue
                if next_pos in blocked or next_pos in reached:
                    continue

                reached.add(next_pos)
                frontier.append((next_pos, path + [action]))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        blocked = set(walls)
        frontier = [(start_pos, [])]
        reached = {start_pos}

        while frontier:
            position, path = frontier.pop()
            if position == goal_pos:
                return path

            for action, (dx, dy) in reversed(self._moves()):
                next_pos = (position[0] + dx, position[1] + dy)
                if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    continue
                if next_pos in blocked or next_pos in reached:
                    continue

                reached.add(next_pos)
                frontier.append((next_pos, path + [action]))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        width, height = grid_size
        blocked = set(walls)
        frontier = [(0, start_pos, [])]
        reached = {start_pos: 0}

        while frontier:
            cost, position, path = heapq.heappop(frontier)
            if position == goal_pos:
                return path

            for action, (dx, dy) in self._moves():
                next_pos = (position[0] + dx, position[1] + dy)
                if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    continue
                if next_pos in blocked:
                    continue

                new_cost = cost + 1
                previous_best = reached.get(next_pos)
                if previous_best is not None and new_cost >= previous_best:
                    continue

                reached[next_pos] = new_cost
                heapq.heappush(frontier, (new_cost, next_pos, path + [action]))

        return None

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start_pos = tuple(percept.get('agent_pos', (0, 0)))
            food_positions = percept.get('all_food', [])
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', (0, 0))

            if food_positions:
                goal = min(food_positions, key=lambda pos: abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1]))
                algo = self.active_algo.lower()
                if algo == 'bfs':
                    self.plan = self.bfs_search(start_pos, goal, walls, grid_size) or []
                elif algo == 'dfs':
                    self.plan = self.dfs_search(start_pos, goal, walls, grid_size) or []
                elif algo == 'ucs':
                    self.plan = self.ucs_search(start_pos, goal, walls, grid_size) or []
                else:
                    self.plan = self.bfs_search(start_pos, goal, walls, grid_size) or []

        if self.plan:
            return self.plan.pop(0)

        if percept.get('food_here'):
            return 'Suck'
        return 'Forward'