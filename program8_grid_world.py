import random

class GridWorld:
    def __init__(self, width, height, start, goal, obstacles=None):
        if width < 1 or height < 1:
            raise ValueError('Grid width and height must be positive.')
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        for position in [start, goal, *self.obstacles]:
            x, y = position
            if not (0 <= x < height and 0 <= y < width):
                raise ValueError(f'Position {position} is outside the grid.')
        if start == goal or start in self.obstacles or goal in self.obstacles:
            raise ValueError('Start and goal must be distinct and cannot be obstacles.')
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def render(self):
        for i in range(self.height):
            row = []
            for j in range(self.width):
                position = (i, j)
                if position == self.agent_pos:
                    row.append('A')
                elif position == self.start:
                    row.append('S')
                elif position == self.goal:
                    row.append('G')
                elif position in self.obstacles:
                    row.append('O')
                else:
                    row.append('.')
            print(' '.join(row))
        print()

    def step(self, action):
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        if action not in moves:
            raise ValueError('Invalid action. Use: up, down, left, right.')
        x, y = self.agent_pos
        dx, dy = moves[action]
        candidate = (x + dx, y + dy)
        if 0 <= candidate[0] < self.height and 0 <= candidate[1] < self.width:
            self.agent_pos = candidate
        position = self.agent_pos
        if position == self.goal:
            return position, 1, True
        if position in self.obstacles:
            return position, -1, True
        return position, -0.01, False

width, height = 5, 5
start, goal = (0, 0), (4, 4)
all_positions = [(i, j) for i in range(height) for j in range(width) if (i, j) not in (start, goal)]
obstacles = random.sample(all_positions, 3)
env = GridWorld(width, height, start, goal, obstacles)
state = env.reset()
done = False
print('Welcome to Grid World!')
print('Reach G to win, avoid O, and move using up, down, left, right.')
env.render()
while not done:
    try:
        action = input('Enter your move (up, down, left, right): ').strip().lower()
        state, reward, done = env.step(action)
    except ValueError as error:
        print(error)
        continue
    print(f'New State: {state}, Reward: {reward}')
    env.render()
if state == goal:
    print('Congratulations! You reached the goal!')
else:
    print('Game over! You hit an obstacle.')
