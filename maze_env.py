import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time
from typing import Tuple
from collections import deque  # dùng cho BFS tính khoảng cách

# ======================== LIGHTWEIGHT MAZE LOADER ========================
class Maze:
    """Loader nhẹ cho .pkl đã lưu từ class Maze của anh."""

    def load(self, filename: str):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

        self.width = int(getattr(self, "width", getattr(self, "maze_size", (0, 0))[0]))
        self.height = int(getattr(self, "height", getattr(self, "maze_size", (0, 0))[1]))
        self.total_nodes = int(getattr(self, "total_nodes", self.width * self.height))
        return self

    def node2xy(self, n: int) -> Tuple[int, int]:
        return 2 * (n // self.height), 2 * (n % self.height)

    def generate_maze_map(self) -> np.ndarray:
        maze_map = np.zeros([2 * self.height - 1, 2 * self.width - 1], dtype='int')
        for j in range(self.width - 1):
            for i in range(self.height - 1):
                if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                    maze_map[2 * i + 1, 2 * j] = 1
                if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                    maze_map[2 * i, 2 * j + 1] = 1
            i = self.height - 1
            if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                maze_map[2 * i, 2 * j + 1] = 1
        j = self.width - 1
        for i in range(self.height - 1):
            if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                maze_map[2 * i + 1, 2 * j] = 1
        for n in range(self.width * self.height):
            x, y = self.node2xy(n)
            maze_map[y, x] = 1
        return maze_map


# ======================== ENV (GYM-LIKE) ========================
class MazeEnv:
    """
    - State: node index (0..N-1)
    - Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

    Reward (Cách 1 - nới hệ số):
        GOAL_REWARD    = +20.0
        ENEMY_PENALTY  = -10.0
        WALL_PENALTY   = -0.5   (đụng tường/không di chuyển)
        STEP_PENALTY   = -0.1   (mỗi bước)

    Reward shaping (Cách 3 - Potential-based):
        Φ(s) = -dist_to_goal(s) (tính bằng BFS)
        reward += SHAPING_LAMBDA * (GAMMA * Φ(s') - Φ(s))
    """

    # ==== Cách 1: thông số thưởng/phạt "dịu" để tổng điểm dễ dương ====
    STEP_PENALTY   = -0.1
    WALL_PENALTY   = -0.5
    GOAL_REWARD    = +20.0
    ENEMY_PENALTY  = -10.0

    # ==== Cách 3: Potential-based shaping ====
    SHAPING_LAMBDA = 1.0   # 1.0 = bật shaping; 0.0 = tắt shaping
    GAMMA          = 0.98  # dùng cùng gamma với agent cho đẹp

    def __init__(self, maze: Maze):
        self.maze = maze
        self.width = int(maze.width)
        self.height = int(maze.height)
        self.total_nodes = int(maze.total_nodes)

        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal = int(getattr(maze, "sinkerNode",
                                getattr(maze, "_sinkerNode", self.width * self.height - 1)))
        self.enemies = set(getattr(maze, "static_enemies",
                                   getattr(maze, "_static_enemies", [])))

        self.adjacency = maze.adjacency
        self.action_space = [0, 1, 2, 3]
        self.n_actions = 4
        self.state = self.start
        self.steps = 0

        # (tùy chọn) phục vụ Cách 2 nếu dùng speed bonus sau này
        self.max_steps_per_ep = 450

        # build nền vẽ
        self._maze_map = self.maze.generate_maze_map()
        self._canvas = self._build_canvas(self._maze_map)

        # ==== Cách 3: tính khoảng cách BFS tới goal 1 lần ====
        self._dist_to_goal = self._bfs_distance_to_goal()

    def _build_canvas(self, maze_map: np.ndarray) -> np.ndarray:
        """Tạo canvas: lối đi trắng, tường đen."""
        h, w = maze_map.shape
        canvas = np.zeros((h, w, 3), dtype=float)
        walk = (maze_map == 1)
        canvas[walk] = [1.0, 1.0, 1.0]
        canvas[~walk] = [0.0, 0.0, 0.0]
        return canvas

    # ==== BFS tính khoảng cách tới goal (cho Potential Φ) ====
    def _bfs_distance_to_goal(self) -> np.ndarray:
        n = self.total_nodes
        dist = np.full(n, np.inf, dtype=float)
        g = self.goal
        dist[g] = 0.0
        q = deque([g])
        while q:
            u = q.popleft()
            nbrs = np.where(self.adjacency[u] > 0)[0]
            for v in nbrs:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1.0
                    q.append(v)
        return dist

    def reset(self) -> int:
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int):
        self.steps += 1
        next_state = self._move(self.state, action)

        # ==== Cách 1: base reward dịu ====
        reward = self.STEP_PENALTY
        done = False
        if next_state == self.state:
            reward = self.WALL_PENALTY
        if next_state in self.enemies:
            reward, done = self.ENEMY_PENALTY, True
        elif next_state == self.goal:
            reward, done = self.GOAL_REWARD, True

        # ==== Cách 3: Potential-based shaping ====
        if self.SHAPING_LAMBDA != 0.0 and self._dist_to_goal is not None:
            # Φ(s) = -dist_to_goal(s)
            phi_s  = -self._dist_to_goal[self.state] if np.isfinite(self._dist_to_goal[self.state]) else 0.0
            phi_sp = -self._dist_to_goal[next_state]  if np.isfinite(self._dist_to_goal[next_state]) else 0.0
            reward += self.SHAPING_LAMBDA * (self.GAMMA * phi_sp - phi_s)

        self.state = next_state
        return next_state, float(reward), done, {}

    def _move(self, state: int, action: int) -> int:
        x, y = state // self.height, state % self.height
        ns = state
        if action == 0 and y + 1 < self.height:
            t = x * self.height + (y + 1)
            if self.adjacency[state, t] > 0: ns = t
        elif action == 1 and x + 1 < self.width:
            t = (x + 1) * self.height + y
            if self.adjacency[state, t] > 0: ns = t
        elif action == 2 and y - 1 >= 0:
            t = x * self.height + (y - 1)
            if self.adjacency[state, t] > 0: ns = t
        elif action == 3 and x - 1 >= 0:
            t = (x - 1) * self.height + y
            if self.adjacency[state, t] > 0: ns = t
        return ns

    def render(self):
        img = self._canvas.copy()
        sx, sy = self.maze.node2xy(self.start)
        gx, gy = self.maze.node2xy(self.goal)
        img[sy, sx] = [0.0, 0.4, 1.0]
        img[gy, gx] = [1.0, 0.2, 0.2]
        for e in self.enemies:
            ex, ey = self.maze.node2xy(e)
            img[ey, ex] = [1.0, 0.6, 0.0]
        ax, ay = self.maze.node2xy(self.state)
        img[ay, ax] = [0.0, 1.0, 0.0]
        plt.imshow(img, origin='lower')
        plt.title(f"Step {self.steps} | State {self.state}")
        plt.xticks([]); plt.yticks([])
        plt.draw()
        plt.pause(0.001)


# ======================== MEDIUM-SPEED RANDOM AGENT ========================
def random_agent_demo_medium(env: MazeEnv, max_steps: int = 300, step_delay: float = 0.06, show_every: int = 1):
    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.draw()
    env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done and env.steps < max_steps:
        if step % show_every == 0:
            env.render()
            time.sleep(step_delay)
        a = random.choice(env.action_space)
        _, r, done, _ = env.step(a)
        total_reward += r
        step += 1

    env.render()
    plt.pause(0.3)
    plt.ioff()
    plt.close('all')
    print(f"Total reward: {total_reward:.1f}, Steps: {env.steps}")


# ============================= MAIN =================================
if __name__ == "__main__":
    MAP_PATH = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"
    maze = Maze().load(MAP_PATH)

    start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
    goal = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", maze.width * maze.height - 1)))
    enemies = set(getattr(maze, "static_enemies", getattr(maze, "_static_enemies", [])))
    print(f"Maze loaded: {maze.width}x{maze.height}, start={start}, goal={goal}, enemies={sorted(list(enemies))}")

    env = MazeEnv(maze)
    print("Running MEDIUM Random Agent demo (one episode)...")
    random_agent_demo_medium(env, max_steps=50, step_delay=0.06, show_every=1)
