import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from typing import Tuple, List, Optional

# ========== CONFIG ==========
MAP_PATH     = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"

# Chọn 1 trong 2 policy tối ưu đã lưu:
#   - Policy Iteration:   "/Users/haiduong/Desktop/Maze/policy_star_PI_12x12.npy"
#   - Value Iteration:    "/Users/haiduong/Desktop/Maze/policy_star_VI_12x12.npy"
POLICY_PATH  = "/Users/haiduong/Desktop/Maze/policy_star_VI_12x12.npy"  # đổi qua PI nếu muốn

MAX_STEPS    = 400   # tránh vòng lặp bất tận nếu map lỗi
FPS          = 10    # khung hình/giây (10 → dễ nhìn)
DOT_SIZE     = 160   # kích thước marker agent
EN_DOT_SIZE  = 140   # enemy marker
# ===========================


# --------- Lightweight Maze loader & helpers ---------
class Maze:
    def load(self, filename: str):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as f:
            tmp = pickle.load(f)
        self.__dict__.update(tmp)
        self.width  = int(getattr(self, "width",  getattr(self, "maze_size", (0,0))[0]))
        self.height = int(getattr(self, "height", getattr(self, "maze_size", (0,0))[1]))
        self.total_nodes = int(getattr(self, "total_nodes", self.width * self.height))
        return self

    def node2xy(self, n: int) -> Tuple[int, int]:
        # to pixel grid (maze map coordinates)
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


# --------- Build deterministic transition from adjacency ---------
ACTIONS = [0, 1, 2, 3]  # 0=UP,1=RIGHT,2=DOWN,3=LEFT

def next_state_from_adjacency(adj: np.ndarray, s: int, a: int, H: int, W: int) -> int:
    x, y = s // H, s % H
    ns = s
    if a == 0 and y + 1 < H:       # UP
        t = x * H + (y + 1)
        if adj[s, t] > 0: ns = t
    elif a == 1 and x + 1 < W:     # RIGHT
        t = (x + 1) * H + y
        if adj[s, t] > 0: ns = t
    elif a == 2 and y - 1 >= 0:    # DOWN
        t = x * H + (y - 1)
        if adj[s, t] > 0: ns = t
    elif a == 3 and x - 1 >= 0:    # LEFT
        t = (x - 1) * H + y
        if adj[s, t] > 0: ns = t
    return ns


# --------- Animation Controller ---------
class PolicyAnimator:
    def __init__(self, maze: Maze, policy: np.ndarray):
        self.maze   = maze
        self.policy = policy
        self.W, self.H = maze.width, maze.height
        self.N = maze.total_nodes
        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", self.W*self.H - 1)))
        self.enemies = set(getattr(maze, "static_enemies", getattr(maze, "_static_enemies", [])))
        self.adj   = maze.adjacency

        self.state = self.start
        self.steps = 0
        self.done  = False

        # canvas
        self.map_img = self._build_canvas(maze.generate_maze_map())
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = self.ax.imshow(self.map_img, origin='lower')
        self.ax.set_xticks([]); self.ax.set_yticks([])

        # scatter handles
        sx, sy = maze.node2xy(self.start); gx, gy = maze.node2xy(self.goal)
        self.start_pt  = self.ax.scatter([sx], [sy], s=DOT_SIZE, c=[[0.0, 0.4, 1.0]], marker='s', label="Start")
        self.goal_pt   = self.ax.scatter([gx], [gy], s=DOT_SIZE, c=[[1.0, 0.2, 0.2]], marker='s', label="Goal")
        if len(self.enemies) > 0:
            exs, eys = zip(*[maze.node2xy(e) for e in self.enemies])
            self.enemy_pts = self.ax.scatter(exs, eys, s=EN_DOT_SIZE, c=[[1.0, 0.6, 0.0]], marker='s', label="Enemy")
        else:
            self.enemy_pts = None

        ax0, ay0 = maze.node2xy(self.state)
        self.agent_pt = self.ax.scatter([ax0], [ay0], s=DOT_SIZE, c=[[0.0, 1.0, 0.0]], marker='o', label="Agent")

        self.text = self.ax.text(0.02, 0.98, self._status_text(), color="w",
                                 transform=self.ax.transAxes, va='top', ha='left',
                                 bbox=dict(facecolor='black', alpha=0.35, boxstyle='round,pad=0.3'))

        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_canvas(self, maze_map: np.ndarray) -> np.ndarray:
        """Tường đen, lối đi trắng — đơn giản, dễ nhìn."""
        h, w = maze_map.shape
        canvas = np.zeros((h, w, 3), dtype=float)
        walk = (maze_map == 1)
        canvas[walk] = [1.0, 1.0, 1.0]
        canvas[~walk] = [0.0, 0.0, 0.0]
        return canvas

    def _status_text(self) -> str:
        return f"State: {self.state} | Steps: {self.steps} | Done: {self.done}"

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key in ('r', 'R'):
            self.reset()

    def reset(self):
        self.state = self.start
        self.steps = 0
        self.done  = False
        ax, ay = self.maze.node2xy(self.state)
        self.agent_pt.set_offsets([[ax, ay]])
        self.text.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _tick(self):
        if self.done: return
        a = int(self.policy[self.state]) if 0 <= self.state < self.N else 0
        ns = next_state_from_adjacency(self.adj, self.state, a, self.maze.height, self.maze.width)
        self.state = ns
        self.steps += 1
        if self.state == self.goal or self.state in self.enemies or self.steps >= MAX_STEPS:
            self.done = True

    def _update_artist(self):
        ax, ay = self.maze.node2xy(self.state)
        self.agent_pt.set_offsets([[ax, ay]])
        self.text.set_text(self._status_text())

    def anim_step(self, frame):
        if not self.paused and not self.done:
            self._tick()
        self._update_artist()
        return (self.agent_pt, self.text)

    def run(self, fps: int = 10):
        self.ani = animation.FuncAnimation(
            self.fig, self.anim_step, interval=1000//max(1, fps), blit=False)
        plt.title("Policy Rollout (space: pause/resume, r: reset)")
        plt.legend(loc='upper right')
        plt.show()


# --------- MAIN ---------
if __name__ == "__main__":
    # Load maze & policy
    maze = Maze().load(MAP_PATH)
    policy = np.load(POLICY_PATH)
    assert policy.shape[0] == maze.total_nodes, f"Policy shape mismatch: {policy.shape} vs N={maze.total_nodes}"

    gui = PolicyAnimator(maze, policy)
    gui.run(fps=FPS)
