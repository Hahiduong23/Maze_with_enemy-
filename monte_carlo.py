import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time
from typing import Tuple, List, Dict, Any

# ==================== LIGHTWEIGHT MAZE & ENV (tối giản, tương thích) ====================
class Maze:
    def load(self, filename: str):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        self.width  = int(getattr(self, "width",  getattr(self, "maze_size", (0,0))[0]))
        self.height = int(getattr(self, "height", getattr(self, "maze_size", (0,0))[1]))
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


class MazeEnv:
    """
    - State: node index (0..N-1)
    - Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    - Reward đã chỉnh trong env chính (Cách 1 + shaping Cách 3).
      Ở đây để fallback tối thiểu để train độc lập.
    """
    def __init__(self, maze: Maze):
        self.maze = maze
        self.width = int(maze.width)
        self.height = int(maze.height)
        self.total_nodes = int(maze.total_nodes)

        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode",
                                 getattr(maze, "_sinkerNode", self.width*self.height - 1)))
        self.enemies = set(getattr(maze, "static_enemies",
                                   getattr(maze, "_static_enemies", [])))

        self.adjacency = maze.adjacency
        self.action_space = [0, 1, 2, 3]
        self.n_actions = 4

        self.state = self.start
        self.steps = 0

        # giữ gamma mặc định đồng bộ shaping nếu env thực tế có
        self.GAMMA = getattr(self, "GAMMA", 0.98)

    def reset(self) -> int:
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int):
        self.steps += 1
        next_state = self._move(self.state, action)

        STEP_PENALTY  = getattr(self, "STEP_PENALTY",  -0.1)
        WALL_PENALTY  = getattr(self, "WALL_PENALTY",  -0.5)
        GOAL_REWARD   = getattr(self, "GOAL_REWARD",   20.0)
        ENEMY_PENALTY = getattr(self, "ENEMY_PENALTY", -10.0)

        reward = STEP_PENALTY
        done = False
        if next_state == self.state:
            reward = WALL_PENALTY
        if next_state in self.enemies:
            reward, done = ENEMY_PENALTY, True
        elif next_state == self.goal:
            reward, done = GOAL_REWARD, True

        self.state = next_state
        return next_state, reward, done, {}

    def _move(self, state: int, action: int) -> int:
        x, y = state // self.height, state % self.height
        ns = state
        if action == 0 and y + 1 < self.height:        # UP
            t = x * self.height + (y + 1)
            if self.adjacency[state, t] > 0: ns = t
        elif action == 1 and x + 1 < self.width:       # RIGHT
            t = (x + 1) * self.height + y
            if self.adjacency[state, t] > 0: ns = t
        elif action == 2 and y - 1 >= 0:               # DOWN
            t = x * self.height + (y - 1)
            if self.adjacency[state, t] > 0: ns = t
        elif action == 3 and x - 1 >= 0:               # LEFT
            t = (x - 1) * self.height + y
            if self.adjacency[state, t] > 0: ns = t
        return ns

# =============================== MONTE CARLO (First-Visit) ===============================
class MonteCarloAgent:
    """
    On-policy epsilon-soft First-Visit MC Control.
    - Q: shape [n_states, n_actions]
    - Returns sum / count cho first-visit trung bình dần
    - Epsilon decay theo episode
    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 800,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.returns_sum = np.zeros_like(self.Q, dtype=np.float64)
        self.returns_count = np.zeros_like(self.Q, dtype=np.int64)

    def epsilon_by_episode(self, ep: int) -> float:
        frac = min(1.0, ep / self.epsilon_decay_episodes)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update_from_episode(self, episode: List[Tuple[int,int,float]], first_visit: bool = True):
        """
        episode: list of (s_t, a_t, r_t) theo thời gian.
        Dùng first-visit: chỉ cập nhật cặp (s,a) lần xuất hiện đầu tiên.
        G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
        """
        G = 0.0
        visited = set()
        # duyệt ngược để tính return tích lũy
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = self.gamma * G + r_t

            key = (s_t, a_t)
            if first_visit:
                if key in visited:
                    continue
                visited.add(key)

            # cập nhật trung bình dần
            self.returns_sum[s_t, a_t] += G
            self.returns_count[s_t, a_t] += 1
            self.Q[s_t, a_t] = self.returns_sum[s_t, a_t] / max(1, self.returns_count[s_t, a_t])

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

# ================================== TRAIN / EVAL ==================================
def generate_episode(env: MazeEnv, agent: MonteCarloAgent, max_steps: int, epsilon: float):
    """Sinh một episode theo epsilon-soft chính sách của agent."""
    s = env.reset()
    episode = []
    for _ in range(max_steps):
        a = agent.act(s, epsilon)
        s_next, r, done, _ = env.step(a)
        episode.append((s, a, r))
        s = s_next
        if done:
            break
    return episode, s  # trả kèm state cuối để xét success

def train_mc_control(
    env: MazeEnv,
    episodes: int = 1500,
    max_steps_per_ep: int = 400,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 900,
    seed: int = 123,
):
    random.seed(seed); np.random.seed(seed)
    if hasattr(env, "GAMMA"):
        assert abs(gamma - env.GAMMA) < 1e-6, f"GAMMA mismatch: agent {gamma} vs env {env.GAMMA}"

    agent = MonteCarloAgent(
        n_states=env.total_nodes,
        n_actions=env.n_actions,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes
    )

    rewards_per_ep, steps_per_ep, success_window = [], [], []

    for ep in range(1, episodes + 1):
        eps = agent.epsilon_by_episode(ep)
        episode, last_state = generate_episode(env, agent, max_steps_per_ep, eps)
        # tổng reward/steps của episode
        total_r = float(np.sum([r for (_, _, r) in episode]))
        rewards_per_ep.append(total_r)
        steps_per_ep.append(len(episode))
        success_window.append(1 if last_state == env.goal else 0)

        # cập nhật first-visit
        agent.update_from_episode(episode, first_visit=True)

        if ep % 50 == 0:
            sr = np.mean(success_window[-50:]) * 100.0
            print(f"Ep {ep:4d} | eps={eps:0.3f} | reward={total_r:6.1f} | steps={len(episode):3d} | success@50={sr:5.1f}%")

    return agent, np.array(rewards_per_ep), np.array(steps_per_ep)

def evaluate_greedy(env: MazeEnv, agent: MonteCarloAgent, episodes: int = 40, max_steps_per_ep: int = 400):
    total_rewards, steps_list, successes = [], [], 0
    for _ in range(episodes):
        s = env.reset()
        total_r = 0.0
        for _t in range(max_steps_per_ep):
            a = agent.greedy_action(s)
            s, r, done, _ = env.step(a)
            total_r += r
            if done:
                break
        total_rewards.append(total_r)
        steps_list.append(env.steps)
        if s == env.goal:
            successes += 1
    success_rate = successes / episodes * 100.0
    return np.array(total_rewards), np.array(steps_list), success_rate

# ================================== MAIN ==================================
if __name__ == "__main__":
    # Map 12x12
    MAP_PATH = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"

    maze = Maze().load(MAP_PATH)
    env = MazeEnv(maze)

    print(f"Maze loaded: {maze.width}x{maze.height}, start={getattr(maze,'startNode',getattr(maze,'_startNode',0))}, "
          f"goal={getattr(maze,'sinkerNode',getattr(maze,'_sinkerNode',maze.width*maze.height-1))}, "
          f"enemies={sorted(list(getattr(maze,'static_enemies',getattr(maze,'_static_enemies',[]))))}")

    # ======== HYPERPARAMETERS (MC control) ========
    EPISODES = 1500           # MC thường cần nhiều ep hơn TD
    MAX_STEPS = 400           # 350–450
    GAMMA = 0.98              # khớp env.GAMMA
    EPS_START = 1.0
    EPS_END   = 0.05
    EPS_DECAY_EPISODES = 900  # decay dài hơn Q/SARSA một chút

    # ======== TRAIN ========
    t0 = time.time()
    agent, rewards, steps = train_mc_control(
        env,
        episodes=EPISODES,
        max_steps_per_ep=MAX_STEPS,
        gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay_episodes=EPS_DECAY_EPISODES,
        seed=123
    )
    t1 = time.time()
    print(f"Training done in {t1 - t0:.2f}s")

    # ======== SAVE Q-TABLE ========
    out_q = "/Users/haiduong/Desktop/Maze/qtable_12x12_mc.npy"
    np.save(out_q, agent.Q)
    print(f"Saved Q-table: {out_q} | shape={agent.Q.shape}")

    # ======== EVALUATE (greedy) ========
    eval_rewards, eval_steps, success = evaluate_greedy(env, agent, episodes=40, max_steps_per_ep=MAX_STEPS)
    print(f"Greedy evaluation: success={success:.1f}% | avg_reward={eval_rewards.mean():.2f} | avg_steps={eval_steps.mean():.1f}")

    # ======== PLOTS ========
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    window = 50
    if len(rewards) >= window:
        mv = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, window-1+len(mv)), mv, label=f"Reward MA({window})")
    plt.plot(rewards, alpha=0.3, linewidth=0.8, label="Reward (raw)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("Training Reward (Monte Carlo)"); plt.legend()

    plt.subplot(1,2,2)
    if len(steps) >= window:
        mv_s = np.convolve(steps, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, window-1+len(mv_s)), mv_s, label=f"Steps MA({window})")
    plt.plot(steps, alpha=0.3, linewidth=0.8, label="Steps (raw)")
    plt.xlabel("Episode"); plt.ylabel("Steps"); plt.title("Episode length"); plt.legend()

    plt.tight_layout()
    plt.show()
