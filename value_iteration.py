import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from typing import Tuple

# ======================== MAZE LOADER (nhẹ) ========================
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

# ======================== BUILD MDP FROM MAZE ========================
class MazeMDP:
    """
    Deterministic MDP:
    - States: 0..N-1
    - Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
    - P[s,a] = s' (nếu tường -> s giữ nguyên)
    - R[s,a]: step -0.1; wall -0.5; goal +20; enemy -10
    - Terminal: goal & enemies (absorbing, reward 0 sau khi vào)
    """
    ACTIONS = [0,1,2,3]

    def __init__(self, maze: Maze):
        self.maze = maze
        self.N = maze.total_nodes
        self.W = maze.width
        self.H = maze.height
        self.adj = maze.adjacency

        self.start  = int(getattr(maze, "startNode",  getattr(maze, "_startNode", 0)))
        self.goal   = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", self.W*self.H-1)))
        self.enemies = set(getattr(maze,"static_enemies", getattr(maze,"_static_enemies", [])))

        self.STEP_PENALTY  = -0.1
        self.WALL_PENALTY  = -0.5
        self.GOAL_REWARD   = +20.0
        self.ENEMY_PENALTY = -10.0
        self.GAMMA = 0.9

        self.P = np.zeros((self.N, 4), dtype=np.int32)    # next state
        self.R = np.zeros((self.N, 4), dtype=np.float64)  # reward
        self.terminal = np.zeros(self.N, dtype=bool)

        for s in range(self.N):
            if s == self.goal or s in self.enemies:
                self.terminal[s] = True

        for s in range(self.N):
            x, y = s // self.H, s % self.H
            if self.terminal[s]:
                for a in self.ACTIONS:
                    self.P[s, a] = s
                    self.R[s, a] = 0.0
                continue

            for a in self.ACTIONS:
                ns = s
                if a == 0 and y + 1 < self.H:       # UP
                    cand = x * self.H + (y + 1)
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 1 and x + 1 < self.W:     # RIGHT
                    cand = (x + 1) * self.H + y
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 2 and y - 1 >= 0:         # DOWN
                    cand = x * self.H + (y - 1)
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 3 and x - 1 >= 0:         # LEFT
                    cand = (x - 1) * self.H + y
                    if self.adj[s, cand] > 0: ns = cand

                self.P[s, a] = ns
                if ns == s:
                    r = self.WALL_PENALTY
                elif ns in self.enemies:
                    r = self.ENEMY_PENALTY
                elif ns == self.goal:
                    r = self.GOAL_REWARD
                else:
                    r = self.STEP_PENALTY
                self.R[s, a] = r

# ======================== VALUE ITERATION ========================
def value_iteration(mdp: MazeMDP, theta: float = 1e-6, max_iter: int = 100000):
    """
    Bellman optimality:
    V_{k+1}(s) = max_a [ R[s,a] + γ V_k(P[s,a]) ]  (terminal: V=0)
    """
    V = np.zeros(mdp.N, dtype=np.float64)
    gamma = mdp.GAMMA

    for it in range(1, max_iter+1):
        delta = 0.0
        for s in range(mdp.N):
            if mdp.terminal[s]:
                continue
            q_vals = mdp.R[s] + gamma * V[ mdp.P[s] ]   # shape (4,)
            v_new = np.max(q_vals)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            # print(f"[VI] Converged at iter {it} | delta={delta:.3e}")
            break
    # derive greedy policy
    policy = np.zeros(mdp.N, dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]:
            policy[s] = 0
        else:
            q_vals = mdp.R[s] + gamma * V[ mdp.P[s] ]
            policy[s] = int(np.argmax(q_vals))
    return V, policy

# ======================== EVALUATION (rollout greedy) ========================
def rollout_greedy(mdp: MazeMDP, policy: np.ndarray, episodes: int = 40, max_steps: int = 400) -> Tuple[np.ndarray, np.ndarray, float]:
    total_rewards, steps_list, success = [], [], 0
    for _ in range(episodes):
        s = mdp.start
        total_r = 0.0
        for t in range(max_steps):
            if s == mdp.goal or s in mdp.enemies:
                break
            a = policy[s]
            ns = mdp.P[s, a]
            r  = mdp.R[s, a]
            total_r += r
            s = ns
            if s == mdp.goal or s in mdp.enemies:
                break
        total_rewards.append(total_r)
        steps_list.append(t + 1)
        if s == mdp.goal:
            success += 1
    return np.array(total_rewards), np.array(steps_list), success / episodes * 100.0

# ======================== VISUALS ========================
def node2yx(n, H):
    x, y = n // H, n % H
    return y, x

def policy_to_grid(policy: np.ndarray, V: np.ndarray, W: int, H: int):
    V_grid = np.zeros((H, W), dtype=float)
    A_grid = np.zeros((H, W), dtype=int)
    for s in range(W * H):
        y, x = node2yx(s, H)
        V_grid[y, x] = V[s]
        A_grid[y, x] = policy[s]
    return V_grid, A_grid

def arrows_from_policy(ax, A_grid):
    H, W = A_grid.shape
    vec = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}  # imshow default axes
    for y in range(H):
        for x in range(W):
            a = int(A_grid[y,x])
            dx, dy = vec.get(a,(0,0))
            ax.arrow(x, y, dx*0.35, dy*0.35, head_width=0.2, head_length=0.2,
                     fc='k', ec='k', length_includes_head=True)

# ======================== MAIN ========================
if __name__ == "__main__":
    MAP_PATH = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"

    maze = Maze().load(MAP_PATH)
    mdp = MazeMDP(maze)

    print(f"Maze loaded: {maze.width}x{maze.height}")
    print(f"Start={mdp.start}, Goal={mdp.goal}, Enemies={sorted(list(mdp.enemies))}")
    print(f"Gamma={mdp.GAMMA}, Rewards: step={mdp.STEP_PENALTY}, wall={mdp.WALL_PENALTY}, goal={mdp.GOAL_REWARD}, enemy={mdp.ENEMY_PENALTY}")

    # ---- Value Iteration ----
    t0 = time.time()
    V, policy = value_iteration(mdp, theta=1e-7, max_iter=200000)
    t1 = time.time()
    print(f"Value Iteration done in {t1 - t0:.2f}s")

    # Save
    out_policy = "/Users/haiduong/Desktop/Maze/policy_star_VI_12x12.npy"
    out_value  = "/Users/haiduong/Desktop/Maze/V_star_VI_12x12.npy"
    np.save(out_policy, policy)
    np.save(out_value, V)
    print(f"Saved π*_VI: {out_policy} | V*_VI: {out_value}")

    # Evaluate
    eval_rewards, eval_steps, sr = rollout_greedy(mdp, policy, episodes=50, max_steps=400)
    print(f"Greedy rollout: success={sr:.1f}% | avg_reward={eval_rewards.mean():.2f} | avg_steps={eval_steps.mean():.1f}")

    # ---- Visuals ----
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(V, bins=30, edgecolor='k', alpha=0.85)
    plt.title("Phân bố V* (Value Iteration)")
    plt.xlabel("V*"); plt.ylabel("Số trạng thái")

    V_grid, A_grid = policy_to_grid(policy, V, mdp.W, mdp.H)
    plt.subplot(1,3,2)
    im = plt.imshow(V_grid, cmap='viridis')
    plt.title("Heatmap V* (VI)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(mdp.W)); plt.yticks(range(mdp.H))

    plt.subplot(1,3,3)
    im2 = plt.imshow(V_grid, cmap='viridis')
    plt.title("Policy (mũi tên) trên V* (VI)")
    arrows_from_policy(plt.gca(), A_grid)
    plt.xticks(range(mdp.W)); plt.yticks(range(mdp.H))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(eval_rewards, bins=20, edgecolor='k', alpha=0.85)
    plt.axvline(eval_rewards.mean(), color='r', linestyle='--', label=f"Mean = {eval_rewards.mean():.2f}")
    plt.title("Phân bố tổng reward (rollout greedy, VI)")
    plt.xlabel("Total Reward / episode"); plt.ylabel("Số episode"); plt.legend()

    plt.subplot(1,2,2)
    bins = min(30, max(5, int(np.sqrt(len(eval_steps)))))
    plt.hist(eval_steps, bins=bins, edgecolor='k', alpha=0.85)
    plt.axvline(eval_steps.mean(), color='r', linestyle='--', label=f"Mean = {eval_steps.mean():.1f} steps")
    plt.title("Phân bố số bước (rollout greedy, VI)")
    plt.xlabel("Steps / episode"); plt.ylabel("Số episode"); plt.legend()
    plt.tight_layout()
    plt.show()
