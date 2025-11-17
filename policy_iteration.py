import numpy as np
import pickle
from typing import Tuple, Dict, List
import time
import matplotlib.pyplot as plt

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
    MDP deterministic:
    - States: 0..N-1 (node indices)
    - Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
    - P[s,a] = s' (deterministic). Nếu không có cạnh -> ở lại s (đụng tường)
    - R[s,a,s']: theo preset "dịu": step -0.1; wall -0.5; goal +20; enemy -10
    - Terminal: goal và mọi enemy (hấp thụ)
    """
    ACTIONS = [0,1,2,3]  # up, right, down, left

    def __init__(self, maze: Maze):
        self.maze = maze
        self.N = maze.total_nodes
        self.W = maze.width
        self.H = maze.height
        self.adj = maze.adjacency

        self.start  = int(getattr(maze, "startNode",  getattr(maze, "_startNode", 0)))
        self.goal   = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", self.W*self.H-1)))
        self.enemies = set(getattr(maze,"static_enemies", getattr(maze,"_static_enemies", [])))

        # Reward preset (khớp env đã chỉnh)
        self.STEP_PENALTY  = -0.1
        self.WALL_PENALTY  = -0.5
        self.GOAL_REWARD   = +20.0
        self.ENEMY_PENALTY = -10.0

        # Discount (khớp env/shaping)
        self.GAMMA = 0.9

        # Build deterministic transition model
        self.P = np.zeros((self.N, 4), dtype=np.int32)      # next state s'
        self.R = np.zeros((self.N, 4), dtype=np.float64)    # reward r(s,a)
        self.terminal = np.zeros(self.N, dtype=bool)        # terminal flags

        for s in range(self.N):
            if s == self.goal or s in self.enemies:
                self.terminal[s] = True

        for s in range(self.N):
            x, y = s // self.H, s % self.H
            # For terminal states: self-loop with zero reward (absorbing)
            if self.terminal[s]:
                for a in self.ACTIONS:
                    self.P[s, a] = s
                    # reward khi ở terminal giữ 0 (vì đã nhận khi vào terminal)
                    self.R[s, a] = 0.0
                continue

            # otherwise deterministic move
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
                # reward function
                if ns == s:
                    r = self.WALL_PENALTY           # đụng tường
                elif ns in self.enemies:
                    r = self.ENEMY_PENALTY          # vào enemy (terminal âm)
                elif ns == self.goal:
                    r = self.GOAL_REWARD            # vào goal (terminal dương)
                else:
                    r = self.STEP_PENALTY           # bước thường
                self.R[s, a] = r

# ======================== POLICY EVALUATION + IMPROVEMENT ========================
def policy_evaluation(mdp: MazeMDP, policy: np.ndarray, theta: float = 1e-6, max_iter: int = 10_000) -> np.ndarray:
    """
    Đánh giá giá trị trạng thái V^π cho chính sách xác định (deterministic policy: policy[s] = a)
    P: deterministic, nên Bellman: V[s] = R[s,a] + gamma * V[s'] (nếu s terminal, V[s]=0)
    Lặp cho tới khi hội tụ (||ΔV||_∞ < theta)
    """
    V = np.zeros(mdp.N, dtype=np.float64)
    gamma = mdp.GAMMA

    # terminal value = 0 (absorbing with zero reward onwards)
    for it in range(max_iter):
        delta = 0.0
        for s in range(mdp.N):
            if mdp.terminal[s]:
                continue
            a = policy[s]
            ns = mdp.P[s, a]
            r  = mdp.R[s, a]
            v_new = r + gamma * V[ns]
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V

def policy_improvement(mdp: MazeMDP, V: np.ndarray) -> np.ndarray:
    """
    Tạo policy greedy theo V:
    π_new(s) = argmax_a [ R[s,a] + γ V[s'] ]
    Với terminal, action vô nghĩa → giữ 0.
    """
    gamma = mdp.GAMMA
    policy_new = np.zeros(mdp.N, dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]:
            policy_new[s] = 0
            continue
        q_vals = mdp.R[s] + gamma * V[mdp.P[s]]
        policy_new[s] = int(np.argmax(q_vals))
    return policy_new

def policy_iteration(mdp: MazeMDP, theta_eval: float = 1e-6, max_eval_iter: int = 10000, max_pi_iter: int = 1000, verbose: bool = True):
    """
    Policy Iteration:
      1) Init policy ngẫu nhiên
      2) Evaluate V^π
      3) Improve π <- greedy(V)
      4) Nếu không đổi → dừng
    """
    np.random.seed(123)
    # init deterministic policy (random among legal actions; cho đơn giản, chọn random 0..3)
    policy = np.random.randint(0, 4, size=mdp.N, dtype=np.int32)
    # terminal action set to 0
    policy[mdp.terminal] = 0

    for it in range(1, max_pi_iter + 1):
        V = policy_evaluation(mdp, policy, theta=theta_eval, max_iter=max_eval_iter)
        new_policy = policy_improvement(mdp, V)
        stable = np.all(new_policy == policy)
        if verbose:
            changed = np.sum(new_policy != policy)
            print(f"[PI] Iter {it:3d} | changed states = {changed}")
        policy = new_policy
        if stable:
            if verbose:
                print("[PI] Policy stable. Converged.")
            break
    return policy, V

# ======================== EVALUATION (rollout greedy) ========================
def rollout_greedy(mdp: MazeMDP, policy: np.ndarray, episodes: int = 40, max_steps: int = 400) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Đánh giá bằng cách chạy policy xác định trên MDP (deterministic).
    Trả về tổng reward/steps/ success rate.
    """
    total_rewards = []
    steps_list = []
    success = 0
    for _ in range(episodes):
        s = mdp.start
        total_r = 0.0
        for t in range(max_steps):
            if s == mdp.goal:
                break
            if s in mdp.enemies:
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

# ======================== MAIN ========================
if __name__ == "__main__":
    MAP_PATH = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"

    maze = Maze().load(MAP_PATH)
    mdp = MazeMDP(maze)

    print(f"Maze loaded: {maze.width}x{maze.height}")
    print(f"Start={mdp.start}, Goal={mdp.goal}, Enemies={sorted(list(mdp.enemies))}")
    print(f"Gamma={mdp.GAMMA}, Rewards: step={mdp.STEP_PENALTY}, wall={mdp.WALL_PENALTY}, goal={mdp.GOAL_REWARD}, enemy={mdp.ENEMY_PENALTY}")

    t0 = time.time()
    policy, V = policy_iteration(mdp, theta_eval=1e-7, max_eval_iter=20000, max_pi_iter=200, verbose=True)
    t1 = time.time()
    print(f"Policy Iteration done in {t1 - t0:.2f}s")

    # Lưu kết quả
    out_policy = "/Users/haiduong/Desktop/Maze/policy_star_12x12.npy"
    out_value  = "/Users/haiduong/Desktop/Maze/V_star_12x12.npy"
    np.save(out_policy, policy)
    np.save(out_value, V)
    print(f"Saved π*: {out_policy} | V*: {out_value}")

    # Đánh giá rollout greedy
    eval_rewards, eval_steps, sr = rollout_greedy(mdp, policy, episodes=50, max_steps=400)
    print(f"Greedy rollout: success={sr:.1f}% | avg_reward={eval_rewards.mean():.2f} | avg_steps={eval_steps.mean():.1f}")

    # Gợi ý xem một vài trạng thái mẫu
    print("\nSample states (start/goal) values:")
    print(f"V[start]={V[mdp.start]:.3f}, V[goal]={V[mdp.goal]:.3f}")


# ======== VISUALS: đánh giá trực quan cho Policy Iteration ========
import matplotlib.pyplot as plt

def node2yx(n, H):
    # Vẽ heatmap cần (y,x) theo ma trận 2D [H, W]
    x, y = n // H, n % H
    return y, x

def policy_to_grid(policy: np.ndarray, V: np.ndarray, W: int, H: int):
    """
    Chuyển policy & V (vector) về dạng lưới [H, W] để vẽ heatmap & mũi tên.
    """
    V_grid = np.zeros((H, W), dtype=float)
    A_grid = np.zeros((H, W), dtype=int)
    for s in range(W * H):
        y, x = node2yx(s, H)
        V_grid[y, x] = V[s]
        A_grid[y, x] = policy[s]
    return V_grid, A_grid

def arrows_from_policy(ax, A_grid):
    """
    Vẽ mũi tên theo action trên lưới:
    0=UP(0,+1y), 1=RIGHT(+1x), 2=DOWN(0,-1y), 3=LEFT(-1x)
    Chú ý: trong heatmap imshow(origin='upper' mặc định), trục y ngược.
    Ở đây giữ origin mặc định và bù vector cho đúng trực quan.
    """
    H, W = A_grid.shape
    # map action -> (dx, dy) theo hệ trục imshow mặc định (y xuống là +)
    vec = {
        0: (0, -1),   # UP: mũi tên đi lên
        1: (1, 0),    # RIGHT
        2: (0, 1),    # DOWN
        3: (-1, 0)    # LEFT
    }
    for y in range(H):
        for x in range(W):
            a = int(A_grid[y, x])
            dx, dy = vec.get(a, (0,0))
            ax.arrow(x, y, dx*0.35, dy*0.35, head_width=0.2, head_length=0.2, fc='k', ec='k', length_includes_head=True)

# 1) Histogram V*
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.hist(V, bins=30, edgecolor='k', alpha=0.8)
plt.title("Phân bố V* (Policy Iteration)")
plt.xlabel("V*"); plt.ylabel("Số trạng thái")

# 2) Heatmap V* + 5) Mũi tên policy
V_grid, A_grid = policy_to_grid(policy, V, mdp.W, mdp.H)
plt.subplot(1,3,2)
im = plt.imshow(V_grid, cmap='viridis')
plt.title("Heatmap V*")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(mdp.W)); plt.yticks(range(mdp.H))

plt.subplot(1,3,3)
im2 = plt.imshow(V_grid, cmap='viridis')
plt.title("Policy (arrows) over V*")
arrows_from_policy(plt.gca(), A_grid)
plt.xticks(range(mdp.W)); plt.yticks(range(mdp.H))

plt.tight_layout()
plt.show()

# 3) & 4) Phân bố reward / số bước từ rollout greedy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(eval_rewards, bins=20, edgecolor='k', alpha=0.85)
plt.axvline(eval_rewards.mean(), color='r', linestyle='--', label=f"Mean = {eval_rewards.mean():.2f}")
plt.title("Phân bố tổng reward (rollout greedy)")
plt.xlabel("Total Reward / episode"); plt.ylabel("Số episode"); plt.legend()

plt.subplot(1,2,2)
plt.hist(eval_steps, bins=min(30, max(5, int(np.sqrt(len(eval_steps))))), edgecolor='k', alpha=0.85)
plt.axvline(eval_steps.mean(), color='r', linestyle='--', label=f"Mean = {eval_steps.mean():.1f} steps")
plt.title("Phân bố số bước (rollout greedy)")
plt.xlabel("Steps / episode"); plt.ylabel("Số episode"); plt.legend()

plt.tight_layout()
plt.show()
