import numpy as np
import matplotlib.pyplot as plt
import pickle, random, time
from typing import Tuple, List

# =========================== CONFIG ===========================
MAP_PATH    = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"

# Epsilon set (giữ 0.05 là tham số tối ưu đã dùng)
EPS_LIST    = [0.9, 0.5, 0.05]
# Gamma set (giữ 0.98 là tham số tối ưu đã dùng)
GAMMA_LIST  = [0.1, 0.9, 0.98]

EPISODES_1000 = 1000
MAX_STEPS     = 400
SEED          = 123
MA_WIN        = 25   # moving average window

# ====================== LIGHTWEIGHT MAZE & ENV ======================
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

class MazeEnv:
    STEP_PENALTY   = -0.1
    WALL_PENALTY   = -0.5
    GOAL_REWARD    = +20.0
    ENEMY_PENALTY  = -10.0

    def __init__(self, maze: Maze):
        self.maze = maze
        self.width, self.height = maze.width, maze.height
        self.total_nodes = maze.total_nodes
        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", self.width*self.height - 1)))
        self.enemies = set(getattr(maze, "static_enemies", getattr(maze, "_static_enemies", [])))
        self.adj = maze.adjacency
        self.n_actions = 4
        self.action_space = [0,1,2,3]
        self.state = self.start
        self.steps = 0

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def _move(self, s: int, a: int) -> int:
        H, W = self.height, self.width
        x, y = s // H, s % H
        ns = s
        if a == 0 and y + 1 < H:
            t = x * H + (y + 1)
            if self.adj[s, t] > 0: ns = t
        elif a == 1 and x + 1 < W:
            t = (x + 1) * H + y
            if self.adj[s, t] > 0: ns = t
        elif a == 2 and y - 1 >= 0:
            t = x * H + (y - 1)
            if self.adj[s, t] > 0: ns = t
        elif a == 3 and x - 1 >= 0:
            t = (x - 1) * H + y
            if self.adj[s, t] > 0: ns = t
        return ns

    def step(self, a: int):
        self.steps += 1
        s = self.state
        ns = self._move(s, a)
        r, done = self.STEP_PENALTY, False
        if ns == s:
            r = self.WALL_PENALTY
        if ns in self.enemies:
            r, done = self.ENEMY_PENALTY, True
        elif ns == self.goal:
            r, done = self.GOAL_REWARD, True
        self.state = ns
        return ns, r, done, {}

# =========================== MDP (PI/VI) ===========================
class MazeMDP:
    ACTIONS = [0,1,2,3]
    def __init__(self, maze: Maze, gamma: float):
        self.maze = maze
        self.N, self.W, self.H = maze.total_nodes, maze.width, maze.height
        self.adj = maze.adjacency
        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode", getattr(maze, "_sinkerNode", self.W*self.H-1)))
        self.enemies = set(getattr(maze,"static_enemies", getattr(maze,"_static_enemies", [])))

        self.STEP_PENALTY, self.WALL_PENALTY = -0.1, -0.5
        self.GOAL_REWARD, self.ENEMY_PENALTY = 20.0, -10.0
        self.GAMMA = gamma

        self.P = np.zeros((self.N, 4), dtype=np.int32)
        self.R = np.zeros((self.N, 4), dtype=np.float64)
        self.terminal = np.zeros(self.N, dtype=bool)

        for s in range(self.N):
            if s == self.goal or s in self.enemies:
                self.terminal[s] = True

        for s in range(self.N):
            x, y = s // self.H, s % self.H
            if self.terminal[s]:
                self.P[s] = s
                self.R[s] = 0.0
                continue
            for a in self.ACTIONS:
                ns = s
                if a == 0 and y + 1 < self.H:
                    cand = x * self.H + (y + 1)
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 1 and x + 1 < self.W:
                    cand = (x + 1) * self.H + y
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 2 and y - 1 >= 0:
                    cand = x * self.H + (y - 1)
                    if self.adj[s, cand] > 0: ns = cand
                elif a == 3 and x - 1 >= 0:
                    cand = (x - 1) * self.H + y
                    if self.adj[s, cand] > 0: ns = cand
                self.P[s, a] = ns
                if ns == s:              r = self.WALL_PENALTY
                elif ns in self.enemies: r = self.ENEMY_PENALTY
                elif ns == self.goal:    r = self.GOAL_REWARD
                else:                    r = self.STEP_PENALTY
                self.R[s, a] = r

# ====================== Agents (Q / SARSA / MC) ======================
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.98, eps_fixed=0.05):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps_fixed
        self.n_actions = n_actions
    def act(self, s):
        return random.randrange(self.n_actions) if random.random() < self.eps else int(np.argmax(self.Q[s]))
    def update(self, s,a,r,sn,done):
        best = 0.0 if done else np.max(self.Q[sn])
        td = r + self.gamma*best - self.Q[s,a]
        self.Q[s,a] += self.alpha*td

class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=0.98, eps_fixed=0.05):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps_fixed
        self.n_actions = n_actions
    def act(self, s):
        return random.randrange(self.n_actions) if random.random() < self.eps else int(np.argmax(self.Q[s]))
    def update(self, s,a,r,sn,an,done):
        nxt = 0.0 if done else self.Q[sn,an]
        td = r + self.gamma*nxt - self.Q[s,a]
        self.Q[s,a] += self.alpha*td

class MonteCarloAgent:
    def __init__(self, n_states, n_actions, gamma=0.98, eps_fixed=0.05):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.sum = np.zeros_like(self.Q, dtype=np.float64)
        self.cnt = np.zeros_like(self.Q, dtype=np.int64)
        self.gamma = gamma
        self.eps = eps_fixed
        self.n_actions = n_actions
    def act(self, s):
        return random.randrange(self.n_actions) if random.random() < self.eps else int(np.argmax(self.Q[s]))
    def update_from_episode(self, episode, first_visit=True):
        G=0.0; visited=set()
        for t in reversed(range(len(episode))):
            s,a,r = episode[t]
            G = self.gamma*G + r
            key = (s,a)
            if first_visit and key in visited: continue
            visited.add(key)
            self.sum[s,a] += G
            self.cnt[s,a] += 1
            self.Q[s,a] = self.sum[s,a] / max(1, self.cnt[s,a])

# ====================== Training loops (ε experiments) ======================
def run_q_eps(env: MazeEnv, eps: float, episodes: int = EPISODES_1000, max_steps: int = MAX_STEPS, seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    ag = QLearningAgent(env.total_nodes, env.n_actions, alpha=0.15, gamma=0.98, eps_fixed=eps)
    rewards=[]
    t0 = time.time()
    for _ in range(episodes):
        s=env.reset(); R=0.0
        for _t in range(max_steps):
            a=ag.act(s)
            sn,r,d,_=env.step(a)
            ag.update(s,a,r,sn,d)
            s=sn; R+=r
            if d: break
        rewards.append(R)
    elapsed = time.time() - t0
    return np.array(rewards), elapsed

def run_sarsa_eps(env: MazeEnv, eps: float, episodes: int = EPISODES_1000, max_steps: int = MAX_STEPS, seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    ag = SarsaAgent(env.total_nodes, env.n_actions, alpha=0.15, gamma=0.98, eps_fixed=eps)
    rewards=[]
    t0 = time.time()
    for _ in range(episodes):
        s=env.reset(); a=ag.act(s); R=0.0
        for _t in range(max_steps):
            sn,r,d,_=env.step(a)
            an=ag.act(sn)
            ag.update(s,a,r,sn,an,d)
            s,a=sn,an; R+=r
            if d: break
        rewards.append(R)
    elapsed = time.time() - t0
    return np.array(rewards), elapsed

def run_mc_eps(env: MazeEnv, eps: float, episodes: int = EPISODES_1000, max_steps: int = MAX_STEPS, seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    ag = MonteCarloAgent(env.total_nodes, env.n_actions, gamma=0.98, eps_fixed=eps)
    rewards=[]
    t0 = time.time()
    for _ in range(episodes):
        s=env.reset(); ep=[]
        for _t in range(max_steps):
            a=ag.act(s)
            sn,r,d,_=env.step(a)
            ep.append((s,a,r))
            s=sn
            if d: break
        ag.update_from_episode(ep, first_visit=True)
        R=float(np.sum([r for (_,_,r) in ep]))
        rewards.append(R)
    elapsed = time.time() - t0
    return np.array(rewards), elapsed

# ====================== DP solvers (γ experiments) ======================
def policy_evaluation(mdp: MazeMDP, policy: np.ndarray, theta=1e-6, max_iter=100000):
    V=np.zeros(mdp.N,dtype=np.float64); g=mdp.GAMMA
    for _ in range(max_iter):
        delta=0.0
        for s in range(mdp.N):
            if mdp.terminal[s]: continue
            a=policy[s]; ns=mdp.P[s,a]; r=mdp.R[s,a]
            v = r + g*V[ns]
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta: break
    return V

def policy_improvement(mdp: MazeMDP, V: np.ndarray):
    g=mdp.GAMMA
    policy=np.zeros(mdp.N,dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]: continue
        q = mdp.R[s] + g*V[mdp.P[s]]
        policy[s]=int(np.argmax(q))
    return policy

def policy_iteration(mdp: MazeMDP, theta=1e-6, max_pi_iter=200):
    policy=np.random.randint(0,4,size=mdp.N,dtype=np.int32)
    policy[mdp.terminal]=0
    for _ in range(max_pi_iter):
        V = policy_evaluation(mdp, policy, theta=theta)
        new_policy = policy_improvement(mdp, V)
        if np.all(new_policy==policy): break
        policy = new_policy
    return V, policy

def value_iteration(mdp: MazeMDP, theta=1e-6, max_iter=200000):
    V=np.zeros(mdp.N,dtype=np.float64); g=mdp.GAMMA
    for _ in range(max_iter):
        delta=0.0
        for s in range(mdp.N):
            if mdp.terminal[s]: continue
            q = mdp.R[s] + g*V[mdp.P[s]]
            v = np.max(q)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta: break
    policy=np.zeros(mdp.N,dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]: continue
        q = mdp.R[s] + g*V[mdp.P[s]]
        policy[s]=int(np.argmax(q))
    return V, policy

def rollout_mdp(mdp: MazeMDP, policy: np.ndarray, episodes: int = EPISODES_1000, max_steps: int = MAX_STEPS):
    totals=[]
    for _ in range(episodes):
        s=mdp.start; R=0.0
        for _t in range(max_steps):
            if s==mdp.goal or s in mdp.enemies: break
            a=policy[s]; ns=mdp.P[s,a]; r=mdp.R[s,a]
            R+=r; s=ns
            if s==mdp.goal or s in mdp.enemies: break
        totals.append(R)
    return np.array(totals)

# ====================== Utils ======================
def moving_average(y: np.ndarray, k: int):
    if len(y) < k: return None, None
    ker = np.ones(k)/k
    ma = np.convolve(y, ker, mode='valid')
    xs = np.arange(k-1, k-1+len(ma))
    return xs, ma

def summary_stats(name, table_rewards, table_times):
    print(f"\n[{name}] mean reward over {EPISODES_1000} eps & total train time:")
    for k in sorted(table_rewards.keys(), key=lambda x: float(x)):
        mean_r = np.mean(table_rewards[k])
        t = table_times[k]
        print(f"  ε={k}: mean_reward={mean_r:.2f} | time={t:.3f}s")

def summary_gamma(name, times, rewards, gammas):
    print(f"\n{name}:")
    for i,g in enumerate(gammas):
        print(f"  γ={g}: time={times[i]:.3f}s | avg_reward={rewards[i]:.2f}")

# ====================== MAIN ======================
if __name__ == "__main__":
    maze = Maze().load(MAP_PATH)
    env  = MazeEnv(maze)

    # ---------- A) EPSILON EFFECTS (Q / SARSA / MC) ----------
    print("=== A) EPSILON effects (Q / SARSA / MC) ===")
    results_q, results_s, results_m = {}, {}, {}
    times_q,   times_s,   times_m   = {}, {}, {}

    for eps in EPS_LIST:
        print(f"[Q] eps={eps}")
        arr, t = run_q_eps(env, eps=eps)
        results_q[eps] = arr; times_q[eps] = t

        print(f"[SARSA] eps={eps}")
        arr, t = run_sarsa_eps(env, eps=eps)
        results_s[eps] = arr; times_s[eps] = t

        print(f"[MC] eps={eps}")
        arr, t = run_mc_eps(env, eps=eps)
        results_m[eps] = arr; times_m[eps] = t

    # ---- Plots for ε ----
    # Q-Learning
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for eps, arr in results_q.items():
        xs, ma = moving_average(arr, MA_WIN)
        if xs is not None:
            plt.plot(xs, ma, label=f"ε={eps} (MA{MA_WIN})", linewidth=2)
        plt.plot(arr, alpha=0.25, linewidth=0.8)
    plt.title("Q-Learning: Total Reward vs Episode (1000 eps)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()

    # SARSA
    plt.subplot(1,2,2)
    for eps, arr in results_s.items():
        xs, ma = moving_average(arr, MA_WIN)
        if xs is not None:
            plt.plot(xs, ma, label=f"ε={eps} (MA{MA_WIN})", linewidth=2)
        plt.plot(arr, alpha=0.25, linewidth=0.8)
    plt.title("SARSA: Total Reward vs Episode (1000 eps)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()
    plt.tight_layout()
    plt.show()

    # MC
    plt.figure(figsize=(7,5))
    for eps, arr in results_m.items():
        xs, ma = moving_average(arr, MA_WIN)
        if xs is not None:
            plt.plot(xs, ma, label=f"ε={eps} (MA{MA_WIN})", linewidth=2)
        plt.plot(arr, alpha=0.25, linewidth=0.8)
    plt.title("Monte Carlo: Total Reward vs Episode (1000 eps)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Training time bars for ε ----
    eps_labels = [f"ε={e}" for e in EPS_LIST]
    x_eps = np.arange(len(EPS_LIST)); w = 0.25

    plt.figure(figsize=(10,4))
    plt.bar(x_eps - w, [times_q[e] for e in EPS_LIST], width=w, label="Q-Learning")
    plt.bar(x_eps,     [times_s[e] for e in EPS_LIST], width=w, label="SARSA")
    plt.bar(x_eps + w, [times_m[e] for e in EPS_LIST], width=w, label="Monte Carlo")
    plt.xticks(x_eps, eps_labels)
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs ε (Q / SARSA / MC)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Summaries ε ----
    summary_stats("Q-Learning (ε)", results_q, times_q)
    summary_stats("SARSA (ε)",      results_s, times_s)
    summary_stats("Monte Carlo (ε)",results_m, times_m)

    # ---------- B) GAMMA EFFECTS (PI / VI) ----------
    print("\n=== B) GAMMA effects (PI / VI) ===")
    pi_time=[]; pi_reward=[]; vi_time=[]; vi_reward=[]
    pi_table={}; vi_table={}

    for gamma in GAMMA_LIST:
        mdp = MazeMDP(maze, gamma=gamma)

        # Policy Iteration
        t0=time.time()
        Vpi, poli = policy_iteration(mdp, theta=1e-7, max_pi_iter=300)
        t1=time.time()
        pi_time.append(t1-t0)
        r_pi = rollout_mdp(mdp, poli)
        pi_reward.append(float(np.mean(r_pi)))
        pi_table[gamma] = r_pi

        # Value Iteration
        t0=time.time()
        Vvi, poli_vi = value_iteration(mdp, theta=1e-7, max_iter=600_000)
        t1=time.time()
        vi_time.append(t1-t0)
        r_vi = rollout_mdp(mdp, poli_vi)
        vi_reward.append(float(np.mean(r_vi)))
        vi_table[gamma] = r_vi

        print(f"γ={gamma} | PI: time={pi_time[-1]:.3f}s, reward={pi_reward[-1]:.2f} | "
              f"VI: time={vi_time[-1]:.3f}s, reward={vi_reward[-1]:.2f}")

    pi_time=np.array(pi_time); pi_reward=np.array(pi_reward)
    vi_time=np.array(vi_time); vi_reward=np.array(vi_reward)

    # ---- Plots for γ (separate PI and VI) ----
    x = np.arange(len(GAMMA_LIST))
    gamma_ticks = [f"γ={g}" for g in GAMMA_LIST]

    # === Figure: POLICY ITERATION ===
    plt.figure(figsize=(12,4))
    # (a) Time
    plt.subplot(1,2,1)
    plt.bar(x, pi_time, width=0.5)
    plt.xticks(x, gamma_ticks)
    plt.ylabel("Training Time (s)")
    plt.title("Policy Iteration: Training Time vs γ")
    # (b) Avg reward
    plt.subplot(1,2,2)
    plt.bar(x, pi_reward, width=0.5)
    plt.xticks(x, gamma_ticks)
    plt.ylabel("Average Reward (1000 rollouts)")
    plt.title("Policy Iteration: Reward vs γ")
    plt.tight_layout()
    plt.show()

    # === Figure: VALUE ITERATION ===
    plt.figure(figsize=(12,4))
    # (a) Time
    plt.subplot(1,2,1)
    plt.bar(x, vi_time, width=0.5)
    plt.xticks(x, gamma_ticks)
    plt.ylabel("Training Time (s)")
    plt.title("Value Iteration: Training Time vs γ")
    # (b) Avg reward
    plt.subplot(1,2,2)
    plt.bar(x, vi_reward, width=0.5)
    plt.xticks(x, gamma_ticks)
    plt.ylabel("Average Reward (1000 rollouts)")
    plt.title("Value Iteration: Reward vs γ")
    plt.tight_layout()
    plt.show()
