import numpy as np
import matplotlib.pyplot as plt
import pickle, time, random
from typing import Tuple, List

# =============== CONFIG ===============
MAP_PATH = "/Users/haiduong/Desktop/Maze/maze/maze_12x12_20251110_030522.pkl"
SAVE_DIR = "/Users/haiduong/Desktop/Maze"

# Common hyperparams
GAMMA = 0.98
EPISODES_Q = 1200
EPISODES_SARSA = 1200
EPISODES_MC = 1500
MAX_STEPS = 400
EPS_START = 1.0
EPS_END   = 0.05
DECAY_Q_SARSA = 800
DECAY_MC      = 900
ALPHA_Q = 0.15
ALPHA_SARSA = 0.15
SEED = 123

# ===================================== MAZE / ENV =====================================
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
        """
        Map node index -> toạ độ (x, y) trên lưới maze_map
        Đồng bộ với code GUI: width = số cột (j), height = số dòng (i)
        """
        return 2 * (n // self.height), 2 * (n % self.height)

    def generate_maze_map(self) -> np.ndarray:
        """
        Sinh lưới maze_map (0 = tường, 1 = đường đi) giống code GUI.
        """
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
    Gym-like deterministic maze:
      Actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT
      Reward (soft): step=-0.1, wall=-0.5, goal=+20, enemy=-10
      Shaping (potential) đã được đưa vào ENV chính; ở đây ta giữ bản base cho công bằng.
    """
    STEP_PENALTY   = -0.1
    WALL_PENALTY   = -0.5
    GOAL_REWARD    = +20.0
    ENEMY_PENALTY  = -10.0
    GAMMA          = GAMMA

    def __init__(self, maze: Maze):
        self.maze = maze
        self.width = maze.width
        self.height = maze.height
        self.total_nodes = maze.total_nodes
        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode",
                                 getattr(maze, "_sinkerNode", self.width*self.height - 1)))
        self.enemies = set(getattr(maze, "static_enemies",
                                   getattr(maze, "_static_enemies", [])))
        self.adj = maze.adjacency
        self.action_space = [0,1,2,3]
        self.n_actions = 4
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
        s, ns = self.state, self._move(self.state, a)
        r, done = self.STEP_PENALTY, False
        if ns == s: r = self.WALL_PENALTY
        if ns in self.enemies:
            r, done = self.ENEMY_PENALTY, True
        elif ns == self.goal:
            r, done = self.GOAL_REWARD, True
        self.state = ns
        return ns, r, done, {}

# ===================================== MDP builder (for PI/VI) =====================================
class MazeMDP:
    ACTIONS = [0,1,2,3]
    def __init__(self, maze: Maze):
        self.maze = maze
        self.N, self.W, self.H = maze.total_nodes, maze.width, maze.height
        self.adj = maze.adjacency
        self.start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
        self.goal  = int(getattr(maze, "sinkerNode",
                                 getattr(maze, "_sinkerNode", self.W*self.H-1)))
        self.enemies = set(getattr(maze,"static_enemies", getattr(maze,"_static_enemies", [])))
        self.STEP_PENALTY, self.WALL_PENALTY = -0.1, -0.5
        self.GOAL_REWARD, self.ENEMY_PENALTY = 20.0, -10.0
        self.GAMMA = GAMMA

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
                if ns == s: r = self.WALL_PENALTY
                elif ns in self.enemies: r = self.ENEMY_PENALTY
                elif ns == self.goal:    r = self.GOAL_REWARD
                else:                    r = self.STEP_PENALTY
                self.R[s, a] = r

# ===================================== TD & MC agents =====================================
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=GAMMA,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=800):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.decay = max(1, eps_decay_episodes)
        self.n_actions = n_actions

    def eps(self, ep): 
        f = min(1.0, ep / self.decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, s, eps): 
        return random.randrange(self.n_actions) if random.random() < eps else int(np.argmax(self.Q[s]))

    def update(self, s, a, r, sn, done):
        best = 0.0 if done else np.max(self.Q[sn])
        td = r + self.gamma * best - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.15, gamma=GAMMA,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=800):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.decay = max(1, eps_decay_episodes)
        self.n_actions = n_actions

    def eps(self, ep):
        f = min(1.0, ep / self.decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, s, eps):
        return random.randrange(self.n_actions) if random.random() < eps else int(np.argmax(self.Q[s]))

    def update(self, s, a, r, sn, an, done):
        nxt = 0.0 if done else self.Q[sn, an]
        td = r + self.gamma * nxt - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

class MonteCarloAgent:
    def __init__(self, n_states, n_actions, gamma=GAMMA, eps_start=1.0, eps_end=0.05, eps_decay_episodes=900):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.sum = np.zeros_like(self.Q, dtype=np.float64)
        self.cnt = np.zeros_like(self.Q, dtype=np.int64)
        self.gamma = gamma
        self.eps_start, self.eps_end, self.decay = eps_start, eps_end, max(1, eps_decay_episodes)
        self.n_actions = n_actions

    def eps(self, ep):
        f = min(1.0, ep / self.decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, s, eps):
        return random.randrange(self.n_actions) if random.random() < eps else int(np.argmax(self.Q[s]))

    def update_from_episode(self, episode, first_visit=True):
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            k = (s, a)
            if first_visit and k in visited:
                continue
            visited.add(k)
            self.sum[s, a] += G
            self.cnt[s, a] += 1
            self.Q[s, a] = self.sum[s, a] / max(1, self.cnt[s, a])

# ===================================== TRAINERS & EVAL =====================================
def train_q(env, episodes=EPISODES_Q, max_steps=MAX_STEPS, alpha=ALPHA_Q, gamma=GAMMA,
            eps_start=EPS_START, eps_end=EPS_END, decay=DECAY_Q_SARSA, seed=SEED):
    random.seed(seed); np.random.seed(seed)
    ag = QLearningAgent(env.total_nodes, env.n_actions, alpha, gamma, eps_start, eps_end, decay)
    rewards = []; steps = []
    for ep in range(1, episodes+1):
        s = env.reset(); R = 0.0
        for _ in range(max_steps):
            a = ag.act(s, ag.eps(ep))
            sn, r, d, _ = env.step(a)
            ag.update(s, a, r, sn, d)
            s = sn; R += r
            if d: break
        rewards.append(R); steps.append(env.steps)
    return ag, np.array(rewards), np.array(steps)

def train_sarsa(env, episodes=EPISODES_SARSA, max_steps=MAX_STEPS, alpha=ALPHA_SARSA, gamma=GAMMA,
                eps_start=EPS_START, eps_end=EPS_END, decay=DECAY_Q_SARSA, seed=SEED):
    random.seed(seed); np.random.seed(seed)
    ag = SarsaAgent(env.total_nodes, env.n_actions, alpha, gamma, eps_start, eps_end, decay)
    rewards = []; steps = []
    for ep in range(1, episodes+1):
        s = env.reset(); a = ag.act(s, ag.eps(ep)); R = 0.0
        for _ in range(max_steps):
            sn, r, d, _ = env.step(a)
            an = ag.act(sn, ag.eps(ep))
            ag.update(s, a, r, sn, an, d)
            s, a = sn, an; R += r
            if d: break
        rewards.append(R); steps.append(env.steps)
    return ag, np.array(rewards), np.array(steps)

def generate_episode(env, agent, max_steps, eps):
    s = env.reset(); ep = []
    for _ in range(max_steps):
        a = agent.act(s, eps)
        sn, r, d, _ = env.step(a)
        ep.append((s, a, r)); s = sn
        if d: break
    return ep, s

def train_mc(env, episodes=EPISODES_MC, max_steps=MAX_STEPS, gamma=GAMMA,
             eps_start=EPS_START, eps_end=EPS_END, decay=DECAY_MC, seed=SEED):
    random.seed(seed); np.random.seed(seed)
    ag = MonteCarloAgent(env.total_nodes, env.n_actions, gamma, eps_start, eps_end, decay)
    rewards = []; steps = []
    for ep in range(1, episodes+1):
        ep_e, last = generate_episode(env, ag, max_steps, ag.eps(ep))
        R = float(np.sum([r for (_,_,r) in ep_e]))
        rewards.append(R); steps.append(len(ep_e))
        ag.update_from_episode(ep_e, first_visit=True)
    return ag, np.array(rewards), np.array(steps)

def evaluate_greedy_env(env, Q, episodes=40, max_steps=MAX_STEPS):
    totals = []
    succ = 0
    for _ in range(episodes):
        s = env.reset(); R = 0.0
        for t in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, r, d, _ = env.step(a); R += r
            if d: break
        totals.append(R)
        if s == env.goal:
            succ += 1
    return np.array(totals), succ / episodes * 100.0

def evaluate_greedy_env_full(env, Q, episodes=40, max_steps=MAX_STEPS):
    totals = []; steps = []; succ = 0
    for _ in range(episodes):
        s = env.reset(); R = 0.0
        for t in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, r, d, _ = env.step(a); R += r
            if d: break
        totals.append(R); steps.append(env.steps)
        if s == env.goal: succ += 1
    return np.array(totals), np.array(steps), succ/episodes*100.0

# ========================= DP (PI & VI) =========================
class MDPWrap:
    def __init__(self, mdp: MazeMDP): self.mdp=mdp

def policy_evaluation(mdp: MazeMDP, policy, theta=1e-6, max_iter=100000):
    V = np.zeros(mdp.N, dtype=np.float64); g = mdp.GAMMA
    for _ in range(max_iter):
        delta = 0.0
        for s in range(mdp.N):
            if mdp.terminal[s]: continue
            a = policy[s]; ns = mdp.P[s, a]; r = mdp.R[s, a]
            v = r + g * V[ns]; delta = max(delta, abs(v - V[s])); V[s] = v
        if delta < theta: break
    return V

def policy_improvement(mdp: MazeMDP, V):
    g = mdp.GAMMA; pi = np.zeros(mdp.N, dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]:
            pi[s] = 0
            continue
        q = mdp.R[s] + g * V[mdp.P[s]]
        pi[s] = int(np.argmax(q))
    return pi

def policy_iteration(mdp: MazeMDP, theta=1e-6, max_pi_iter=200):
    pi = np.random.randint(0, 4, size=mdp.N, dtype=np.int32)
    pi[mdp.terminal] = 0
    for _ in range(max_pi_iter):
        V = policy_evaluation(mdp, pi, theta=theta)
        new_pi = policy_improvement(mdp, V)
        if np.all(new_pi == pi): break
        pi = new_pi
    return V, pi

def value_iteration(mdp: MazeMDP, theta=1e-6, max_iter=200000):
    V = np.zeros(mdp.N, dtype=np.float64); g = mdp.GAMMA
    for _ in range(max_iter):
        delta = 0.0
        for s in range(mdp.N):
            if mdp.terminal[s]: continue
            q = mdp.R[s] + g * V[mdp.P[s]]
            v = np.max(q); delta = max(delta, abs(v - V[s])); V[s] = v
        if delta < theta: break
    pi = np.zeros(mdp.N, dtype=np.int32)
    for s in range(mdp.N):
        if mdp.terminal[s]: continue
        q = mdp.R[s] + g * V[mdp.P[s]]
        pi[s] = int(np.argmax(q))
    return V, pi

def rollout_mdp(mdp: MazeMDP, policy, episodes=40, max_steps=MAX_STEPS):
    totals = []; steps = []; succ = 0
    for _ in range(episodes):
        s = mdp.start; R = 0.0
        for t in range(max_steps):
            if s == mdp.goal or s in mdp.enemies: break
            a = policy[s]; ns = mdp.P[s, a]; r = mdp.R[s, a]
            R += r; s = ns
            if s == mdp.goal or s in mdp.enemies: break
        totals.append(R); steps.append(t+1)
        if s == mdp.goal: succ += 1
    return np.array(totals), np.array(steps), succ/episodes*100.0

# ========================= POLICY VISUALIZATION =========================
def plot_policy(maze: Maze, policy: np.ndarray, title: str):
    """
    Minh hoạ policy trên nền mê cung:
    - Nền: tường đen, đường đi trắng (giống GUI)
    - Start: ô vuông xanh dương
    - Goal: ô vuông đỏ
    - Enemy: ô vuông vàng
    - Mũi tên: hướng action ưu tiên tại mỗi state
    """
    H, W = maze.height, maze.width
    N = maze.total_nodes

    start = int(getattr(maze, "startNode", getattr(maze, "_startNode", 0)))
    goal  = int(getattr(maze, "sinkerNode",
                         getattr(maze, "_sinkerNode", W * H - 1)))
    enemies = set(getattr(maze, "static_enemies",
                          getattr(maze, "_static_enemies", [])))

    # --- Nền mê cung trắng/đen giống GUI ---
    maze_map = maze.generate_maze_map()
    h_map, w_map = maze_map.shape
    canvas = np.zeros((h_map, w_map, 3), dtype=float)
    walk = (maze_map == 1)
    canvas[walk] = [1.0, 1.0, 1.0]   # đường đi: trắng
    canvas[~walk] = [0.0, 0.0, 0.0]  # tường: đen

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])

    # --- Vẽ start / goal / enemies bằng ô vuông màu ---
    sx, sy = maze.node2xy(start)
    gx, gy = maze.node2xy(goal)

    # Start: xanh dương
    ax.scatter([sx], [sy], s=160, c=[[0.0, 0.4, 1.0]], marker='s', label="Start")
    # Goal: đỏ
    ax.scatter([gx], [gy], s=160, c=[[1.0, 0.2, 0.2]], marker='s', label="Goal")

    if enemies:
        exs, eys = zip(*[maze.node2xy(e) for e in enemies])
        # Enemy: vàng, ô vuông
        ax.scatter(exs, eys, s=140, c=[[1.0, 0.8, 0.0]], marker='s', edgecolors='k', label="Enemy")

    # --- Mũi tên policy trên từng node ---
    xs, ys, us, vs = [], [], [], []
    for s in range(N):
        if s == goal or s in enemies:
            continue  # không vẽ trên goal và enemy

        a = int(policy[s])
        x, y = maze.node2xy(s)

        # 0=UP,1=RIGHT,2=DOWN,3=LEFT
        # origin='lower' nên dy>0 là đi lên trên hình
        if a == 0:       # UP
            dx, dy = 0, 1
        elif a == 1:     # RIGHT
            dx, dy = 1, 0
        elif a == 2:     # DOWN
            dx, dy = 0, -1
        else:            # LEFT
            dx, dy = -1, 0

        xs.append(x)
        ys.append(y)
        us.append(dx * 0.8)   # scale nhỏ cho đẹp
        vs.append(dy * 0.8)

    if xs:
        ax.quiver(xs, ys, us, vs,
                  angles='xy', scale_units='xy', scale=1,
                  color='cyan', width=0.005, alpha=0.9)

    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ========================= MAIN =========================
if __name__ == "__main__":
    # Load
    maze = Maze().load(MAP_PATH)
    env  = MazeEnv(maze)
    mdp  = MazeMDP(maze)

    print(f"Maze {maze.width}x{maze.height} | start={getattr(maze,'startNode',getattr(maze,'_startNode',0))} "
          f"| goal={getattr(maze,'sinkerNode',getattr(maze,'_sinkerNode',maze.width*maze.height-1))} "
          f"| enemies={sorted(list(getattr(maze,'static_enemies',getattr(maze,'_static_enemies',[]))))}")

    # ---------- Q-Learning ----------
    t0 = time.time()
    q_agent, q_rewards, q_steps = train_q(env)
    t1 = time.time()
    q_eval_rewards, q_eval_steps, q_sr = evaluate_greedy_env_full(env, q_agent.Q)
    print(f"[Q] time={t1-t0:.2f}s | SR={q_sr:.1f}% | avgR={q_eval_rewards.mean():.2f} | avgSteps={q_eval_steps.mean():.1f}")
    np.save(f"{SAVE_DIR}/qtable_12x12_q.npy", q_agent.Q)

    # ---------- SARSA ----------
    t0 = time.time()
    s_agent, s_rewards, s_steps = train_sarsa(env)
    t1 = time.time()
    s_eval_rewards, s_eval_steps, s_sr = evaluate_greedy_env_full(env, s_agent.Q)
    print(f"[SARSA] time={t1-t0:.2f}s | SR={s_sr:.1f}% | avgR={s_eval_rewards.mean():.2f} | avgSteps={s_eval_steps.mean():.1f}")
    np.save(f"{SAVE_DIR}/qtable_12x12_sarsa.npy", s_agent.Q)

    # ---------- Monte Carlo ----------
    t0 = time.time()
    mc_agent, mc_rewards, mc_steps = train_mc(env)
    t1 = time.time()
    mc_eval_rewards, mc_eval_steps, mc_sr = evaluate_greedy_env_full(env, mc_agent.Q)
    print(f"[MC] time={t1-t0:.2f}s | SR={mc_sr:.1f}% | avgR={mc_eval_rewards.mean():.2f} | avgSteps={mc_eval_steps.mean():.1f}")
    np.save(f"{SAVE_DIR}/qtable_12x12_mc.npy", mc_agent.Q)

    # ---------- Policy Iteration ----------
    t0 = time.time()
    pi_V, pi_policy = policy_iteration(mdp, theta=1e-7, max_pi_iter=200)
    t1 = time.time()
    pi_eval_rewards, pi_eval_steps, pi_sr = rollout_mdp(mdp, pi_policy, episodes=50)
    print(f"[PI] time={t1-t0:.2f}s | SR={pi_sr:.1f}% | avgR={pi_eval_rewards.mean():.2f} | avgSteps={pi_eval_steps.mean():.1f}")
    np.save(f"{SAVE_DIR}/policy_star_PI_12x12.npy", pi_policy)
    np.save(f"{SAVE_DIR}/V_star_PI_12x12.npy", pi_V)

    # ---------- Value Iteration ----------
    t0 = time.time()
    vi_V, vi_policy = value_iteration(mdp, theta=1e-7, max_iter=200000)
    t1 = time.time()
    vi_eval_rewards, vi_eval_steps, vi_sr = rollout_mdp(mdp, vi_policy, episodes=50)
    print(f"[VI] time={t1-t0:.2f}s | SR={vi_sr:.1f}% | avgR={vi_eval_rewards.mean():.2f} | avgSteps={vi_eval_steps.mean():.1f}")
    np.save(f"{SAVE_DIR}/policy_star_VI_12x12.npy", vi_policy)
    np.save(f"{SAVE_DIR}/V_star_VI_12x12.npy", vi_V)

    # ================== COMPARISON PLOTS ==================
    labels = ["Q-Learning", "SARSA", "Monte Carlo", "Policy Iteration", "Value Iteration"]
    sr = np.array([q_sr, s_sr, mc_sr, pi_sr, vi_sr])
    ar = np.array([q_eval_rewards.mean(), s_eval_rewards.mean(), mc_eval_rewards.mean(),
                   pi_eval_rewards.mean(), vi_eval_rewards.mean()])
    st = np.array([q_eval_steps.mean(), s_eval_steps.mean(), mc_eval_steps.mean(),
                   pi_eval_steps.mean(), vi_eval_steps.mean()])

    # 1) Bar charts
    plt.figure(figsize=(12,6))
    idx = np.arange(len(labels))
    w = 0.25
    plt.bar(idx - w, sr, width=w, label="Success Rate (%)")
    plt.bar(idx,     ar, width=w, label="Avg Reward")
    plt.bar(idx + w, st, width=w, label="Avg Steps")
    plt.xticks(idx, labels, rotation=15)
    plt.title("Comparison of Five RL Models on 12x12 Maze")
    plt.ylabel("Value (unit according to column)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Radar (normalized 0..1, Avg Steps đảo chiều)
    def normalize(x):
        mn, mx = x.min(), x.max()
        return np.ones_like(x) if mx - mn == 0 else (x - mn) / (mx - mn)

    sr_n = normalize(sr)              # cao hơn tốt
    ar_n = normalize(ar)              # cao hơn tốt
    st_n = 1 - normalize(st)          # ít bước hơn tốt ⇒ đảo chiều

    radar_metrics = np.vstack([sr_n, ar_n, st_n])  # 3x5
    radar_labels = ["Success Rate", "Avg Reward", "Steps (lower=better)"]

    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    for i, name in enumerate(labels):
        data = radar_metrics[:, i]
        data = np.concatenate([data, data[:1]])
        ax.plot(angles, data, linewidth=2, label=name)
        ax.fill(angles, data, alpha=0.10)
    ax.set_thetagrids(angles[:-1]*(180/np.pi), radar_labels)
    ax.set_title("Radar comparison (normalized)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    plt.show()

    # 3) Training curves (Q/SARSA/MC)
    plt.figure(figsize=(12,5))
    window = 25
    def ma(x, k=window):
        if len(x) < k:
            return np.arange(len(x)), x
        ker = np.ones(k) / k
        y = np.convolve(x, ker, mode='valid')
        xs = np.arange(k-1, k-1+len(y))
        return xs, y

    # Reward curves
    plt.subplot(1,2,1)
    for name, arr in [("Q-Learning", q_rewards), ("SARSA", s_rewards), ("MC", mc_rewards)]:
        xs, y = ma(arr)
        plt.plot(xs, y, label=f"{name} MA({window})")
        plt.plot(np.arange(len(arr)), arr, alpha=0.25, linewidth=0.8)
    plt.title("Training Reward (moving average)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()

    # Steps curves
    plt.subplot(1,2,2)
    for name, arr in [("Q-Learning", q_steps), ("SARSA", s_steps), ("MC", mc_steps)]:
        xs, y = ma(arr)
        plt.plot(xs, y, label=f"{name} MA({window})")
        plt.plot(np.arange(len(arr)), arr, alpha=0.25, linewidth=0.8)
    plt.title("Episode Length (moving average)")
    plt.xlabel("Episode"); plt.ylabel("Steps"); plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== POLICY VISUALIZATION ==================
    # Policy từ SARSA (greedy theo Q)
    pi_sarsa = np.argmax(s_agent.Q, axis=1)
    plot_policy(maze, pi_sarsa, "Learned Policy (SARSA, 12x12 Maze)")

    # Policy tối ưu từ Value Iteration
    plot_policy(maze, vi_policy, "Optimal Policy (Value Iteration, 12x12 Maze)")

    print("\n==> Done. Artifacts saved to:", SAVE_DIR)
