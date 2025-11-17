# run.py — Runner chính xác cho Q-learning, SARSA, Monte Carlo, Policy Iteration, Value Iteration
# Fix: tự nhận dạng Q-table, policy dạng action_id hoặc next_node, không fallback sai
# Không cần chỉnh ở các file train — chỉ thay thế file này là đủ

import argparse, pickle, random, numpy as np
from pathlib import Path
from maze_env import MazeEnv
from maze import Maze

# ===== [0] Toạ độ & tiện ích cơ bản =====
DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # UP, RIGHT, DOWN, LEFT

def node_to_col_row(maze, node):
    h = maze.height
    return node // h, node % h

def col_row_to_node(maze, col, row):
    return col * maze.height + row

def neighbor_of(env: MazeEnv, s: int, a: int):
    c, r = node_to_col_row(env.maze, s)
    dc, dr = DIRS[a]
    nc, nr = c + dc, r + dr
    if not (0 <= nc < env.maze.width and 0 <= nr < env.maze.height):
        return None
    nxt = col_row_to_node(env.maze, nc, nr)
    return nxt if nxt in env.neighbors[s] else None

def valid_actions(env: MazeEnv, s: int):
    return [a for a in range(4) if neighbor_of(env, s, a) is not None]

# ===== [1] Hàm hỗ trợ =====
def manhattan_to_goal(env, s):
    sc, sr = node_to_col_row(env.maze, s)
    gc, gr = node_to_col_row(env.maze, env.goal)
    return abs(sc - gc) + abs(sr - gr)

# ===== [2] Nhận dạng & chuẩn hoá mô hình =====
def detect_model(model_obj):
    """Tự động nhận dạng Q-table, policy, value"""
    if isinstance(model_obj, np.ndarray) and model_obj.ndim == 2 and model_obj.shape[1] == 4:
        return "q_table", model_obj
    if isinstance(model_obj, dict):
        # {"Q": ndarray}
        if "Q" in model_obj and isinstance(model_obj["Q"], np.ndarray):
            return "q_table", model_obj["Q"]
        # {"policy": dict}
        if "policy" in model_obj and isinstance(model_obj["policy"], dict):
            return "policy_dict", model_obj["policy"]
        # {"theta":..., "V":...} (policy gradient loại bỏ)
        if "policy" not in model_obj and all(isinstance(k, (int,np.integer)) for k in model_obj.keys()):
            return "policy_dict", model_obj
    raise ValueError("Không nhận dạng được model. Hỗ trợ: ndarray (nS,4), {'Q':...}, {'policy':...}, hoặc dict state→next_node/action_id.")

def normalize_policy_dict(env, raw_policy: dict):
    """Chuyển {state: action_id hoặc next_node} -> {state: next_node}"""
    norm = {}
    for s, v in raw_policy.items():
        try:
            v = int(v)
        except:
            continue
        # Nếu là action_id (0–3)
        if v in (0,1,2,3):
            nn = neighbor_of(env, s, v)
            if nn is not None:
                norm[s] = nn
        else:
            # Nếu là node thật (có trong hàng xóm)
            if v in env.neighbors.get(s, []):
                norm[s] = v
    return norm

# ===== [3] Chọn hành động =====
def choose_from_q(env, q_table, s):
    acts = valid_actions(env, s)
    if not acts: return None
    q_vals = [q_table[s, a] for a in acts]
    return acts[int(np.argmax(q_vals))]

def choose_from_policy_dict(env, pol_nextnode: dict, s: int):
    nn = pol_nextnode.get(s)
    if nn is None: return None
    for a in valid_actions(env, s):
        if neighbor_of(env, s, a) == nn:
            return a
    return None

# ===== [4] Chạy 1 episode =====
def run_one_episode(env, model_type, model_data, max_steps=4000, visualize=False, use_shaping=False):
    if visualize:
        import pygame
        clock = pygame.time.Clock()

    s = env.reset()
    total = 0.0
    reached_goal = False

    for step in range(max_steps):
        if visualize:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total, True, reached_goal
            env.render(); pygame.event.pump()

        if model_type == "q_table":
            a = choose_from_q(env, model_data, s)
        elif model_type == "policy_dict":
            a = choose_from_policy_dict(env, model_data, s)
        else:
            a = None

        if a is None:
            acts = valid_actions(env, s)
            if not acts:
                total += env.invalid_penalty
                break
            a = random.choice(acts)

        nn = neighbor_of(env, s, a)
        if nn is None:
            total += env.invalid_penalty
            continue

        if nn == env.goal:
            reached_goal = True

        s_old = s
        s, r, done = env.step(nn)

        if use_shaping:
            d_prev = manhattan_to_goal(env, s_old)
            d_next = manhattan_to_goal(env, s)
            r += 0.03 * (d_prev - d_next)

        total += r
        if done: break
        if visualize:
            clock.tick(30)

    return total, False, reached_goal

# ===== [5] Đánh giá nhiều episode =====
def evaluate(env, model_type, model_data, episodes=20, max_steps=4000, visualize=False, use_shaping=False):
    returns, successes = [], 0
    for _ in range(episodes):
        ep_ret, aborted, reached = run_one_episode(env, model_type, model_data, max_steps, visualize, use_shaping)
        if aborted: break
        returns.append(ep_ret)
        if reached: successes += 1
    sr = 100.0 * successes / max(1,len(returns))
    ar = float(np.mean(returns)) if returns else float("nan")
    return sr, ar, returns

# ===== [6] CLI chính =====
def main():
    ap = argparse.ArgumentParser(description="Runner 5 mô hình RL trên Maze")
    ap.add_argument("--model", required=True)
    ap.add_argument("--maze", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--visualize", type=int, default=0)
    ap.add_argument("--n_enemies", type=int, default=0)
    ap.add_argument("--enemy_move", type=str, default="random", choices=["random","chase","still"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--use_shaping_eval", type=int, default=0)
    ap.add_argument("--eval_step_penalty", type=float, default=-0.1)
    ap.add_argument("--eval_goal_reward", type=float, default=10.0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    with open(args.model, "rb") as f:
        model_obj = pickle.load(f)
    model_type, model_data = detect_model(model_obj)

    myMaze = Maze().load(args.maze)
    env = MazeEnv(
        myMaze,
        visualize=bool(args.visualize),
        n_enemies=args.n_enemies,
        enemy_move=args.enemy_move,
        step_penalty=args.eval_step_penalty,
        invalid_penalty=-1.0,
        goal_reward=args.eval_goal_reward,
        enemy_collision_penalty=-20.0,
        seed=args.seed,
    )

    # chuẩn hoá policy (nếu cần)
    if model_type == "policy_dict":
        model_data = normalize_policy_dict(env, model_data)

    print(f"[RUN] model={args.model}")
    print(f"[RUN] maze={args.maze}")
    print(f"[RUN] Detected model_type={model_type}")
    print(f"[RUN] n_enemies={args.n_enemies} | enemy_move={args.enemy_move}")
    print(f"[RUN] episodes={args.episodes} | visualize={bool(args.visualize)}")
    print(f"[RUN] shaping_eval={bool(args.use_shaping_eval)} | step_penalty={args.eval_step_penalty} | goal_reward={args.eval_goal_reward}")

    try:
        sr, ar, rets = evaluate(env, model_type, model_data,
                                episodes=args.episodes, max_steps=args.max_steps,
                                visualize=bool(args.visualize),
                                use_shaping=bool(args.use_shaping_eval))
        print(f"[RESULT] Success Rate: {sr:.1f}% | Avg Return: {ar:.2f}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
