import numpy as np
import matplotlib.pyplot as plt
from random import randrange, shuffle
import pickle
import datetime
import random
from collections import deque


# ======================== MAZE GENERATION (ITERATIVE) ========================
def generate_adjacency_matrix(w, h, seed=None):
    """
    Sinh mê cung bằng DFS (randomized) dạng iterative để tránh RecursionError.
    Node index: n = x*h + y (column-major).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    adjacency = np.zeros((h * w, h * w), dtype='float64')
    visited = np.zeros((w, h), dtype=bool)

    def pos2node(x, y):
        return int(x * h + y)

    # Chọn điểm xuất phát ngẫu nhiên
    sx, sy = randrange(w), randrange(h)
    stack = [(sx, sy)]
    visited[sx, sy] = True

    while stack:
        x, y = stack[-1]

        neighbors = []
        if x - 1 >= 0 and not visited[x - 1, y]:
            neighbors.append((x - 1, y))
        if y + 1 < h and not visited[x, y + 1]:
            neighbors.append((x, y + 1))
        if x + 1 < w and not visited[x + 1, y]:
            neighbors.append((x + 1, y))
        if y - 1 >= 0 and not visited[x, y - 1]:
            neighbors.append((x, y - 1))
        shuffle(neighbors)

        if neighbors:
            xx, yy = neighbors[0]  # lấy 1 hàng xóm chưa thăm
            # Mở tường (nối cạnh)
            a = pos2node(x, y)
            b = pos2node(xx, yy)
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0

            visited[xx, yy] = True
            stack.append((xx, yy))
        else:
            stack.pop()

    return adjacency


# =============================== MAZE CLASS =================================
class Maze(object):
    def __init__(self,
                 adjacency=None,
                 maze_size=(20, 10),
                 startNode=0,
                 sinkerNode=None,
                 seed=None,
                 loops_k: int = 0,        # mở thêm tường để tạo vòng
                 min_dist_from_SG=None,   # khoảng cách Manhattan tối thiểu tới S/G cho enemy
                 ):
        self.maze_size = maze_size
        self.width = int(self.maze_size[0])
        self.height = int(self.maze_size[1])

        if adjacency is None:
            self.adjacency = generate_adjacency_matrix(self.width, self.height, seed=seed)
        else:
            assert adjacency.shape == (self.width * self.height, self.width * self.height), \
                "Adjacency must have shape (W*H, W*H)"
            self.adjacency = adjacency

        self.total_nodes = self.width * self.height
        self.startNode = startNode
        self.sinkerNode = self.width * self.height - 1 if sinkerNode is None else sinkerNode

        self.vertical_links = (self.height - 1) * self.width
        self.horizontal_links = (self.width - 1) * self.height
        self.total_links = self.vertical_links + self.horizontal_links

        # Autoscale khoảng cách tối thiểu
        self.min_dist_from_SG = self._autoscale_min_dist() if min_dist_from_SG is None else int(min_dist_from_SG)

        # Tạo vòng trước (nếu muốn) để có nhiều đường → enemy “khôn” hơn
        if loops_k > 0:
            self.add_loops(loops_k=loops_k)

        # === Chọn enemy “khôn” với số lượng phụ thuộc size ===
        self.static_enemies = self._choose_enemies_safely_k(self._autoscale_enemy_count())
        # -----------------------------------------------------

    # ---------- AUTOSCALE ----------
    def _autoscale_min_dist(self):
        """
        Khoảng cách Manhattan tối thiểu tới S/G cho enemy (map nhỏ hơn thì nới lỏng).
        """
        N = self.total_nodes
        if N <= 16:   return 1   # 4x4
        if N <= 36:   return 2   # 6x6
        if N <= 64:   return 2   # 8x8
        if N <= 144:  return 3   # 12x12
        if N <= 256:  return 3   # 16x16
        if N <= 1024: return 4   # 32x32
        return 5                 # 64x64 trở lên

    def _autoscale_enemy_count(self):
        """
        Số enemy hợp lý theo size nhỏ:
          - 4x4, 6x6  → 1
          - 8x8       → 1 (tối đa 2 nếu anh muốn khó)
          - 12x12     → 2
        (giữ nhỏ để không khóa đường đi trên map nhỏ)
        """
        N = self.total_nodes
        if N <= 16:   return 1   # 4x4
        if N <= 36:   return 1   # 6x6
        if N <= 64:   return 1   # 8x8 (đề xuất 1; nếu muốn khó: tăng thủ công)
        if N <= 144:  return 2   # 12x12
        if N <= 256:  return 2   # 16x16
        if N <= 1024: return 3   # 32x32
        return 4                 # 64x64+

    # ---------- COORD HELPERS ----------
    def _manhattan(self, a, b):
        ax, ay = (a // self.height), (a % self.height)
        bx, by = (b // self.height), (b % self.height)
        return abs(ax - bx) + abs(ay - by)

    def _ok_far_from_SG(self, n):
        return (self._manhattan(n, self.startNode) >= self.min_dist_from_SG and
                self._manhattan(n, self.sinkerNode) >= self.min_dist_from_SG)

    # ---------- GRAPH HELPERS ----------
    def graph_degrees(self):
        """Độ bậc của mỗi node trong đồ thị mê cung."""
        return (self.adjacency > 0).sum(axis=1).astype(int)

    def bfs_shortest_path(self, start=None, goal=None):
        """Trả về 1 đường ngắn nhất S->G (list node). Nếu không có, trả []."""
        s = self.startNode if start is None else start
        g = self.sinkerNode if goal is None else goal
        n = self.total_nodes
        prev = [-1] * n
        q = deque([s])
        seen = {s}
        while q:
            u = q.popleft()
            if u == g:
                break
            nbrs = np.where(self.adjacency[u] > 0)[0]
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    prev[v] = u
                    q.append(v)
        if prev[g] == -1 and s != g:
            return []
        # reconstruct
        path = []
        cur = g
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def exists_path_without(self, blocked_nodes:set, start=None, goal=None):
        """Kiểm tra còn đường S->G nếu chặn 'blocked_nodes'."""
        s = self.startNode if start is None else start
        g = self.sinkerNode if goal is None else goal
        if s in blocked_nodes or g in blocked_nodes:
            return False
        q = deque([s])
        seen = {s} | set(blocked_nodes)
        while q:
            u = q.popleft()
            if u == g:
                return True
            nbrs = np.where(self.adjacency[u] > 0)[0]
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    # ---------- THUẬT TOÁN CHỌN k ENEMY “KHÔN” ----------
    def _choose_enemies_safely_k(self, k: int, tries: int = 2000):
        """
        Chọn k vị trí enemy thoả mãn:
        - Không nằm trên đường ngắn nhất S→G (shield shortest path)
        - Không là dead-end (deg >= 2) — có fallback mềm nếu hiếm
        - Xa Start/Goal (min_dist_from_SG)
        - Path-safe tập hợp: khi chặn đồng thời k vị trí, vẫn còn ít nhất một đường S→G
        """
        if k <= 0:
            return []

        S, G = self.startNode, self.sinkerNode
        deg = self.graph_degrees()
        shortest = set(self.bfs_shortest_path(S, G))

        # Tập candidate “đẹp”
        candidates = [i for i in range(self.total_nodes)
                      if i not in {S, G}
                      and i not in shortest
                      and deg[i] >= 2
                      and self._ok_far_from_SG(i)]

        # Fallback mềm nếu quá khắt khe
        if len(candidates) < k:
            candidates = [i for i in range(self.total_nodes)
                          if i not in {S, G}
                          and i not in shortest
                          and self._ok_far_from_SG(i)]

        if len(candidates) == 0:
            return []

        candidates = list(set(candidates))  # unique
        if len(candidates) <= k:
            if self.exists_path_without(set(candidates)):
                return candidates
            # không path-safe → giảm dần
            cand = candidates[:]
            while len(cand) > 0 and not self.exists_path_without(set(cand)):
                cand.pop()
            return cand

        best = None
        for _ in range(tries):
            chosen = random.sample(candidates, k)
            if self.exists_path_without(set(chosen)):
                return chosen
            if best is None:
                best = chosen

        chosen = best or random.sample(candidates, min(k, len(candidates)))
        while len(chosen) > 0 and not self.exists_path_without(set(chosen)):
            chosen.pop(random.randrange(len(chosen)))
        return chosen

    # ---------- TẠO VÒNG (tuỳ chọn) ----------
    def add_loops(self, loops_k: int = 1, max_attempts: int = 4000):
        """
        Mở thêm 'loops_k' bức tường để tạo vòng. Mỗi lần mở là nối 2 node kề nhau đang bị tường chặn.
        Trả về số tường đã mở thực tế.
        """
        h, w = self.height, self.width
        n = self.total_nodes
        opened = 0
        attempts = 0

        while opened < loops_k and attempts < max_attempts:
            attempts += 1
            a = random.randrange(n)
            ax, ay = (a // h), (a % h)
            candidates = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                bx, by = ax + dx, ay + dy
                if 0 <= bx < w and 0 <= by < h:
                    b = bx * h + by
                    if self.adjacency[a, b] == 0:
                        candidates.append(b)
            if not candidates:
                continue
            b = random.choice(candidates)
            self.adjacency[a, b] = 1.0
            self.adjacency[b, a] = 1.0
            opened += 1

        return opened

    # ---------- PROPERTIES ----------
    @property
    def startNode(self):
        return self._startNode

    @startNode.setter
    def startNode(self, value):
        if value is None:
            self._startNode = 0
        else:
            assert 0 <= value < self.total_nodes, "startNode is outside the node range"
            self._startNode = value

    @property
    def sinkerNode(self):
        return self._sinkerNode

    @sinkerNode.setter
    def sinkerNode(self, value):
        if value is None:
            self._sinkerNode = self.width * self.height - 1
        else:
            assert 0 <= value < self.total_nodes, "sinkerNode is outside the node range"
            self._sinkerNode = value

    # ---------- MAZE MAP & VẼ ----------
    def generate_maze_map(self):
        maze_map = np.zeros([2 * self.height - 1, 2 * self.width - 1], dtype='int')
        for j in range(self.width - 1):  # upper triangle
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

    def plot_maze(self, show_nodes=False, show_links=False, show_ticks=False, show=True):
        """
        - Xanh lam: Start
        - Đỏ: Goal
        - Cam: Enemies (k vị trí “khôn” & path-safe)
        """
        maze_map = self.generate_maze_map()
        xshift = 0.33
        yshift = 0.33
        cmap = plt.colormaps.get_cmap('gray')
        norm = plt.Normalize(maze_map.min(), maze_map.max())
        img = cmap(norm(maze_map))

        # Start (xanh)
        if self.startNode is not None:
            x, y = self.node2xy(self.startNode)
            img[y, x, :3] = (0, 0, 1)

        # Goal (đỏ)
        if self.sinkerNode is not None:
            x, y = self.node2xy(self.sinkerNode)
            img[y, x, :3] = (1, 0, 0)

        # Enemies (cam)
        for e in getattr(self, "static_enemies", []):
            x, y = self.node2xy(e)
            img[y, x, :3] = (1, 0.6, 0)

        if show:
            if show_nodes:
                for n in range(self.height * self.width):
                    x, y = self.node2xy(n)
                    plt.text(x - xshift, y - yshift, str(n), fontweight='bold')

            if show_links:
                for n in range(1, (self.height - 1) * self.width + (self.width - 1) * self.height + 1):
                    x, y = self.link2xy(n)
                    plt.text(x, y - yshift, str(n), style='italic', color='red')

            if show_ticks:
                plt.xticks(np.arange(0, img.shape[1], step=4),
                           np.arange(0, (img.shape[1] - 1) / 2, step=2, dtype='int'))
                plt.yticks(np.arange(0, img.shape[0], step=4),
                           np.arange(0, (img.shape[0] - 1) / 2, step=2, dtype='int'))
            else:
                plt.xticks([]); plt.yticks([])

            ax = plt.imshow(img, origin='lower'); plt.show()
        else:
            ax = None
        return img, ax

    # ---------- LINK & COORD ----------
    def set_link(self, link=None, value=None):
        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (link - 1) % (self.height - 1)
            col = row + 1
        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height
        self.adjacency[row, col] = value
        self.adjacency[col, row] = value
        return value

    def get_link(self, link=None):
        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (self.height - 1)
            col = row + 1
        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height
        return self.adjacency[row, col]

    def reverse_link(self, link=None):
        value = self.get_link(link)
        if value == 0:
            self.set_link(link, value=np.float64(1)); return np.float64(1)
        elif value > 0:
            self.set_link(link, value=np.float64(0)); return np.float64(0)

    def xy2node(self, x, y):
        assert 0 <= x <= 2 * self.width + 1
        assert 0 <= y <= 2 * self.height + 1
        if x % 2 == 0 and y % 2 == 0:
            return int((x // 2) * self.height + y // 2)
        return np.nan

    def node2xy(self, n):
        return 2 * (n // self.height), 2 * (n % self.height)

    def xy2link(self, x, y):
        if 0 <= x <= (self.width - 1) * 2 and x % 2 == 0:
            n = int((x // 2) * (self.height - 1) + (y - 1) // 2 + 1)
        elif 0 <= y <= (self.height - 1) * 2 and y % 2 == 0:
            n = (x - 1) // 2 * self.height + y // 2 + 1
            n = int(n + (self.height - 1) * self.width)
        else:
            n = np.nan
        return n

    def link2xy(self, n):
        if 1 <= n <= self.vertical_links:
            x, y = 2 * ((n - 1) // (self.height - 1)), 2 * ((n - 1) % (self.height - 1)) + 1
        elif self.vertical_links < n <= self.total_links:
            n = n - self.vertical_links
            x, y = 2 * ((n - 1) // self.height) + 1, 2 * ((n - 1) % self.height)
        else:
            x, y = np.nan, np.nan
        return x, y

    # ---------- SAVE / LOAD ----------
    def save(self, filename=None):
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_maze'
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        return filename

    def load(self, filename):
        assert filename is not None, "filename parameter in Maze.load() is None"
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        return self


# ================================ MAIN ======================================
if __name__ == "__main__":
    print('Batch generating mazes...')
    from pathlib import Path
    from datetime import datetime
    import time as _time

    def ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    root = Path.home() / "Desktop" / "Maze" / "maze"
    ensure_dir(root)

    # ⬇️ Chỉ tạo bốn map nhỏ: 4x4, 6x6, 8x8, 12x12
    sizes = [(4, 4), (6, 6), (8, 8), (12, 12)]

    for w, h in sizes:
        seed = int(_time.time() * 1000) % (2**31 - 1)

        # gợi ý số vòng theo size nhỏ (ít mở để tránh làm map quá “phẳng”)
        if (w, h) == (4, 4):    loops = 1
        elif (w, h) == (6, 6):  loops = 2
        elif (w, h) == (8, 8):  loops = 3
        else:                   loops = 4  # 12x12

        myMaze = Maze(maze_size=(w, h), seed=seed, loops_k=loops)
        print(f"Size {w}x{h} | seed={seed} | loops_k={loops} -> min_dist={myMaze.min_dist_from_SG}")
        print(f"Static enemies (k={len(myMaze.static_enemies)}): {myMaze.static_enemies}")

        # Xem nhanh nếu muốn:
        myMaze.plot_maze(show_nodes=False, show_links=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = root / f"maze_{w}x{h}_{ts}.pkl"
        myMaze.save(str(out))
        print(f"Saved: {out}")
