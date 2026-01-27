from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class HopcroftKarp:
    left_size: int
    right_size: int

    def __post_init__(self) -> None:
        self._adj: List[List[int]] = [[] for _ in range(self.left_size)]

    def add_edge(self, u: int, v: int) -> None:
        self._adj[u].append(v)

    def maximum_matching_size(self) -> int:
        pair_u = [-1] * self.left_size
        pair_v = [-1] * self.right_size
        dist = [0] * self.left_size

        def bfs() -> bool:
            q: deque[int] = deque()
            found_free_right = False

            for u in range(self.left_size):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1

            while q:
                u = q.popleft()
                for v in self._adj[u]:
                    u2 = pair_v[v]
                    if u2 == -1:
                        found_free_right = True
                    elif dist[u2] == -1:
                        dist[u2] = dist[u] + 1
                        q.append(u2)

            return found_free_right

        def dfs(u: int) -> bool:
            for v in self._adj[u]:
                u2 = pair_v[v]
                if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(self.left_size):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1
        return matching
