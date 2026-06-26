import numpy as np
import heapq
from .reconstruct_path import reconstruct_path

def astar_planner(source, goal, obs_mask, grid_size):
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    COSTS   = [1,1,1,1, 2**0.5, 2**0.5, 2**0.5, 2**0.5]

    def octile(r, c):
        dx, dy = abs(r - goal[0]), abs(c - goal[1])
        return max(dx, dy) + (2**0.5 - 1) * min(dx, dy)

    g = np.full((grid_size, grid_size), np.inf)
    parent = np.full((grid_size, grid_size), -1, dtype=int)
    g[source] = 0
    heap = [(octile(*source), source[0], source[1])]
    closed = np.zeros((grid_size, grid_size), dtype=bool)

    while heap:
        _, cr, cc = heapq.heappop(heap)
        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        if (cr, cc) == goal:
            break
        for i, (dr, dc) in enumerate(ACTIONS):
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if obs_mask[nr, nc] or closed[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                if obs_mask[cr, nc] or obs_mask[nr, cc]:
                    continue
            new_g = g[cr, cc] + COSTS[i]
            if new_g < g[nr, nc]:
                g[nr, nc] = new_g
                parent[nr, nc] = cr * grid_size + cc
                heapq.heappush(heap, (new_g + octile(nr, nc), nr, nc))

    return reconstruct_path(parent, source, goal, grid_size)
