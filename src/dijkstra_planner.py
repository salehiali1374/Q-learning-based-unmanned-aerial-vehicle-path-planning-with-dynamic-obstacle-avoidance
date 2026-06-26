import numpy as np
import heapq
from .reconstruct_path import reconstruct_path

def dijkstra_planner(source, goal, obs_mask, grid_size):
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    dist = np.full((grid_size, grid_size), np.inf)
    parent = np.full((grid_size, grid_size), -1, dtype=int)
    dist[source] = 0
    heap = [(0, source[0], source[1])]
    visited = np.zeros((grid_size, grid_size), dtype=bool)

    while heap:
        d, cr, cc = heapq.heappop(heap)
        if visited[cr, cc]:
            continue
        visited[cr, cc] = True
        if (cr, cc) == goal:
            break
        for dr, dc in ACTIONS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if obs_mask[nr, nc] or visited[nr, nc]:
                continue
            new_d = d + 1
            if new_d < dist[nr, nc]:
                dist[nr, nc] = new_d
                parent[nr, nc] = cr * grid_size + cc
                heapq.heappush(heap, (new_d, nr, nc))

    return reconstruct_path(parent, source, goal, grid_size)
