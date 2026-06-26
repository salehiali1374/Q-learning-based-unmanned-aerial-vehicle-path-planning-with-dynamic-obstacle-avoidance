import numpy as np

def generate_dynamic_obstacles(n, grid_size, obs_mask, source, goal):
    """Returns list of (r, c) tuples, 0-based, randomly placed."""
    placed = []
    attempts = 0
    while len(placed) < n and attempts < 10000:
        attempts += 1
        r = np.random.randint(grid_size)
        c = np.random.randint(grid_size)
        if obs_mask[r, c]:
            continue
        if (r, c) == source or (r, c) == goal:
            continue
        if (r, c) in placed:
            continue
        placed.append((r, c))
    return placed
