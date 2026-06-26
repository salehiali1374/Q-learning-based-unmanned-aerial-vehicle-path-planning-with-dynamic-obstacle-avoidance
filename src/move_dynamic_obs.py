import numpy as np

def move_dynamic_obs(dyn_pos, obs_mask, grid_size, source, goal):
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    new_pos = []
    for r, c in dyn_pos:
        valid = []
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if obs_mask[nr, nc]:
                continue
            if (nr, nc) == source or (nr, nc) == goal:
                continue
            valid.append((nr, nc))
        if valid:
            new_pos.append(valid[np.random.randint(len(valid))])
        else:
            new_pos.append((r, c))
    return new_pos
