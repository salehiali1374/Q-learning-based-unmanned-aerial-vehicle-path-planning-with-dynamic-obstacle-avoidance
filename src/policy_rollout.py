import numpy as np

def policy_rollout(Q, source, goal, obs_mask, grid_size, ACTIONS, n_actions, max_steps):
    VISIT_CAP = 4
    r, c = source
    path = [source]
    visit_count = np.zeros((grid_size, grid_size), dtype=int)
    visit_count[r, c] = 1
    reached = False

    for _ in range(max_steps):
        if (r, c) == goal:
            reached = True
            break
        state = r * grid_size + c
        sorted_a = np.argsort(Q[state, :])[::-1]
        moved = False
        nr2, nc2 = r, c
        for a in sorted_a:
            dr, dc = ACTIONS[a]
            nr, nc = max(0, min(grid_size-1, r+dr)), max(0, min(grid_size-1, c+dc))
            if nr == r and nc == c:
                continue
            if obs_mask[nr, nc]:
                continue
            if visit_count[nr, nc] >= VISIT_CAP:
                continue
            nr2, nc2 = nr, nc
            moved = True
            break
        if not moved:
            break
        visit_count[nr2, nc2] += 1
        path.append((nr2, nc2))
        r, c = nr2, nc2

    return path, reached
