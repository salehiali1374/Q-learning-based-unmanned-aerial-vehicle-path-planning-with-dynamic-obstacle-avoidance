def reconstruct_path(parent, source, goal, grid_size):
    """
    parent: (grid_size, grid_size) int ndarray, value = r*gs+c of parent, -1 if none
    source, goal: (r, c) tuples, 0-based
    Returns: list of (r, c) from source to goal, or [] if no path
    """
    path = [goal]
    r, c = goal
    for _ in range(grid_size * grid_size):
        if (r, c) == source:
            break
        p = parent[r, c]
        if p < 0:
            return []
        r, c = divmod(p, grid_size)
        path.append((r, c))
    else:
        return []
    return list(reversed(path))
