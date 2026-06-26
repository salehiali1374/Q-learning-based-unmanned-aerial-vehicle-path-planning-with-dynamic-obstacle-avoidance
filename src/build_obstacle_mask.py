import numpy as np

def build_obstacle_mask(grid_size, static_obs):
    """
    static_obs: list of (top_row, left_col, height, width), 0-based
    Returns: (grid_size, grid_size) bool ndarray, True = obstacle
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for r0, c0, h, w in static_obs:
        r_end = min(r0 + h, grid_size)
        c_end = min(c0 + w, grid_size)
        mask[r0:r_end, c0:c_end] = True
    return mask
