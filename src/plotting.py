import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_path(ax, path, obs_mask, grid_size, source, goal, dyn_obs=None, title_str='', dot_color='k'):
    """
    path: list of (r,c) tuples or None
    source, goal, dyn_obs: (r,c) tuples; dyn_obs is list
    """
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Grid lines
    for i in range(grid_size + 1):
        ax.axhline(i, color='#d0d0d0', lw=0.3)
        ax.axvline(i, color='#d0d0d0', lw=0.3)

    # Static obstacles
    for r in range(grid_size):
        for c in range(grid_size):
            if obs_mask[r, c]:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, color=(0.2, 0.5, 0.9), ec='none'))

    # Dynamic obstacles
    if dyn_obs:
        for r, c in dyn_obs:
            circ = plt.Circle((c + 0.5, r + 0.5), 0.35, color=(0.9, 0.1, 0.1), zorder=3)
            ax.add_patch(circ)

    # Path dots
    if path:
        for r, c in path:
            ax.plot(c + 0.5, r + 0.5, '.', color=dot_color, markersize=8, zorder=4)

    # Source: large black filled circle
    src_r, src_c = source
    circ_src = plt.Circle((src_c + 0.5, src_r + 0.5), 0.42, color='k', zorder=5)
    ax.add_patch(circ_src)

    # Goal: yellow square
    g_r, g_c = goal
    ax.add_patch(patches.Rectangle((g_c, g_r), 1, 1, color=(1, 0.9, 0), ec='k', lw=1, zorder=5))

    ax.set_title(title_str, fontsize=9)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_box_aspect(1)


def plot_cost_curve(ax, cost_input, color, title_str, window=10, mode='sum'):
    if cost_input.ndim == 1:
        cost_mat = cost_input.reshape(1, -1)
    else:
        cost_mat = cost_input
    n_runs, episodes = cost_mat.shape

    curves = np.zeros((n_runs, episodes))
    for r in range(n_runs):
        row = cost_mat[r]
        cs = np.cumsum(row)
        csum = cs.copy()
        if episodes > window:
            csum[window:] = cs[window:] - cs[:episodes - window]
        if mode == 'avg':
            wlen = np.minimum(np.arange(1, episodes + 1), window)
            curves[r] = csum / wlen
        else:
            curves[r] = csum

    mean_curve = curves.mean(axis=0)
    ep_axis = np.arange(1, episodes + 1)

    if n_runs > 1:
        std_curve = curves.std(axis=0)
        ax.fill_between(ep_axis, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=color)

    ax.plot(ep_axis, mean_curve, '-', color=color, lw=0.8)
    ax.set_xlabel('Episode', fontsize=9)
    ax.set_ylabel('Cost', fontsize=9)
    ax.set_title(title_str, fontsize=9)
    ax.set_xlim(1, episodes)
    ax.grid(True)


def plot_step_curve(ax, steps_input, color, title_str):
    if steps_input.ndim == 1:
        steps_mat = steps_input.reshape(1, -1)
    else:
        steps_mat = steps_input
    n_runs, episodes = steps_mat.shape

    mean_steps = steps_mat.mean(axis=0)
    ep_axis = np.arange(1, episodes + 1)

    if n_runs > 1:
        std_steps = steps_mat.std(axis=0)
        ax.fill_between(ep_axis, mean_steps - std_steps, mean_steps + std_steps,
                        alpha=0.15, color=color)

    ax.plot(ep_axis, mean_steps, '-', color=color, lw=0.8)
    ax.set_xlabel('Episode', fontsize=9)
    ax.set_ylabel('Steps', fontsize=9)
    ax.set_title(title_str, fontsize=9)
    ax.set_xlim(1, episodes)
    ax.grid(True)


def plot_performance_bar(fig, results):
    """
    results: list of dicts with keys: name, time, short_mean, long_mean, short_std, long_std
    """
    n = len(results)
    names     = [r['name']       for r in results]
    times     = [r['time']       for r in results]
    short     = [r['short_mean'] for r in results]
    short_std = [r.get('short_std', 0) or 0 for r in results]

    ax1 = fig.add_subplot(1, 2, 1)
    y = np.arange(n)
    ax1.barh(y, times, color=(0.2, 0.5, 0.8))
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=7.5)
    ax1.set_xlabel('Training Time (s)', fontsize=9)
    ax1.set_title('Training Time Comparison', fontsize=10, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True)

    ax2 = fig.add_subplot(1, 2, 2)
    short_plot = [s if s == s else 0 for s in short]  # nan -> 0
    ax2.barh(y, short_plot, color=(0.2, 0.75, 0.4))
    for i in range(n):
        if short_std[i] > 0 and short[i] == short[i]:
            ax2.errorbar(short[i], i, xerr=short_std[i], fmt='none', color='k', lw=1.5, capsize=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7.5)
    ax2.set_xlabel('Shortest Path Length (steps)', fontsize=9)
    ax2.set_title('Shortest Path Length Comparison', fontsize=10, fontweight='bold')
    ax2.grid(True)

    fig.suptitle('Table 2 - Performance Comparison Summary', fontsize=11, fontweight='bold')
