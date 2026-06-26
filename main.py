import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.build_obstacle_mask import build_obstacle_mask
from src.astar_planner import astar_planner
from src.dijkstra_planner import dijkstra_planner
from src.sarsa_planner import sarsa_planner
from src.qlearning_original import qlearning_original
from src.qlearning_sdp import qlearning_sdp
from src.generate_dynamic_obstacles import generate_dynamic_obstacles
from src.plotting import plot_path, plot_cost_curve, plot_step_curve, plot_performance_bar

# =====================================================================
#  GLOBAL ENVIRONMENT PARAMETERS
# =====================================================================
GRID_SIZE = 25

# 0-based (paper uses 1-based: source=[1,1], goal=[25,15])
source = (0, 0)
goal   = (24, 14)

# Paper static_obs: [top_row, left_col, height, width] 1-based -> 0-based
static_obs = [
    (2,  3, 4, 2),
    (5, 11, 3, 3),
    (12,  3, 5, 2),
    (15, 13, 4, 3),
]

# =====================================================================
#  Q-LEARNING HYPERPARAMETERS  (Table 1)
# =====================================================================
base_params = dict(b=5000, alpha=0.3, epsilon=0.9, max_steps=3000)

EP_ORIG_STATIC = 1000
EP_ORIG_DYN    = 4000
EP_SARSA       = 1500
EP_PROP_STATIC = 500
EP_PROP_DYN    = 1500

N_RUNS = 1

ACTIONS   = [(-1,0),(1,0),(0,-1),(0,1)]
N_ACTIONS = 4
N_STATES  = GRID_SIZE * GRID_SIZE

print('=' * 58)
print(' UAV Q-Learning Path Planning (Sonny et al., 2023)')
print(f' Multi-run mode: N_RUNS = {N_RUNS}')
print('=' * 58)

# =====================================================================
#  BUILD STATIC OBSTACLE MASK
# =====================================================================
obs_mask = build_obstacle_mask(GRID_SIZE, static_obs)

# =====================================================================
#  HELPERS
# =====================================================================
def goal_sh(steps, cost):
    vals = steps[cost == 1]
    return float(np.min(vals)) if len(vals) > 0 else float('nan')

def goal_lo(steps, cost):
    vals = steps[cost == 1]
    return float(np.max(vals)) if len(vals) > 0 else float('nan')

def fmt_val(v):
    return '-' if (v != v) else str(int(v))

def fmt_mean_std(m, sd):
    if m != m:
        return '-'
    if sd != sd or sd == 0:
        return f'{m:.1f}'
    return f'{m:.1f} +/- {sd:.1f}'

output_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
#  1) A-STAR
# =====================================================================
print('\n[1/6] Running A-star...')
t0 = time.perf_counter()
path_astar = astar_planner(source, goal, obs_mask, GRID_SIZE)
t_astar = time.perf_counter() - t0
print(f'  A-star: time={t_astar:.5f} s, path length={len(path_astar)}')

# =====================================================================
#  2) DIJKSTRA
# =====================================================================
print('[2/6] Running Dijkstra...')
t0 = time.perf_counter()
path_dijkstra = dijkstra_planner(source, goal, obs_mask, GRID_SIZE)
t_dijkstra = time.perf_counter() - t0
print(f'  Dijkstra: time={t_dijkstra:.5f} s, path length={len(path_dijkstra)}')

# =====================================================================
#  3) SARSA
# =====================================================================
print(f'[3/6] Running SARSA ({EP_SARSA} episodes x {N_RUNS} runs)...')
cost_sarsa_all  = np.zeros((N_RUNS, EP_SARSA))
steps_sarsa_all = np.zeros((N_RUNS, EP_SARSA))
sh_sarsa = np.full(N_RUNS, float('nan'))
lo_sarsa = np.full(N_RUNS, float('nan'))
t_sarsa_vec = np.zeros(N_RUNS)
path_sarsa = []; best_len_sarsa = np.inf

for run in range(N_RUNS):
    np.random.seed(run + 1)
    p = {**base_params, 'episodes': EP_SARSA}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = sarsa_planner(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p)
    t_sarsa_vec[run] = time.perf_counter() - t0
    cost_sarsa_all[run]  = cost_r
    steps_sarsa_all[run] = steps_r
    sh_sarsa[run] = goal_sh(steps_r, cost_r)
    lo_sarsa[run] = goal_lo(steps_r, cost_r)
    if sh_sarsa[run] == sh_sarsa[run] and sh_sarsa[run] < best_len_sarsa:
        best_len_sarsa = sh_sarsa[run]; path_sarsa = path_r
    print(f'  run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_sarsa[run])}, longest={fmt_val(lo_sarsa[run])}')
t_sarsa = t_sarsa_vec.sum()

# =====================================================================
#  4) ORIGINAL Q-LEARNING — static only
# =====================================================================
print(f'[4/6] Running Original Q-learning static ({EP_ORIG_STATIC} episodes x {N_RUNS} runs)...')
cost_qls_all  = np.zeros((N_RUNS, EP_ORIG_STATIC))
steps_qls_all = np.zeros((N_RUNS, EP_ORIG_STATIC))
sh_qls = np.full(N_RUNS, float('nan'))
lo_qls = np.full(N_RUNS, float('nan'))
t_qls_vec = np.zeros(N_RUNS)
path_qlearn_static = []; best_len_qls = np.inf

for run in range(N_RUNS):
    np.random.seed(run + 1)
    p = {**base_params, 'episodes': EP_ORIG_STATIC}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = qlearning_original(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p, False, [])
    t_qls_vec[run] = time.perf_counter() - t0
    cost_qls_all[run]  = cost_r
    steps_qls_all[run] = steps_r
    sh_qls[run] = goal_sh(steps_r, cost_r)
    lo_qls[run] = goal_lo(steps_r, cost_r)
    if sh_qls[run] == sh_qls[run] and sh_qls[run] < best_len_qls:
        best_len_qls = sh_qls[run]; path_qlearn_static = path_r
    print(f'  run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_qls[run])}, longest={fmt_val(lo_qls[run])}')
t_qlearn_static = t_qls_vec.sum()

# =====================================================================
#  5) ORIGINAL Q-LEARNING — static + dynamic
# =====================================================================
print(f'[5/6] Running Original Q-learning dynamic ({EP_ORIG_DYN} episodes x {N_RUNS} runs)...')
cost_qld_all  = np.zeros((N_RUNS, EP_ORIG_DYN))
steps_qld_all = np.zeros((N_RUNS, EP_ORIG_DYN))
sh_qld = np.full(N_RUNS, float('nan'))
lo_qld = np.full(N_RUNS, float('nan'))
t_qld_vec = np.zeros(N_RUNS)
path_qlearn_dyn = []; best_len_qld = np.inf
dyn_obs_4_best = []

for run in range(N_RUNS):
    np.random.seed(run + 1)
    dyn_obs_4 = generate_dynamic_obstacles(4, GRID_SIZE, obs_mask, source, goal)
    p = {**base_params, 'episodes': EP_ORIG_DYN}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = qlearning_original(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p, True, dyn_obs_4)
    t_qld_vec[run] = time.perf_counter() - t0
    cost_qld_all[run]  = cost_r
    steps_qld_all[run] = steps_r
    sh_qld[run] = goal_sh(steps_r, cost_r)
    lo_qld[run] = goal_lo(steps_r, cost_r)
    if sh_qld[run] == sh_qld[run] and sh_qld[run] < best_len_qld:
        best_len_qld = sh_qld[run]; path_qlearn_dyn = path_r; dyn_obs_4_best = dyn_obs_4
    print(f'  run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_qld[run])}, longest={fmt_val(lo_qld[run])}')
t_qlearn_dyn = t_qld_vec.sum()

# =====================================================================
#  6) PROPOSED Q-LEARNING WITH SDP
# =====================================================================
print(f'[6/6] Running Proposed Q-learning with SDP ({N_RUNS} runs each)...')

# 6a) Static only
print(f'  6a) Static ({EP_PROP_STATIC} episodes x {N_RUNS} runs)...')
cost_pqs_all  = np.zeros((N_RUNS, EP_PROP_STATIC))
steps_pqs_all = np.zeros((N_RUNS, EP_PROP_STATIC))
sh_pqs = np.full(N_RUNS, float('nan'))
lo_pqs = np.full(N_RUNS, float('nan'))
t_pqs_vec = np.zeros(N_RUNS)
path_prop_static = []; best_len_pqs = np.inf

for run in range(N_RUNS):
    np.random.seed(run + 1)
    p = {**base_params, 'episodes': EP_PROP_STATIC}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = qlearning_sdp(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p, 0)
    t_pqs_vec[run] = time.perf_counter() - t0
    cost_pqs_all[run]  = cost_r
    steps_pqs_all[run] = steps_r
    sh_pqs[run] = goal_sh(steps_r, cost_r)
    lo_pqs[run] = goal_lo(steps_r, cost_r)
    if sh_pqs[run] == sh_pqs[run] and sh_pqs[run] < best_len_pqs:
        best_len_pqs = sh_pqs[run]; path_prop_static = path_r
    print(f'    run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_pqs[run])}')
t_prop_static = t_pqs_vec.sum()

# 6b) 2 dynamic
print(f'  6b) 2 dynamic ({EP_PROP_DYN} episodes x {N_RUNS} runs)...')
cost_pqd2_all  = np.zeros((N_RUNS, EP_PROP_DYN))
steps_pqd2_all = np.zeros((N_RUNS, EP_PROP_DYN))
sh_pqd2 = np.full(N_RUNS, float('nan'))
lo_pqd2 = np.full(N_RUNS, float('nan'))
t_pqd2_vec = np.zeros(N_RUNS)
path_prop_dyn2 = []; best_len_pqd2 = np.inf
dyn_obs_2_best = []

for run in range(N_RUNS):
    np.random.seed(run + 1)
    dyn_obs_2 = generate_dynamic_obstacles(2, GRID_SIZE, obs_mask, source, goal)
    p = {**base_params, 'episodes': EP_PROP_DYN}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = qlearning_sdp(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p, 2)
    t_pqd2_vec[run] = time.perf_counter() - t0
    cost_pqd2_all[run]  = cost_r
    steps_pqd2_all[run] = steps_r
    sh_pqd2[run] = goal_sh(steps_r, cost_r)
    lo_pqd2[run] = goal_lo(steps_r, cost_r)
    if sh_pqd2[run] == sh_pqd2[run] and sh_pqd2[run] < best_len_pqd2:
        best_len_pqd2 = sh_pqd2[run]; path_prop_dyn2 = path_r; dyn_obs_2_best = dyn_obs_2
    print(f'    run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_pqd2[run])}')
t_prop_dyn2 = t_pqd2_vec.sum()

# 6c) 4 dynamic
print(f'  6c) 4 dynamic ({EP_PROP_DYN} episodes x {N_RUNS} runs)...')
cost_pqd4_all  = np.zeros((N_RUNS, EP_PROP_DYN))
steps_pqd4_all = np.zeros((N_RUNS, EP_PROP_DYN))
sh_pqd4 = np.full(N_RUNS, float('nan'))
lo_pqd4 = np.full(N_RUNS, float('nan'))
t_pqd4_vec = np.zeros(N_RUNS)
path_prop_dyn4 = []; best_len_pqd4 = np.inf
dyn_obs_4b_best = []

for run in range(N_RUNS):
    np.random.seed(run + 1)
    dyn_obs_4b = generate_dynamic_obstacles(4, GRID_SIZE, obs_mask, source, goal)
    p = {**base_params, 'episodes': EP_PROP_DYN}
    t0 = time.perf_counter()
    path_r, steps_r, cost_r = qlearning_sdp(source, goal, obs_mask, GRID_SIZE, ACTIONS, N_ACTIONS, N_STATES, p, 4)
    t_pqd4_vec[run] = time.perf_counter() - t0
    cost_pqd4_all[run]  = cost_r
    steps_pqd4_all[run] = steps_r
    sh_pqd4[run] = goal_sh(steps_r, cost_r)
    lo_pqd4[run] = goal_lo(steps_r, cost_r)
    if sh_pqd4[run] == sh_pqd4[run] and sh_pqd4[run] < best_len_pqd4:
        best_len_pqd4 = sh_pqd4[run]; path_prop_dyn4 = path_r; dyn_obs_4b_best = dyn_obs_4b
    print(f'    run {run+1:2d}/{N_RUNS}: shortest={fmt_val(sh_pqd4[run])}')
t_prop_dyn4 = t_pqd4_vec.sum()

# =====================================================================
#  PERFORMANCE TABLE
# =====================================================================
nan = float('nan')

def _m(v): return float(np.nanmean(v))
def _s(v): return float(np.nanstd(v))

results = [
    dict(name='A-star',                               time=t_astar,        short_mean=float(len(path_astar)),    long_mean=nan,           short_std=nan, long_std=nan),
    dict(name='Dijkstra',                             time=t_dijkstra,     short_mean=float(len(path_dijkstra)), long_mean=nan,           short_std=nan, long_std=nan),
    dict(name='SARSA',                                time=t_sarsa,        short_mean=_m(sh_sarsa), long_mean=_m(lo_sarsa), short_std=_s(sh_sarsa), long_std=_s(lo_sarsa)),
    dict(name='Original Q-learning (static)',         time=t_qlearn_static,short_mean=_m(sh_qls),  long_mean=_m(lo_qls),  short_std=_s(sh_qls),  long_std=_s(lo_qls)),
    dict(name='Original Q-learning (static+dynamic)', time=t_qlearn_dyn,   short_mean=_m(sh_qld),  long_mean=_m(lo_qld),  short_std=_s(sh_qld),  long_std=_s(lo_qld)),
    dict(name='Proposed Q-learning (static only)',    time=t_prop_static,  short_mean=_m(sh_pqs),  long_mean=_m(lo_pqs),  short_std=_s(sh_pqs),  long_std=_s(lo_pqs)),
    dict(name='Proposed Q-learning (static+2dyn)',    time=t_prop_dyn2,    short_mean=_m(sh_pqd2), long_mean=_m(lo_pqd2), short_std=_s(sh_pqd2), long_std=_s(lo_pqd2)),
    dict(name='Proposed Q-learning (static+4dyn)',    time=t_prop_dyn4,    short_mean=_m(sh_pqd4), long_mean=_m(lo_pqd4), short_std=_s(sh_pqd4), long_std=_s(lo_pqd4)),
]

print()
print('=' * 99)
print(f' Table 2: Performance Comparison (mean +/- std over {N_RUNS} runs)')
print('=' * 99)
print(f'{"Parameter":<45} {"Training Time":>14} {"Shortest":>18} {"Longest":>18}')
print('-' * 99)
for r in results:
    print(f'{r["name"]:<45} {r["time"]:12.5f} s {fmt_mean_std(r["short_mean"], r["short_std"]):>18} {fmt_mean_std(r["long_mean"], r["long_std"]):>18}')

# =====================================================================
#  GENERATE ALL FIGURES
# =====================================================================
print('\nGenerating figures...')

# Fig 4: Static obstacle paths
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
plot_path(axes[0,0], path_astar,         obs_mask, GRID_SIZE, source, goal, None, 'A-star (Static)')
plot_path(axes[0,1], path_dijkstra,      obs_mask, GRID_SIZE, source, goal, None, 'Dijkstra (Static)')
plot_path(axes[1,0], path_sarsa,         obs_mask, GRID_SIZE, source, goal, None, 'SARSA (Static)',               (0.2,0.5,0.9))
plot_path(axes[1,1], path_qlearn_static, obs_mask, GRID_SIZE, source, goal, None, 'Original Q-learning (Static)', (0.2,0.5,0.9))
fig.suptitle(f'Fig. 4 - UAV Path Planning: Static Obstacles  (best of {N_RUNS} runs)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig4_static_obstacle_paths.png'), dpi=150)
plt.close(fig)

# Fig 5: Proposed approach paths
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
plot_path(axes[0,0], path_qlearn_static, obs_mask, GRID_SIZE, source, goal, None,           'Orig Q-learn (Static)',          (0.2,0.5,0.9))
plot_path(axes[0,1], path_prop_static,   obs_mask, GRID_SIZE, source, goal, None,           'Proposed Q-learn (Static)',      (0.2,0.5,0.9))
plot_path(axes[1,0], path_prop_dyn2,     obs_mask, GRID_SIZE, source, goal, dyn_obs_2_best, 'Proposed Q-learn (2 Dynamic)',   (0.2,0.5,0.9))
plot_path(axes[1,1], path_prop_dyn4,     obs_mask, GRID_SIZE, source, goal, dyn_obs_4b_best,'Proposed Q-learn (4 Dynamic)',   (0.2,0.5,0.9))
fig.suptitle(f'Fig. 5 - Proposed Approach: Static & Dynamic Obstacles  (best of {N_RUNS} runs)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig5_proposed_paths.png'), dpi=150)
plt.close(fig)

# Fig 6: Original Q-learning static
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cost_curve(ax1, cost_qls_all, 'r', f'Episode via Cost (mean+/-std, {N_RUNS} runs)', 10, 'sum')
plot_step_curve(ax2, steps_qls_all, 'b', f'Episode via Steps (mean+/-std, {N_RUNS} runs)')
fig.suptitle('Fig. 6 - Original Q-learning (No Dynamic Obstacles)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig6_orig_qlearn_static.png'), dpi=150)
plt.close(fig)

# Fig 7: Original Q-learning dynamic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cost_curve(ax1, cost_qld_all, 'r', f'Episode via Cost (mean+/-std, {N_RUNS} runs)', 10, 'sum')
plot_step_curve(ax2, steps_qld_all, 'b', f'Episode via Steps (mean+/-std, {N_RUNS} runs)')
fig.suptitle('Fig. 7 - Original Q-learning (Dynamic Obstacles)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig7_orig_qlearn_dynamic.png'), dpi=150)
plt.close(fig)

# Fig 8: Proposed static
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cost_curve(ax1, cost_pqs_all, 'r', f'Episode via Cost ({N_RUNS} runs)', 10, 'avg')
plot_step_curve(ax2, steps_pqs_all, 'b', f'Episode via Steps ({N_RUNS} runs)')
fig.suptitle('Fig. 8 - Proposed Q-learning (No Dynamic Obstacles)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig8_proposed_qlearn_static.png'), dpi=150)
plt.close(fig)

# Fig 9: Proposed 2 dynamic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cost_curve(ax1, cost_pqd2_all, 'r', f'Episode via Cost ({N_RUNS} runs)', 10, 'avg')
plot_step_curve(ax2, steps_pqd2_all, 'b', f'Episode via Steps ({N_RUNS} runs)')
fig.suptitle('Fig. 9 - Proposed Q-learning (2 Dynamic Obstacles)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig9_proposed_qlearn_2dyn.png'), dpi=150)
plt.close(fig)

# Fig 10: Proposed 4 dynamic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cost_curve(ax1, cost_pqd4_all, 'r', f'Episode via Cost ({N_RUNS} runs)', 10, 'avg')
plot_step_curve(ax2, steps_pqd4_all, 'b', f'Episode via Steps ({N_RUNS} runs)')
fig.suptitle('Fig. 10 - Proposed Q-learning (4 Dynamic Obstacles)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Fig10_proposed_qlearn_4dyn.png'), dpi=150)
plt.close(fig)

# Performance bar chart
fig = plt.figure(figsize=(11, 5))
plot_performance_bar(fig, results)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Performance_Summary_Table2.png'), dpi=150)
plt.close(fig)

print(f'All figures saved to: {output_dir}')
print('Done.')
