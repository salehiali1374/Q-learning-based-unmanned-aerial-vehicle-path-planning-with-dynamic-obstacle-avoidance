import numpy as np
from .policy_rollout import policy_rollout

def sarsa_planner(source, goal, obs_mask, grid_size, ACTIONS, n_actions, n_states, params):
    episodes  = params['episodes']
    max_steps = params['max_steps']
    alpha     = params['alpha']
    epsilon   = params['epsilon']
    gamma_vec = np.linspace(0.1, 0.9, episodes)

    Q = 0.01 * np.random.rand(n_states, n_actions)
    steps_per_ep = np.zeros(episodes)
    cost_per_ep  = np.zeros(episodes)
    best_len     = np.inf
    best_path_idx = []

    source_state = source[0] * grid_size + source[1]

    def eps_greedy(state):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state, :]))

    for ep in range(episodes):
        gamma = gamma_vec[ep]
        state = source_state
        action = eps_greedy(state)
        step = 0
        total_cost = 0
        path_ep = [state]

        while True:
            step += 1
            if step > max_steps:
                break
            cr, cc = divmod(state, grid_size)
            dr, dc = ACTIONS[action]
            nr = max(0, min(grid_size-1, cr+dr))
            nc = max(0, min(grid_size-1, cc+dc))

            if obs_mask[nr, nc]:
                reward = -1
                cost_reward = 0
                next_state = state
            elif (nr, nc) == goal:
                reward = 1
                cost_reward = 1
                next_state = nr * grid_size + nc
            else:
                reward = 0
                cost_reward = 0
                next_state = nr * grid_size + nc

            total_cost += cost_reward
            next_action = eps_greedy(next_state)

            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            if next_state != state:
                path_ep.append(next_state)
            state = next_state
            action = next_action

            if (nr, nc) == goal:
                break

        steps_per_ep[ep] = step
        cost_per_ep[ep]  = total_cost

        lr, lc = divmod(path_ep[-1], grid_size)
        if step < best_len and (lr, lc) == goal:
            best_len = step
            best_path_idx = path_ep[:]

    if best_path_idx:
        best_path = [divmod(s, grid_size) for s in best_path_idx]
    else:
        best_path, _ = policy_rollout(Q, source, goal, obs_mask, grid_size, ACTIONS, n_actions, max_steps)

    return best_path, steps_per_ep, cost_per_ep
