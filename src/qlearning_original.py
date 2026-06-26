import numpy as np
from .generate_dynamic_obstacles import generate_dynamic_obstacles
from .move_dynamic_obs import move_dynamic_obs
from .policy_rollout import policy_rollout

def qlearning_original(source, goal, obs_mask, grid_size, ACTIONS, n_actions, n_states, params, use_dynamic, dyn_obs_init=None):
    episodes  = params['episodes']
    max_steps = params['max_steps']
    alpha     = params['alpha']
    epsilon   = params['epsilon']
    buf_cap   = params['b']
    gamma_vec = np.linspace(0.1, 0.9, episodes)

    Q = 0.01 * np.random.rand(n_states, n_actions)
    steps_per_ep = np.zeros(episodes)
    cost_per_ep  = np.zeros(episodes)
    best_len      = np.inf
    best_path_idx = []

    source_state = source[0] * grid_size + source[1]
    n_dyn = len(dyn_obs_init) if (use_dynamic and dyn_obs_init) else 0

    buf      = np.zeros((buf_cap, 4), dtype=np.float64)
    buf_ptr  = 0
    buf_size = 0

    for ep in range(episodes):
        gamma = gamma_vec[ep]
        state = source_state

        if use_dynamic and n_dyn > 0:
            dyn_pos = generate_dynamic_obstacles(n_dyn, grid_size, obs_mask, source, goal)
        else:
            dyn_pos = []

        step = 0
        total_cost = 0.0
        path_ep = [state]
        done = False

        while True:
            step += 1
            if step > max_steps:
                break
            cr, cc = divmod(state, grid_size)

            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q[state, :]))

            dr, dc = ACTIONS[action]
            nr = max(0, min(grid_size-1, cr+dr))
            nc = max(0, min(grid_size-1, cc+dc))
            next_state = nr * grid_size + nc

            if use_dynamic and (nr, nc) in dyn_pos:
                total_cost -= 1
                buf[buf_ptr] = [state, action, -1, next_state]
                buf_ptr = (buf_ptr + 1) % buf_cap
                buf_size = min(buf_size + 1, buf_cap)
                sidx = np.random.randint(buf_size)
                s0, a0, r0, ns0 = int(buf[sidx,0]), int(buf[sidx,1]), buf[sidx,2], int(buf[sidx,3])
                td = r0 + gamma * np.max(Q[ns0, :])
                Q[s0, a0] = (1 - alpha) * Q[s0, a0] + alpha * td
                done = True

            if done:
                cost_per_ep[ep] = total_cost
                steps_per_ep[ep] = step
                break

            if obs_mask[nr, nc]:
                reward      = -1.0
                cost_reward = 0.0
                next_state  = state
            elif (nr, nc) == goal:
                reward      = 1.0
                cost_reward = 1.0
            else:
                reward      = 0.0
                cost_reward = 0.0

            total_cost += cost_reward

            buf[buf_ptr] = [state, action, reward, next_state]
            buf_ptr = (buf_ptr + 1) % buf_cap
            buf_size = min(buf_size + 1, buf_cap)
            sidx = np.random.randint(buf_size)
            s0, a0, r0, ns0 = int(buf[sidx,0]), int(buf[sidx,1]), buf[sidx,2], int(buf[sidx,3])
            td = r0 + gamma * np.max(Q[ns0, :])
            Q[s0, a0] = (1 - alpha) * Q[s0, a0] + alpha * td

            if next_state != state:
                path_ep.append(next_state)
            state = next_state

            if use_dynamic and dyn_pos:
                dyn_pos = move_dynamic_obs(dyn_pos, obs_mask, grid_size, source, goal)

            if (nr, nc) == goal:
                break

        steps_per_ep[ep] = step
        cost_per_ep[ep]  = total_cost

        lr, lc = divmod(path_ep[-1], grid_size)
        if step < best_len and (lr, lc) == goal:
            best_len = step
            best_path_idx = path_ep[:]

    if best_path_idx:
        best_path = [divmod(int(s), grid_size) for s in best_path_idx]
    else:
        best_path, _ = policy_rollout(Q, source, goal, obs_mask, grid_size, ACTIONS, n_actions, max_steps)

    return best_path, steps_per_ep, cost_per_ep
