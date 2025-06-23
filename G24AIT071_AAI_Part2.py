import numpy as np
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import njit

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 11
        self.start = (1, 1)
        self.goal = (5, 11)
        self.terminal = (4, 7)

        self.obstacles = {
            (2, 3), (3, 3), (4, 3),  # Grid 1 obstacles
            (3, 9), (3, 10), (5, 9)  # Grid 2 obstacles
        }

        self.valid_states = []
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                # tunnel at (3,6)
                if c == 6 and r != 3:
                    continue
                # obstacles to skip
                if (r, c) not in self.obstacles:
                    self.valid_states.append((r, c))

        self.state_to_index = {s: i for i, s in enumerate(self.valid_states)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}

    def get_next_state(self, state, action):
        """
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        r, c = state

        # Handle tunnel connections
        if state == (3, 5) and action == 1:  # Moving right from (3,5)
            return (3, 6)
        if state == (3, 6) and action == 3:  # Moving left from (3,6)
            return (3, 5)
        if state == (3, 6) and action == 1:  # Moving right from (3,6)
            return (3, 7)
        if state == (3, 7) and action == 3:  # Moving left from (3,7)
            return (3, 6)

        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if nr < 1 or nr > self.rows or nc < 1 or nc > self.cols:
            return state
        if nc == 6 and nr != 3:
            return state
        if (nr, nc) in self.obstacles:
            return state
        return (nr, nc)

    def get_reward(self, state, next_state):
        if next_state == self.goal:
            return 5.0  # Goal reward
        elif next_state == self.terminal:
            return -5.0  # Terminal penalty
        elif next_state == state:  # Hit wall, obstacle, or boundary
            return -1.0
        else:
            return 0.0  # Step cost

    def state_id(self, state):
        return self.state_to_index.get(state, -1)

    def id_to_state(self, idx):
        return self.index_to_state.get(idx, (-1, -1))

    def is_terminal(self, state):
        return state == self.goal or state == self.terminal

@njit
def softmax_jit(q_values, beta):
    """Numerically stable softmax"""
    q_max = np.max(q_values)
    exp_q = np.exp(beta * (q_values - q_max))
    return exp_q / np.sum(exp_q)

@njit
def weighted_choice_jit(probs):
    """Choose action based on probabilities"""
    cdf = np.cumsum(probs)
    rand_val = np.random.random()
    for i in range(len(probs)):
        if rand_val <= cdf[i]:
            return i
    return len(probs) - 1

@njit
def run_q_learning_jit(Q, transitions, rewards, is_terminal_mask, start_id, alpha, gamma, beta, episodes, max_steps):
    steps_per_episode = np.zeros(episodes, dtype=np.int32)

    for ep in range(episodes):
        s_id = start_id
        step_count = 0

        while step_count < max_steps:
            if is_terminal_mask[s_id]:
                break

            probs = softmax_jit(Q[s_id], beta)
            action = weighted_choice_jit(probs)

            ns_id = transitions[s_id, action]
            r = rewards[s_id, action]

            if is_terminal_mask[ns_id]:
                Q[s_id, action] += alpha * (r - Q[s_id, action])
            else:
                max_q_next = np.max(Q[ns_id])
                Q[s_id, action] += alpha * (r + gamma * max_q_next - Q[s_id, action])

            s_id = ns_id
            step_count += 1

        steps_per_episode[ep] = step_count

    return Q, steps_per_episode

def q_learning_worker(worker_id, alpha, gamma, beta, episodes):
    """Single worker for Q-learning"""
    np.random.seed(worker_id * 1000 + 42)
    random.seed(worker_id * 1000 + 42)

    env = GridWorld()
    num_states = len(env.valid_states)
    Q = np.zeros((num_states, 4))
    max_steps =  5 * 11 * 10 #(grid size * 10)

    transitions = np.full((num_states, 4), -1, dtype=np.int32)
    rewards = np.zeros((num_states, 4), dtype=np.float32)
    is_terminal_mask = np.zeros(num_states, dtype=np.bool_)

    for s_id in range(num_states):
        state = env.id_to_state(s_id)
        is_terminal_mask[s_id] = env.is_terminal(state)

        for action in range(4):
            next_state = env.get_next_state(state, action)
            ns_id = env.state_id(next_state)

            if ns_id == -1:
                transitions[s_id, action] = s_id
                rewards[s_id, action] = -1.0
            else:
                transitions[s_id, action] = ns_id
                rewards[s_id, action] = env.get_reward(state, next_state)

    start_id = env.state_id(env.start)

    Q, steps_per_episode = run_q_learning_jit(
        Q, transitions, rewards, is_terminal_mask, start_id,
        alpha, gamma, beta, episodes, max_steps
    )

    return Q, list(steps_per_episode)

def q_learning_softmax_parallel(alpha=0.1, gamma=0.9, beta=0.1, episodes=200000, num_workers=4):
    """Run Q-learning with multiple workers"""
    print(f"Running Q-learning with {num_workers} workers...")
    episodes_per_worker = episodes // num_workers

    worker_args = [
        (worker_id, alpha, gamma, beta, episodes_per_worker)
        for worker_id in range(num_workers)
    ]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(q_learning_worker, worker_args)

    all_Qs = [result[0] for result in results]
    all_steps = []
    for result in results:
        all_steps.extend(result[1])

    #Q-tables avg
    Q_avg = np.mean(all_Qs, axis=0)
    # Convert to policy and values
    env = GridWorld()
    policy = np.full((env.rows, env.cols), -1)
    values = np.full((env.rows, env.cols), np.nan)

    for s_id in range(len(env.valid_states)):
        state = env.id_to_state(s_id)
        r, c = state

        if state == env.goal:
            values[r-1, c-1] = 5.0
        elif state == env.terminal:
            values[r-1, c-1] = -5.0
        else:
            values[r-1, c-1] = np.max(Q_avg[s_id])
            policy[r-1, c-1] = np.argmax(Q_avg[s_id])

    return policy, values, all_steps


def plot_policy(policy, title="Policy"):
    env = GridWorld()
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for i in range(env.rows + 1):
        ax.axhline(i, color='#333333', linewidth=0.8)  # Darker grid lines
    for j in range(env.cols + 1):
        ax.axvline(j, color='#333333', linewidth=0.8)

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r + 1, c + 1)

            if c == 5 and r != 2:  # Column 6, not tunnel
                rect = patches.Rectangle((c, r), 1, 1, facecolor='#F5F5DC', edgecolor='#333333')  # Beige
                ax.add_patch(rect)
            elif state in env.obstacles:
                rect = patches.Rectangle((c, r), 1, 1, facecolor='#696969', edgecolor='#333333')  # DimGray
                ax.add_patch(rect)
            elif state == env.goal:
                rect = patches.Rectangle((c, r), 1, 1, facecolor='#2E8B57', edgecolor='#333333')  # SeaGreen
                ax.add_patch(rect)
                ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=12, color='white', weight='bold')
            elif state == env.terminal:
                rect = patches.Rectangle((c, r), 1, 1, facecolor='#B22222', edgecolor='#333333')  # FireBrick
                ax.add_patch(rect)
                ax.text(c + 0.5, r + 0.5, 'T', ha='center', va='center', fontsize=12, color='white', weight='bold')
            else:
                rect = patches.Rectangle((c, r), 1, 1, facecolor='#F8F8FF', edgecolor='#333333')  # GhostWhite
                ax.add_patch(rect)
                if policy[r, c] != -1:
                    ax.text(c + 0.5, r + 0.5, action_arrows[policy[r, c]],
                           ha='center', va='center', fontsize=14, color='#191970')  # MidnightBlue

    plt.tight_layout()

    filename = f"policy_{title.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_').replace(',', '_').replace('γ', 'gamma').replace('β', 'beta')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Policy saved as: {filename}")
    plt.show()


def plot_value_function(values, title="Value Function"):
    env = GridWorld()
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_values = values.copy()
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r + 1, c + 1)
            if c == 5 and r != 2:  # Column 6, not tunnel
                plot_values[r, c] = np.nan
            elif state in env.obstacles:
                plot_values[r, c] = np.nan

    masked_values = np.ma.masked_invalid(plot_values)
    im = ax.imshow(masked_values, cmap='plasma', origin='upper', aspect='equal')  # Changed to plasma colormap

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r + 1, c + 1)
            if c == 5 and r != 2:
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2, edgecolor='#FF4500', facecolor='#F5F5DC', alpha=0.8)  # OrangeRed + Beige
                ax.add_patch(rect)
            elif state in env.obstacles:
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2, edgecolor='#333333', facecolor='#696969')  # DimGray
                ax.add_patch(rect)
            elif state == env.goal:
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=3, edgecolor='#32CD32', facecolor='none')  # LimeGreen
                ax.add_patch(rect)
            elif state == env.terminal:
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=3, edgecolor='#DC143C', facecolor='none')  # Crimson
                ax.add_patch(rect)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels(range(1, env.cols + 1))
    ax.set_yticklabels(range(1, env.rows + 1))

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    filename = f"value_function_{title.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_').replace(',', '_').replace('γ', 'gamma').replace('β', 'beta')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Value function saved as: {filename}")
    plt.show()


def plot_steps(steps_list, beta_val):
    plt.figure(figsize=(12, 6))

    # Smooth the data
    window_size = 1000
    if len(steps_list) > window_size:
        smoothed = np.convolve(steps_list, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed, linewidth=2, color='#4169E1')  # RoyalBlue
        plt.xlabel(f'Episode (smoothed every {window_size} episodes)')
    else:
        plt.plot(steps_list, linewidth=1, color='#4169E1')
        plt.xlabel('Episode')

    plt.title(f'Steps to Reach Goal\nγ=0.9, β={beta_val}', fontsize=14, fontweight='bold')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3, color='#DDDDDD')
    plt.tight_layout()

    filename = f"steps_gamma_0.9_beta_{beta_val}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Steps plot saved as: {filename}")
    plt.show()


def plot_combined_steps(all_steps_data, gamma_val):
    plt.figure(figsize=(12, 8))

    colors = ['#4169E1', '#FF8C00', '#2E8B57']  # RoyalBlue, DarkOrange, SeaGreen
    beta_values = sorted(all_steps_data.keys())

    for i, beta in enumerate(beta_values):
        steps_list = all_steps_data[beta]

        block_size = 1000
        num_blocks = len(steps_list) // block_size
        avg_steps_per_block = []

        for block in range(num_blocks):
            start_idx = block * block_size
            end_idx = start_idx + block_size
            avg_steps = np.mean(steps_list[start_idx:end_idx])
            avg_steps_per_block.append(avg_steps)

        episode_blocks = range(len(avg_steps_per_block))
        plt.plot(episode_blocks, avg_steps_per_block,
                color=colors[i], linewidth=2, label=f'beta={beta}', alpha=0.8)

    plt.title(f'Avg Steps vs Episodes (Varying Beta, gamma={gamma_val})', fontsize=14, fontweight='bold')
    plt.xlabel('Episode Block (1000 episodes per point)')
    plt.ylabel('Avg Steps to Goal')
    plt.legend()
    plt.grid(True, alpha=0.3, color='#DDDDDD')
    plt.tight_layout()

    filename = f"combined_steps_gamma_{gamma_val}_all_beta.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Combined steps plot saved as: {filename}")
    plt.show()

if __name__ == "__main__":
    #Different gamma values with beta=0.1
    print("\nPolicy and Value Function for different gamma values")
    print("-" * 60)

    gamma_values = [0.1, 0.5, 0.9]
    alpha = 0.1
    beta = 0.1
    episodes = 200000

    for gamma in gamma_values:
        print(f"\nRunning γ={gamma}, β={beta}, α={alpha}, episodes={episodes}")

        policy, values, steps = q_learning_softmax_parallel(
            alpha=alpha, gamma=gamma, beta=beta, episodes=episodes
        )

        avg_final_steps = np.mean(steps[-5000:]) 
        print(f"Average steps: {avg_final_steps:.1f}")

        plot_policy(policy, title=f"Policy (γ={gamma}, β={beta})")
        plot_value_function(values, title=f"Value Function (γ={gamma}, β={beta})")

    #Different beta values with gamma=0.9
    print("\n" + "=" * 60)
    print("Steps analysis for γ=0.9 with different beta values")
    print("-" * 60)

    gamma = 0.9
    beta_values = [0.1, 0.3, 0.5]
    all_steps_data = {}

    for beta in beta_values:
        print(f"\nRunning γ={gamma}, β={beta}, α={alpha}, episodes={episodes}")

        policy, values, steps = q_learning_softmax_parallel(
            alpha=alpha, gamma=gamma, beta=beta, episodes=episodes
        )

        avg_final_steps = np.mean(steps[-5000:])
        print(f"Average steps (final 5000 episodes): {avg_final_steps:.1f}")

        all_steps_data[beta] = steps
        plot_steps(steps, beta_val=beta)

    # Combined plot
    print(f"\nCreating combined plot for all beta values...")
    plot_combined_steps(all_steps_data, gamma_val=0.9)

    print("\n" + "=" * 60)
    print("Generated files:")
    print("policy + value function)")
    print("individual steps plots)")
    print("combined steps comparison")
    print("=" * 60)