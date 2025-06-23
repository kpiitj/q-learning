import numpy as np
import matplotlib.pyplot as plt
import os

class GridEnv:
    def __init__(self, grid_size, target_position, terminal, obstacles):
        self.grid_size = grid_size
        self.target = target_position
        self.terminal = terminal
        self.obstacles = obstacles
        self.movement_map = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
        self.initialize_agent()

    def initialize_agent(self):
        self.current_position = [1, 1]
        return self.current_position

    def execute_action(self, movement):
        delta_x, delta_y = self.movement_map[movement]
        new_position = [self.current_position[0] + delta_x, self.current_position[1] + delta_y]
        if self.is_position_valid(new_position):
            self.current_position = new_position
        reward_value = self.calculate_reward(self.current_position if self.is_position_valid(new_position) else new_position)
        return self.current_position, reward_value, self.is_end_state(self.current_position)

    def is_position_valid(self, position):
        return (1 <= position[0] <= self.grid_size and 1 <= position[1] <= self.grid_size and position not in self.obstacles)

    def is_end_state(self, position):
        return position == self.target or position == self.terminal

    def calculate_reward(self, position):
        if position == self.target: return 5
        if position == self.terminal: return -5
        if not self.is_position_valid(position): return -1
        return 0

    def visualize_maze(self, strategy=None, show_strategy_flag=False):
        plt.figure()
        ax = plt.gca()

        for i in range(1, self.grid_size):
            ax.axhline(i, color='k', linestyle='-', linewidth=1)
            ax.axvline(i, color='k', linestyle='-', linewidth=1)

        # Plot maze elements
        ax.plot(self.target[1] - 0.5, self.target[0] - 0.5, 'gs', markersize=30)
        ax.plot(self.terminal[1] - 0.5, self.terminal[0] - 0.5, 'rs', markersize=30)
        for obstacle in self.obstacles:
            ax.plot(obstacle[1] - 0.5, obstacle[0] - 0.5, 'ks', markersize=30)
        ax.plot(self.current_position[1] - 0.5, self.current_position[0] - 0.5, 'ro', markersize=20)

        if show_strategy_flag and strategy is not None:
            self.display_strategy_arrows(strategy)

        ax.set_xlim([0, self.grid_size])
        ax.set_ylim([self.grid_size, 0])
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.set_aspect('equal')
        plt.title('Gridworld Environment')
        plt.grid(True)
        plt.show()

    def visualize_maze_and_save(self, strategy=None, show_strategy_flag=False, output_file=None):
        plt.figure()
        ax = plt.gca()
        for i in range(1, self.grid_size):
            ax.axhline(i, color='k', linestyle='-', linewidth=1)
            ax.axvline(i, color='k', linestyle='-', linewidth=1)
        ax.plot(self.target[1] - 0.5, self.target[0] - 0.5, 'gs', markersize=30)
        ax.plot(self.terminal[1] - 0.5, self.terminal[0] - 0.5, 'rs', markersize=30)
        for obstacle in self.obstacles:
            ax.plot(obstacle[1] - 0.5, obstacle[0] - 0.5, 'ks', markersize=30)
        ax.plot(self.current_position[1] - 0.5, self.current_position[0] - 0.5, 'ro', markersize=20)
        if show_strategy_flag and strategy is not None:
            self.display_strategy_arrows(strategy)
        ax.set_xlim([0, self.grid_size])
        ax.set_ylim([self.grid_size, 0])
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.set_aspect('equal')
        plt.title('Maze Environment Strategy')
        plt.grid(True)
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        plt.show()

    def display_strategy_arrows(self, strategy):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_pos = [row + 1, col + 1]
                if cell_pos in [self.target, self.terminal] or cell_pos in self.obstacles:
                    continue
                action_idx = strategy[row, col]
                if action_idx == 0: delta_x, delta_y = 0, -0.3  # N
                elif action_idx == 1: delta_x, delta_y = 0, 0.3  # S
                elif action_idx == 2: delta_x, delta_y = 0.3, 0  # E
                elif action_idx == 3: delta_x, delta_y = -0.3, 0  # W
                else: continue
                plt.arrow(col + 0.5, row + 0.5, delta_x, delta_y, head_width=0.1, head_length=0.1, fc='k', ec='k')


def create_output_directory():
    output_dir = "results_section1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_strategy_and_value_plots():
    maze_size = 5
    target_pos = [5, 5]
    terminal = [3, 5]
    obstacles = [[1, 4], [2, 2], [3, 3], [5, 3]]
    discount_factors = [0.1, 0.5, 0.9]
    learning_rate = 0.1
    exploration_rate = 0.1
    output_dir = create_output_directory()

    for discount_factor in discount_factors:
        print(f"\nTraining with discount factor = {discount_factor}")
        maze_env = GridEnv(maze_size, target_pos, terminal, obstacles)
        optimal_strategy, state_values, _ = q_learning(maze_env, learning_rate, discount_factor, exploration_rate)

        # Save strategy visualization
        strategy_file = os.path.join(output_dir, f'strategy_discount_{discount_factor}_section1.png')
        maze_env.visualize_maze_and_save(optimal_strategy, show_strategy_flag=True, output_file=strategy_file)

        # Save value function visualization
        plt.figure()
        plt.imshow(state_values, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title(f'State Values (discount factor = {discount_factor})')
        plt.xticks(np.arange(maze_size))
        plt.yticks(np.arange(maze_size))
        plt.grid(True)
        values_file = os.path.join(output_dir, f'state_values_discount_{discount_factor}_section1.png')
        plt.savefig(values_file, bbox_inches='tight')
        plt.show()


def analyze_performance_vs_exploration(discount_factor=0.9, learning_rate=0.1, exploration_rates=[0.1, 0.3, 0.5], training_episodes=100000):
    performance_data = {}
    for exploration_rate in exploration_rates:
        np.random.seed(1)
        maze_env = GridEnv(5, [5, 5], [3, 5], [[1, 4], [2, 2], [3, 3], [5, 3]])
        _, _, episode_lengths = q_learning(maze_env, learning_rate, discount_factor, exploration_rate, num_episodes=training_episodes)
        performance_data[exploration_rate] = episode_lengths
    return performance_data

# Q-learning implementation with different variable names
def q_learning(maze_env, learning_rate, discount_factor, exploration_rate, num_episodes=100000):
    maze_size = maze_env.grid_size
    available_actions = ['N', 'S', 'E', 'W']
    action_value_table = 0.001 * np.random.rand(maze_size * maze_size, len(available_actions))
    episode_step_counts = []

    def get_state_index(position): 
        return (position[0] - 1) * maze_size + (position[1] - 1)

    for episode in range(num_episodes):
        maze_env.initialize_agent()
        current_state_idx = get_state_index(maze_env.current_position)
        for step_count in range(maze_size * maze_size * 10):
            action_index = np.random.randint(4) if np.random.rand() < exploration_rate else np.argmax(action_value_table[current_state_idx])
            next_position, reward, episode_finished = maze_env.execute_action(available_actions[action_index])
            next_state_idx = get_state_index(next_position)
            action_value_table[current_state_idx, action_index] += learning_rate * (reward + discount_factor * np.max(action_value_table[next_state_idx]) - action_value_table[current_state_idx, action_index])
            current_state_idx = next_state_idx
            if episode_finished: 
                break
        episode_step_counts.append(step_count + 1)

    optimal_strategy = np.full((maze_size, maze_size), -1)
    state_value_function = np.full((maze_size, maze_size), np.nan)

    for row_idx in range(maze_size):
        for col_idx in range(maze_size):
            cell_coordinate = [row_idx + 1, col_idx + 1]
            state_idx = get_state_index(cell_coordinate)
            if cell_coordinate == maze_env.target:
                state_value_function[row_idx, col_idx] = 5
            elif cell_coordinate == maze_env.terminal:
                state_value_function[row_idx, col_idx] = -5
            elif cell_coordinate in maze_env.obstacles:
                continue
            else:
                state_value_function[row_idx, col_idx] = np.max(action_value_table[state_idx])
                optimal_strategy[row_idx, col_idx] = np.argmax(action_value_table[state_idx])

    return optimal_strategy, state_value_function, episode_step_counts


def visualize_performance_analysis(performance_log):
    output_dir = create_output_directory()
    plt.figure(figsize=(10, 6))
    for exploration_rate, step_counts in performance_log.items():
        averaged_chunks = np.mean(np.array(step_counts).reshape(-1, 1000), axis=1)
        plt.plot(np.arange(1, len(averaged_chunks) + 1) * 1000, averaged_chunks, label=f"exploration rate = {exploration_rate}")
    plt.title("Episode Length vs Training Episodes (discount factor = 0.9)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Steps to Reach Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    performance_file = os.path.join(output_dir, 'performance_analysis_discount_0.9_section1.png')
    plt.savefig(performance_file, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    generate_strategy_and_value_plots()

    print("\nAnalyzing performance for exploration rates = 0.1, 0.3, 0.5...")
    performance_log = analyze_performance_vs_exploration()
    visualize_performance_analysis(performance_log)