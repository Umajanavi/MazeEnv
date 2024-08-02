# Imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Training Q-learning agent
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initializing the Q-table:
    q_table = np.zeros((*env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        # Take actions in the environment until "Done" flag is triggered
        while True:
            # Exploration vs. Exploitation
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)
            total_reward += reward
            
            #! Updating the Q-values using the Q-value update rule
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            
            state = next_state

            #! Stop the episode if the agent reaches Goal or Hell-states
            if done:
                break

        #! Perform epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        

    #! Close the environment window
    env.close()
    print("Training finished.\n")

    #! Save the trained Q-table
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


def visualize_q_table(hell_state_coordinates, primary_goal_coordinates, secondary_goal_coordinates, positive_rewards, actions=["Up", "Down", "Right", "Left"], q_values_path="q_table.npy"):
    try:
        # Load Q-table
        q_table = np.load(q_values_path)

        # Create subplots for each action
        _, axes = plt.subplots(1, 4, figsize=(27, 5))

        # Iterate over actions
        for i, action in enumerate(actions):
            ax = axes[i]  # Select current subplot
            heatmap_data = q_table[:, :, i].copy()  # Copy Q-values for current action


            # Create mask to hide certain states in heatmap
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[primary_goal_coordinates[0], primary_goal_coordinates[1]] = True  # Mask out primary goal state
            mask[secondary_goal_coordinates[0], secondary_goal_coordinates[1]] = True  # Mask out secondary goal state
            for reward in positive_rewards:
                mask[reward[0], reward[1]] = True  # Mask out positive reward states
            for hell in hell_state_coordinates:
                mask[hell[0], hell[1]] = True  # Mask out hell states

            # Plot heatmap with annotations
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="plasma", ax=ax, cbar=False, mask=mask, annot_kws={"size": 7})

            # Annotate special states on heatmap
            ax.text(primary_goal_coordinates[1] + 0.5, primary_goal_coordinates[0] + 0.5, 'P', color='green', ha='center', va='center', weight='bold', fontsize=14)
            ax.text(secondary_goal_coordinates[1] + 0.5, secondary_goal_coordinates[0] + 0.5, 'S', color='blue', ha='center', va='center', weight='bold', fontsize=14)
            for idx, reward in enumerate(positive_rewards):
                label = f'R{idx + 1}'
                ax.text(reward[1] + 0.5, reward[0] + 0.5, label, color='purple', ha='center', va='center', weight='bold', fontsize=14)
            for hell in hell_state_coordinates:
                ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')  # Set title for subplot with action name

        plt.tight_layout()  # Adjust layout for better visualization
        plt.show()  # Display the heatmap visualization

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")