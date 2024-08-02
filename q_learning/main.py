# Imports:
from padm_env import MazeEnv
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
train = True
visualize_results = True

learning_rate = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1500 # Number of episodes

primary_goal_coordinates = (3, 5)
secondary_goal_coordinates = (5, 11)
hell_state_coordinates =[(3, 11), (8,1), (7,4)]
positive_rewards = [(1, 5), (7,6), (3,9), (9, 9)]

# Execute:
if train:
    # Create an instance of the environment:
    env = MazeEnv(primary_goal_coordinates=primary_goal_coordinates,
                     secondary_goal_coordinates=secondary_goal_coordinates,
                     hell_state_coordinates=hell_state_coordinates,
                     positive_rewards=positive_rewards)

    # Train a Q-learning agent:
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      primary_goal_coordinates=primary_goal_coordinates,
                      secondary_goal_coordinates=secondary_goal_coordinates,
                      positive_rewards=positive_rewards,
                      q_values_path="q_table.npy")
