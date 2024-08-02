import pygame
import numpy as np
import gymnasium as gym

# Map Layout
MAP = [
    "YYYYYYYYYYYYY",
    "Y X         Y",
    "Y   XXX X X Y",
    "Y X X   X X Y",
    "YXX X XXX XXY",
    "Y         X Y",
    "Y  X X XXXX Y",
    "YX X X    X Y",
    "Y  XXXXXX X Y",
    "Y       X   Y",
    "YYYYYYYYYYYYY", 
]

# Define constants
GRID_SIZE = (11, 13)
CELL_SIZE = min(800 // len(MAP[0]), 800 // len(MAP))

# Custom environment:
class MazeEnv(gym.Env):
    def __init__(self, primary_goal_coordinates, secondary_goal_coordinates, hell_state_coordinates, positive_rewards) -> None:
        super(MazeEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.state = None
        self.reward = 0
        self.info = {}
        self.done = False
        self.primary_goal = np.array(primary_goal_coordinates)
        self.secondary_goal = np.array(secondary_goal_coordinates)
        self.hell_states = [np.array(coord) for coord in hell_state_coordinates]
        self.positive_rewards = [np.array(coord) for coord in positive_rewards]
        self.reached_primary_goal = False
        self.fairy_reward_collected = False
        self.positive_reward_2_collected = False
        self.positive_reward_3_collected = False
        self.collected_rewards = set()  # Track collected rewards

        # Calculate screen dimensions
        self.screen_width = len(MAP[0]) * CELL_SIZE
        self.screen_height = len(MAP) * CELL_SIZE

        # Action-space
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.grid_size[0] - 1, self.grid_size[1] - 1]), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Cinderella")

        # Load Images:
        self.load_images()

    # Method: .reset()
    def reset(self):
        self.state = np.array([1, 1])
        self.done = False
        self.reward = 0
        self.reached_primary_goal = False
        self.fairy_reward_collected = False
        self.positive_reward_2_collected = False
        self.positive_reward_3_collected = False
        self.collected_rewards = set()
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.secondary_goal[0]) ** 2 +
            (self.state[1] - self.secondary_goal[1]) ** 2
        )
        return self.state, self.info
    
    # Method: step()
    def step(self, action):
        # Actions mapping
        dx, dy = 0, 0
        if action == 0 and self.state[0] > 0:  # up
            dy = -1
        elif action == 1 and self.state[0] < self.grid_size[0] - 1:  # down
            dy = 1
        elif action == 2 and self.state[1] < self.grid_size[1] - 1:  # right
            dx = 1
        elif action == 3 and self.state[1] > 0:  # left
            dx = -1

        new_y = self.state[0] + dy
        new_x = self.state[1] + dx

        # Checks for wall collision:
        if MAP[new_y][new_x] not in ['X', 'Y']:
            self.state[0] += dy
            self.state[1] += dx

        current_state = tuple(self.state)

        # Rewards:
        # Reward & Checks if primary goal has been reached:
        if np.array_equal(self.state, self.primary_goal) and not self.reached_primary_goal: 
            if current_state not in self.collected_rewards:
                self.reward += 50
                self.reached_primary_goal = True
                self.collected_rewards.add(current_state)
        # Reward & Checks if secondary goal has been reached after reaching the primary goal:
        elif np.array_equal(self.state, self.secondary_goal):
            if self.reached_primary_goal:
                self.reward += 100
                self.done = True
            else:
                self.done = True
                self.reward -= 50
        # Check if in positive rewards and it has not been collected:
        elif current_state in [tuple(reward) for reward in self.positive_rewards] and current_state not in self.collected_rewards:
            if current_state == tuple(self.positive_rewards[0]):
                self.reward += 15
                self.fairy_reward_collected = True
            elif current_state in [tuple(self.positive_rewards[1]), tuple(self.positive_rewards[2])]:
                self.reward += 10
                self.positive_reward_2_collected = True
            elif current_state == tuple(self.positive_rewards[3]) and self.reached_primary_goal:
                self.reward += 20
                self.positive_reward_3_collected = True
            self.collected_rewards.add(current_state)
        # Negative reward for hell states:
        elif current_state in [tuple(hell) for hell in self.hell_states]:
            self.reward -= 150
            self.done = True
        else:  # Every other state
            self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.secondary_goal[0]) ** 2 +
            (self.state[1] - self.secondary_goal[1]) ** 2
        )
        return self.state, self.reward, self.done, self.info


    # Load Images 
    def load_images(self):
        self.background_img = pygame.image.load("Background.webp")
        self.grass_img = pygame.image.load("grass.jpeg")
        self.positive_reward_3_img = pygame.image.load("Prince.png")
        self.agent_img = pygame.image.load("Maid.png")
        self.agent_img_2 = pygame.image.load("Cinderella.png")
        self.primary_goal_img = pygame.image.load("castle.jpeg")
        self.negative_reward_img = pygame.image.load("Evil_Sisters.png")
        self.positive_reward_1_img = pygame.image.load("Fairy.png")
        self.positive_reward_2_img = pygame.image.load("shoe.png")
        self.secondary_goal_img = pygame.image.load("win.jpg")

        # Scale images
        self.background_img = pygame.transform.scale(self.background_img, (self.screen_width, self.screen_height))
        self.grass_img = pygame.transform.scale(self.grass_img, (CELL_SIZE, CELL_SIZE))
        self.positive_reward_3_img = pygame.transform.scale(self.positive_reward_3_img, (CELL_SIZE, CELL_SIZE))
        self.agent_img = pygame.transform.scale(self.agent_img, (CELL_SIZE, CELL_SIZE))
        self.agent_img_2 = pygame.transform.scale(self.agent_img_2, (CELL_SIZE, CELL_SIZE))
        self.primary_goal_img = pygame.transform.scale(self.primary_goal_img, (CELL_SIZE, CELL_SIZE))
        self.negative_reward_img = pygame.transform.scale(self.negative_reward_img, (CELL_SIZE, CELL_SIZE))
        self.positive_reward_1_img = pygame.transform.scale(self.positive_reward_1_img, (CELL_SIZE, CELL_SIZE))
        self.positive_reward_2_img = pygame.transform.scale(self.positive_reward_2_img, (CELL_SIZE, CELL_SIZE))
        self.secondary_goal_img = pygame.transform.scale(self.secondary_goal_img, (CELL_SIZE, CELL_SIZE))

    # Method: render()
    def render(self):
        # Changing the background image
        self.screen.blit(self.background_img, (0, 0))

        # Draw the Maze:
        for y, row in enumerate(MAP):
            for x, cell in enumerate(row):
                if cell == 'X':  # Draws grass for the places of 'X'
                    grass = pygame.Rect((x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    self.screen.blit(self.grass_img, grass)
                elif cell == 'Y': 
                    pygame.draw.rect(self.screen, (205, 158, 206), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw the secondary goal:
        goal = pygame.Rect(self.secondary_goal[1] * CELL_SIZE, self.secondary_goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.screen.blit(self.secondary_goal_img, goal)

        # Draw the agent_1 and agent_2 (Dressed up Cinderella):
        agent = pygame.Rect(self.state[1] * CELL_SIZE, self.state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if self.fairy_reward_collected:
            self.screen.blit(self.agent_img_2, agent)
        else:
            self.screen.blit(self.agent_img, agent)

        # Draw the primary goal (Cinderella in Castle) if not reached:
        if not self.reached_primary_goal:
            primary_goal = pygame.Rect(self.primary_goal[1] * CELL_SIZE, self.primary_goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            self.screen.blit(self.primary_goal_img, primary_goal)

        # Draw negative reward (The Evil Sisters):
        for each_hell in self.hell_states:
            negative = pygame.Rect(each_hell[1] * CELL_SIZE, each_hell[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            self.screen.blit(self.negative_reward_img, negative)

        # Draw Positive Rewards if not collected
        for idx, reward in enumerate(self.positive_rewards):
            if tuple(reward) not in self.collected_rewards:
                if idx == 0:
                    positive = pygame.Rect(reward[1] * CELL_SIZE, reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    self.screen.blit(self.positive_reward_1_img, positive)
                elif idx in[1,2]:
                    positive = pygame.Rect(reward[1] * CELL_SIZE, reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    self.screen.blit(self.positive_reward_2_img, positive)
                elif idx == 3:
                    positive = pygame.Rect(reward[1] * CELL_SIZE, reward[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    self.screen.blit(self.positive_reward_3_img, positive)

        # Update contents on the window:
        pygame.display.flip()

    # Method: .Close()
    def close(self):
        pygame.quit()

# Function 1: Create an instance of the environment 
def create_env(primary_goal_coordinates, secondary_goal_coordinates, hell_state_coordinates, positive_rewards):
    env = MazeEnv(primary_goal_coordinates=primary_goal_coordinates, secondary_goal_coordinates=secondary_goal_coordinates, hell_state_coordinates=hell_state_coordinates, positive_rewards=positive_rewards)
    return env
