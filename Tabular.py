import gymnasium as gym  # provides taxi v3, the MDP
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    
    def __init__(self, n_states, n_actions, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, alpha=0.8):
        """
        Args:
            n_states: Number of states in the environment, for taxi v3 there are 500
            n_actions: Number of available actions which is 6
            gamma: Discount factor, used to balance the importance of future rewards compared to immediate ones
            alpha: Learning rate, determines how much the Q-values are updated during learning, a value of 0.1 means that the Q-values are updated by 10% of the new information obtained from the reward and the next state
            epsilon: Initial exploration rate, set to 1 to explore a lot at the beginning and decrease over time
            epsilon_decay: Decay factor for epsilon, this is the rate at which epsilon decreases after each episode, used in the decay_epsilon function
            epsilon_min: Minimum value of epsilon, a minimum value is kept to prevent it from becoming 0 and therefore never exploring again
        """
        
        self.n_states = n_states # 500
        self.n_actions = n_actions # 6 = south, north, west, east, pickup, dropoff
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 
        self.alpha = alpha 
        
        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions)) 
        # Q is a table that has states on the rows and actions on the columns,
        # where Q represents how good an action is in that state. 

    
    def select_action(self, state, training=True):

        if training and np.random.random() < self.epsilon: # the random number is between 0 and 1
            return np.random.randint(self.n_actions) # if a random number is less than epsilon, explore
        else:
            return np.argmax(self.Q[state]) # otherwise exploit current knowledge by choosing the action with the highest Q value which will provide the highest reward
        # in both cases the indices of the actions (0-5) are returned because that's how they are encoded in the taxi v3 environment
    
    def update(self, state, action, reward, next_state):

        td_target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * td_target
   
    
    def decay_epsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # decreases epsilon by multiplying it by the decay factor, but does not let it go below epsilon_min
    
    def get_greedy_action(self, state):
        
        return np.argmax(self.Q[state]) # returns the action with the highest Q value for the given state


def train_agent(env, agent, num_episodes=10000, print_interval=1000):

    rewards_history = []
    epsilon_history = []
    
    print("Starting training...")
    
    for episode in range(num_episodes): # each episode is a complete run from start to finish
        state, _ = env.reset()
        done = False
        truncated = False # truncated is used for environments with step limits like taxi v3
        total_reward = 0
        
        while not done and not truncated: # done stands for objective reached, truncated for maximum steps reached
            
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action) # executes the action in the environment and receives the next state, the reward, and the episode end indicators
            total_reward += reward
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Save metrics
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(rewards_history[-print_interval:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\nTraining completed!")
    return rewards_history, epsilon_history

    # this function is used to populate the Q-table. What it does is:
    # for each episode it resets the environment and initializes the variables
    # then for each step of the episode it selects an action using the epsilon-greedy policy
    # executes the action in the environment obtaining the next state, the reward and the episode end indicators
    # updates the Q-table using the Q-learning algorithm


def evaluate_agent(env, agent, num_episodes=100):

    print("\nEvaluating agent...")
    test_rewards = []
    
    for episode in range(num_episodes): 
        state, _ = env.reset()
        done = False
        truncated = False # truncated is used for environments with step limits like taxi v3
        total_reward = 0
        
        while not done and not truncated:   
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        test_rewards.append(total_reward)
    
    # Print statistics
    print(f"Average reward over {num_episodes} episodes: {np.mean(test_rewards):.2f}")
    print(f"Minimum reward: {np.min(test_rewards):.2f}")
    print(f"Maximum reward: {np.max(test_rewards):.2f}")
    
    return test_rewards


def plot_training_results(rewards_history, epsilon_history, window_size=100):
    
    plt.figure(figsize=(12, 4))
    
    moving_avg = np.convolve(rewards_history, 
                            np.ones(window_size)/window_size, 
                            mode='valid')
    
    # Plot 1: Raw rewards with moving average (full scale)
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history, alpha=0.2, label='Reward per episode', linewidth=0.5)
    plt.plot(moving_avg, label=f'Moving average ({window_size})', linewidth=2, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward during training (complete view)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rewards ZOOMED on relevant values (from -50 to 20)
    plt.subplot(1, 3, 2)
    plt.plot(rewards_history, alpha=0.2, label='Reward per episode', linewidth=0.5)
    plt.plot(moving_avg, label=f'Moving average ({window_size})', linewidth=2, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward during training (zoom on relevant values)')
    plt.ylim(-50, 20)  # Zoom on values that matter
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilon_history, linewidth=1.5, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay (exploration → exploitation)')
    plt.yscale('log')  # Logarithmic scale
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_demo(env_name, agent):

    print("\nDemo of one episode (rendered):")

    env_render = gym.make(env_name, render_mode="human")
    
    state, _ = env_render.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not done and not truncated:
        action = agent.get_greedy_action(state) 
        state, reward, done, truncated, _ = env_render.step(action)
        total_reward += reward
        steps += 1
    
    print(f"\nEpisode completed in {steps} steps with total reward: {total_reward}")
    env_render.close()


def main():
    """Main function"""
    # Parameters
    env_name = "Taxi-v3"
    num_episodes = 6000
    num_test_episodes = 100
    
    # Create environment
    env = gym.make(env_name)
    
    # Create agent
    agent = Agent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        gamma=0.99,
        alpha=0.8,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01
    )
    
    # Training
    rewards_history, epsilon_history = train_agent(
        env, agent, num_episodes=num_episodes, print_interval=100
    )
    
    # Evaluation
    test_rewards = evaluate_agent(env, agent, num_episodes=num_test_episodes)
    
    # Display results
    plot_training_results(rewards_history, epsilon_history)
    
    # Demo
    run_demo(env_name, agent)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()

# During training we have in the first 1000 episodes an average reward of approximately -160 -200, which is normal since the agent is still exploring the environment.
# In episodes 2000 the important part happens, because the agent has learned the structure of the problem.
# In subsequent episodes the average reward reaches the optimal policy with 7.3 - 7.5
# This is because we give +20 reward when delivering the passenger and -1 for each step, so if we do 20 - 12/13 (steps of optimal policy) we get approximately 7.3 - 7.5.
