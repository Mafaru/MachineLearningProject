import gymnasium as gym  # provides taxi v3, the MDP
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time




@dataclass(frozen=True)
#frozen=true make transitions immutable
#Any is good to use Numpy or Torch
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool



class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int): 
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} elements from buffer of size {len(self.buffer)}"
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def printElement(self, i: int):
        if not (-len(self.buffer) <= i < len(self.buffer)):
            raise IndexError(
                f"Index {i} out of range for buffer of size {len(self.buffer)}"
            )

        transition = self.buffer[i]
        print(f"[ReplayBuffer] Element {i}: {transition}")




class QNetwork(nn.Module):

    def __init__(self, n_states, n_actions):

        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(n_states, 128) #trasforma gli n_stati in un vettore di 128 elementi, questo è il primo strato completamente connesso
        self.fc2 = nn.Linear(128, 128) #trasforma il vettore di 128 elementi in un altro vettore di 128 elementi, questo è il secondo strato completamente connesso
        self.out = nn.Linear(128, n_actions) #trasforma il vettore di 128 elementi in un vettore con n_actions elementi, questo è lo strato di output che fornisce i Q-values per ogni azione possibile nello stato dato

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.out(x)  # Q-values for all actions




class DQNAgent:
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, buffer_size=50000, min_buffer_size=1000,
                  max_steps_per_episode=200):
        """
        Initializes the DQN agent
        
        Args:
            n_states: Number of states in the environment, for taxi v3 there are 500
            n_actions: Number of available actions which is 6
            alpha: Learning rate which is the learning speed
            gamma: Discount factor, used to balance the importance of future rewards compared to immediate ones
            epsilon: Initial exploration rate, set to 1 to explore a lot at the beginning and decrease over time
            epsilon_decay: Decay factor for epsilon, this is the rate at which epsilon decreases after each episode, used in the decay_epsilon function
            epsilon_min: Minimum value of epsilon, a minimum value is kept to prevent it from becoming 0 and therefore never exploring again
        """
        
        self.n_states = n_states # 500
        self.n_actions = n_actions # 6 = south, north, west, east, pickup, dropoff
        self.alpha = alpha 
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 
        
        #DQN Parameters
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_steps_per_episode = max_steps_per_episode
        
        #REPLAY BUFFER
        self.buffer = ReplayBuffer(self.buffer_size)

        # NEURAL NETWORK
        self.q_network = QNetwork(n_states, n_actions)
        #first optimizer
        #self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3) 
        self.loss_fn = nn.MSELoss() #questo è l'errore quadratico medio, usato per calcolare la differenza tra i Q-values predetti dalla rete e i target Q-values calcolati durante l'addestramento


    #trasform the state number into a vector with all zeros, exept the number of state
    def one_hot(self, state):
        vec = torch.zeros(self.n_states)
        vec[state] = 1.0
        return vec


    def select_action(self, state, training=True):
        """
        Agent policy: epsilon-greedy
        
        Args:
            state: Current state
            training: If True uses epsilon-greedy, otherwise greedy
            
        Returns:
            Selected action
        """
        # ε-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploitation using the Q-network
        state_tensor = self.one_hot(state).unsqueeze(0) 

        with torch.no_grad():
            q_values = self.q_network(state_tensor) # Q-values for all actions in the current state

        return torch.argmax(q_values).item() #questo restituisce l'indice dell'azione con il Q-value più alto, che è l'azione che il DQN ritiene migliore in base alla sua attuale stima
    

    def train_step(self):
        if len(self.buffer) < self.min_buffer_size:
            return

        batch = self.buffer.sample(self.batch_size) # questo è un campione di transizioni, è una lista di oggetti Transition, ognuno con state, action, reward, next_state e done, casuali.
        batch_size = len(batch)

        # Preallocate tensors
        states = torch.zeros(batch_size, self.n_states, dtype=torch.float32) 
        next_states = torch.zeros(batch_size, self.n_states, dtype=torch.float32)
        actions = torch.zeros(batch_size, dtype=torch.long)
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        dones = torch.zeros(batch_size, dtype=torch.float32)

        # Fill tensors
        for i, t in enumerate(batch):
            states[i, t.state] = 1.0 
            next_states[i, t.next_state] = 1.0
            actions[i] = t.action 
            rewards[i] = t.reward
            dones[i] = t.done

        # Compute current Q(s, a)
        q_values = self.q_network(states) #restituisce i Q-values per tutte le azioni in tutti gli stati del batch, è una matrice di dimensione (batch_size, n_actions)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) #qui invece prende il Q value solo dell'azione che stiamo eseguendo.

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states) #calcola i q_values per gli stati successivi.
            max_next_q = next_q_values.max(dim=1)[0] #per ogni riga prende il massimo Q value.
            targets = rewards + self.gamma * max_next_q * (1 - dones) #e questo vettore di massimi Q values viene usato per calcolare i target Q values, che sono la ricompensa immediata più il valore scontato del miglior Q value del prossimo stato, moltiplicato per (1 - done) per azzerare il target se l'episodio è finito.

        # Compute loss
        loss = F.mse_loss(q_sa, targets) #questo è importante, perche confronta i Q values predetti dalla rete q_sa con i target Q values, se la differenza è tanta significa che la rete non sta facendo un buon lavoro nel predire i Q values corretti, e quindi il loss sarà alto, se invece la differenza è piccola, significa che la rete sta predicendo bene i Q values, e quindi il loss sarà basso. L'obiettivo dell'addestramento è minimizzare questo loss, in modo che la rete impari a predire Q values più accurati.

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decay_epsilon(self):
        """Applies decay to epsilon"""
        #TOO FAST with decay of 0.995
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # decreases epsilon by multiplying it by the decay factor, but does not let it go below epsilon_min
        
        #WE HAVE TO MODIFY THIS
        #self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * exp(-self.decay_rate * episode)


def warmup(env, agent):
        """
        Fills the replay buffer with random experiences
        before training starts.
        """
        np.random.seed(0)
        state, _ = env.reset(seed=0)
        steps = 0
        
        print("Starting Warm Up Phase....")

        #initialize time
        start_warmup_time = time.time()

        while len(agent.buffer) < agent.min_buffer_size:
            
            # Choose a random action (pure exploration)
            action = np.random.randint(agent.n_actions)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)

            #instanitate transition
            new_transition = Transition(state, action, reward, next_state, done)
            
            # Store transition in replay buffer
            agent.buffer.push(new_transition)
            
            state = next_state
            steps += 1
            
            # Reset episode if finished
            if done or truncated or steps >= agent.max_steps_per_episode:
                state, _ = env.reset()
                steps = 0
            
        
        #calculate delta
        end_warmup_time = time.time()
        total_time = end_warmup_time - start_warmup_time
        
        print(f"Warm Up ended, Total warmup time {total_time:.2f} seconds")


def train_agent(env, agent, num_episodes=2000, print_interval=100):
    """
    Trains the agent in the environment
    
    Args:
        env: Gymnasium environment
        agent: QLearningAgent
        num_episodes: Number of training episodes
        print_interval: Interval for printing progress
        
    Returns:
        rewards_history: List of rewards per episode
        epsilon_history: List of epsilon values
    """
    rewards_history = []
    epsilon_history = []
    
    print("Starting training...")

    #INITIALIZE TIMER
    start_training_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(agent.max_steps_per_episode):

            # 1. Select action (ε-greedy)
            action = agent.select_action(state, training=True)

            # 2. Execute action
            next_state, reward, done, truncated, _ = env.step(action)

            # 3. Store transition in replay buffer
            transition = Transition(state, action, reward, next_state, done)
            agent.buffer.push(transition)

            # 4. Train the Q-network
            #agent.train_step()
            #TO ENHANCE SPEED
            if step % 4 == 0:
              agent.train_step()

            state = next_state
            total_reward += reward

            if done or truncated:
                break
        
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
    
    #calulating the delta
    end_training_time = time.time()
    total_time = end_training_time - start_training_time
    minutes, seconds = divmod(total_time, 60)

    print(f"\nTraining completed, in Total time: {minutes:.0f} min {seconds:.0f} sec")
    #print(f"\nSome metrics, epsilon decay {agent.epsilon_decay:.5f}")
    return rewards_history, epsilon_history


def evaluate_agent(env, agent, num_episodes=100):
    """
    Evaluates the agent's performance
    
    Args:
        env: Gymnasium environment
        agent: Trained QLearningAgent
        num_episodes: Number of evaluation episodes
        
    Returns:
        test_rewards: List of obtained rewards
    """
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
    
    plt.figure(figsize=(16, 5))
    
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
        action = agent.select_action(state, training=False) 
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
    dqnagent = DQNAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01,
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=1000,
        max_steps_per_episode=200
    )

    print(f"\nAgent's Parameters:\n\n Epsilon decay:", dqnagent.epsilon_decay , "\n Batch size:", dqnagent.batch_size , 
          "\n Gamma:", dqnagent.gamma , "\n Number of episodes:" , num_episodes , "\n\n")

    #WARMUP FUNCTION, FILLS THE REPLAY BUFFER
    warmup(env,dqnagent)

    #TRAINING
    reward_history, epsilon_history = train_agent(env,dqnagent, num_episodes)

    #EVALUATE THE AGENT
    evaluate_agent(env,dqnagent)

    #PLOT TRAINING RESULTS
    plot_training_results(reward_history, epsilon_history)

    #RUN DEMO
    run_demo(env_name,dqnagent)


    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()

