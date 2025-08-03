import logging
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Union
from torch import nn
from torch.utils.data import DataLoader
from numpy import linalg as LA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check PyTorch version
if not torch.__version__.startswith("1.10."):
    raise ValueError(
        "This code has only been tested with PyTorch version 1.10."
    )

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the Markov Game
class MarkovGame:
    def __init__(self, num_states, num_actions, reward_matrix, transition_probs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_matrix = reward_matrix
        self.transition_probs = transition_probs

    def get_num_states(self):
        return self.num_states

    def get_num_actions(self):
        return self.num_actions

    def get_reward(self, state, action):
        return self.reward_matrix[state, action]

    def get_transition_probs(self, state, action):
        return self.transition_probs[state, action, :]

# Load the Markov Game from a file
def load_markov_game(file_path):
    # TODO: Implement the loading of the Markov Game from a file
    # Return the MarkovGame object
    pass

# Class to handle experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, done = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done_ = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            done.append(done_)
        return np.array(states), np.array(actions), np.array(rewards), np.array(
            next_states), np.array(done)

    def __len__(self):
        return len(self.buffer)

# Neural network model for the agent
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Agent that uses the Q-learning algorithm
class QLearningAgent:
    def __init__(self, num_inputs, num_actions, gamma=0.99, lr=0.001):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr

        # Create the Q-network
        self.qnetwork = QNetwork(num_inputs, num_actions)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=self.lr)

        # Initialize the target network
        self.target_qnetwork = QNetwork(num_inputs, num_actions)
        self.update_target_network()

    def update_target_network(self):
        # Copy the weights from the Q-network to the target network
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

    def get_action(self, state, epsilon=0.1):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            # Get the Q-values for the current state
            state = torch.tensor(state, dtype=torch.float).to(device)
            Q_values = self.qnetwork(state)
            return torch.argmax(Q_values).item()

    def update(self, state, action, reward, next_state, done):
        # Convert the data to tensors
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device)

        # Get the current Q-values
        Q_values = self.qnetwork(state)

        # Get the target Q-values
        next_state_values = self.target_qnetwork(next_state)
        next_q_value = reward + self.gamma * torch.max(next_state_values) * (1 - done)

        # Compute the Q-value loss
        q_value = Q_values[range(state.shape[0]), action]
        loss = nn.functional.smooth_l1_loss(q_value, next_q_value.detach())

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Agent that uses the Game Generalized Policy Improvement (GGPI) algorithm
class GGPIAgent:
    def __init__(self, num_inputs, num_actions, markov_game: MarkovGame, lr=0.01):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.markov_game = markov_game
        self.lr = lr

        # Create the Q-network for each action
        self.qnetworks = [QNetwork(num_inputs, 1) for _ in range(num_actions)]
        self.optimizers = [
            torch.optim.Adam(qnet.parameters(), lr=self.lr) for qnet in self.qnetworks
        ]

        # Initialize target Q-networks
        self.target_qnetworks = [QNetwork(num_inputs, 1) for _ in range(num_actions)]
        self.update_target_networks()

    def update_target_networks(self):
        # Copy the weights from the Q-networks to the target Q-networks
        for i in range(self.num_actions):
            self.target_qnetworks[i].load_state_dict(self.qnetworks[i].state_dict())

    def get_action(self, state):
        # Get the Q-values for each action
        state = torch.tensor(state, dtype=torch.float).to(device)
        Q_values = torch.cat([qnet(state).unsqueeze(1) for qnet in self.qnetworks], dim=1)
        return torch.argmax(Q_values).item()

    def update(self, state, action, reward, next_state, done):
        # Convert the data to tensors
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device)

        # Get the current Q-value for the chosen action
        Q_value = self.qnetworks[action](state).squeeze()

        # Get the target Q-value
        next_state_values = torch.cat(
            [qnet(next_state).unsqueeze(1) for qnet in self.target_qnetworks], dim=1
        )
        next_q_value = reward + self.markov_game.get_reward(state, action) + self.gamma * torch.max(
            next_state_values * (1 - done)
        )

        # Compute the Huber loss
        loss = nn.functional.smooth_l1_loss(Q_value, next_q_value.detach())

        # Optimize the Q-network for the chosen action
        self.optimizers[action].zero_grad()
        loss.backward()
        self.optimizers[action].step()

# Function to train the agents
def train(agents, markov_game: MarkovGame, num_episodes=1000, max_steps=100, batch_size=64, epsilon=0.1):
    replay_buffer = ReplayBuffer(10000)
    for i_episode in range(num_episodes):
        state = markov_game.get_initial_state()
        for t in range(max_steps):
            # Select actions for each agent
            actions = [agent.get_action(state, epsilon) for agent in agents]

            # Get the rewards and next states
            rewards = []
            next_states = []
            for i, agent in enumerate(agents):
                reward = markov_game.get_reward(state, agent.get_action(state))
                next_state, _ = markov_game.get_next_state(state, actions)
                rewards.append(reward)
                next_states.append(next_state)

            # Update the replay buffer
            replay_buffer.push(state, actions, rewards, next_states, done=False)

            # Sample a batch from the replay buffer
            states, batch_actions, batch_rewards, batch_next_states, batch_done = replay_buffer.sample(
                batch_size)

            # Update the agents
            for i, agent in enumerate(agents):
                agent.update(
                    states[i],
                    batch_actions[i],
                    batch_rewards[i],
                    batch_next_states[i],
                    batch_done[i],
                )

            # Update the target networks periodically
            if i_episode % 10 == 0:
                for agent in agents:
                    agent.update_target_network()

            # Update the current state
            state = next_state

            # Log the progress
            if i_episode % 100 == 0:
                logger.info(f"Episode {i_episode}/{num_episodes}")

        # Decay epsilon for exploration
        epsilon *= 0.99

# Function to evaluate the agents
def evaluate(agents, markov_game: MarkovGame, num_episodes=100, max_steps=100):
    rewards = []
    for i_episode in range(num_episodes):
        state = markov_game.get_initial_state()
        episode_reward = 0
        for t in range(max_steps):
            # Select actions for each agent
            actions = [agent.get_action(state) for agent in agents]

            # Get the rewards and next states
            reward = sum(
                [
                    markov_game.get_reward(state, agent.get_action(state))
                    for agent in agents
                ]
            )
            next_state, _ = markov_game.get_next_state(state, actions)

            # Update the current state
            state = next_state
            episode_reward += reward

        # Log the progress
        rewards.append(episode_reward)
        if i_episode % 100 == 0:
            logger.info(f"Episode {i_episode}/{num_episodes}")

    return rewards

# Function to save the trained model
def save_model(agent, model_path):
    torch.save(agent.state_dict(), model_path)

# Function to load a trained model
def load_model(agent, model_path):
    agent.load_state_dict(torch.load(model_path))

# Example usage
if __name__ == "__main__":
    # Define the Markov Game
    num_states = 5
    num_actions = 2
    reward_matrix = np.random.rand(num_states, num_actions)
    transition_probs = np.random.rand(num_states, num_actions, num_states)
    markov_game = MarkovGame(num_states, num_actions, reward_matrix, transition_probs)

    # Create the agents
    qlearning_agent = QLearningAgent(num_states, num_actions)
    ggpi_agent = GGPIAgent(num_states, num_actions, markov_game)

    # Train the agents
    train([qlearning_agent, ggpi_agent], markov_game, num_episodes=1000, max_steps=100)

    # Evaluate the agents
    qlearning_rewards = evaluate([qlearning_agent], markov_game, num_episodes=100)
    ggpi_rewards = evaluate([ggpi_agent], markov_game, num_episodes=100)

    # Save the trained models
    save_model(qlearning_agent, "qlearning_model.pth")
    save_model(ggpi_agent, "ggpi_model.pth")

    # Load the trained models
    qlearning_agent = QLearningAgent(num_states, num_actions)
    ggpi_agent = GGPIAgent(num_states, num_actions, markov_game)
    load_model(qlearning_agent, "qlearning_model.pth")
    load_model(ggpi_agent, "ggpi_model.pth")

    # Evaluate the loaded models
    loaded_qlearning_rewards = evaluate([qlearning_agent], markov_game, num_episodes=100)
    loaded_ggpi_rewards = evaluate([ggpi_agent], markov_game, num_episodes=100)