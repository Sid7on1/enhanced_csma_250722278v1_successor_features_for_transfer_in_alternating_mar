import torch
import numpy as np
from typing import Tuple, Dict, Optional
from torch import nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    Policy Network class for the agent in a Markov Game.
    This network represents the policy π(a|s) - the probability of taking action a given state s.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the Policy Network.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        """
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the network parameters using xavier initialization.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean_fc.weight)
        nn.init.xavier_uniform_(self.log_std_fc.weight)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        :param state: Batch of states with shape (batch_size, state_dim).
        :return: Mean and log standard deviation of the Gaussian policy distribution.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, min=-5, max=2)  # Clamp for numerical stability
        return mean, log_std

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy network for a given state.
        :param state: Current state of the environment.
        :return: Action sampled from the Gaussian policy distribution.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        mean, log_std = self.forward(state_tensor)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample action from distribution
        action = torch.tanh(x_t).numpy().squeeze(0)  # Tanh squashing and remove batch dimension
        return action

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Get actions from the policy network for a batch of states.
        :param states: Batch of states with shape (batch_size, state_dim).
        :return: Batch of actions sampled from the Gaussian policy distribution.
        """
        states_tensor = torch.FloatTensor(states)
        mean, log_std = self.forward(states_tensor)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample actions from distribution
        actions = torch.tanh(x_t).numpy()  # Tanh squashing
        return actions

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the network weights and biases to a dictionary.
        :return: Dictionary containing weights and biases of the network layers.
        """
        data = {}
        data['fc1_weight'] = self.fc1.weight.detach().numpy().tolist()
        data['fc1_bias'] = self.fc1.bias.detach().numpy().tolist()
        data['fc2_weight'] = self.fc2.weight.detach().numpy().tolist()
        data['fc2_bias'] = self.fc2.bias.detach().numpy().tolist()
        data['mean_fc_weight'] = self.mean_fc.weight.detach().numpy().tolist()
        data['mean_fc_bias'] = self.mean_fc.bias.detach().numpy().tolist()
        data['log_std_fc_weight'] = self.log_std_fc.weight.detach().numpy().tolist()
        data['log_std_fc_bias'] = self.log_std_fc.bias.detach().numpy().tolist()
        return data

    def from_dict(self, weight_dict: Dict[str, float]):
        """
        Set the network weights and biases from a dictionary.
        :param weight_dict: Dictionary containing weights and biases of the network layers.
        """
        self.fc1.weight.data = torch.Tensor(weight_dict['fc1_weight'])
        self.fc1.bias.data = torch.Tensor(weight_dict['fc1_bias'])
        self.fc2.weight.data = torch.Tensor(weight_dict['fc2_weight'])
        self.fc2.bias.data = torch.Tensor(weight_dict['fc2_bias'])
        self.mean_fc.weight.data = torch.Tensor(weight_dict['mean_fc_weight'])
        self.mean_fc.bias.data = torch.Tensor(weight_dict['mean_fc_bias'])
        self.log_std_fc.weight.data = torch.Tensor(weight_dict['log_std_fc_weight'])
        self.log_std_fc.bias.data = torch.Tensor(weight_dict['log_std_fc_bias'])


class Policy:
    """
    Policy class that encapsulates the Policy Network and provides additional functionality.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the Policy.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy for a given state.
        :param state: Current state of the environment.
        :return: Action sampled from the Gaussian policy distribution.
        """
        return self.policy_net.get_action(state)

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Get actions from the policy for a batch of states.
        :param states: Batch of states with shape (batch_size, state_dim).
        :return: Batch of actions sampled from the Gaussian policy distribution.
        """
        return self.policy_net.get_actions(states)

    def save(self, filename: str):
        """
        Save the policy network weights to a file.
        :param filename: Name of the file to save the weights to.
        """
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename: str):
        """
        Load the policy network weights from a file.
        :param filename: Name of the file to load the weights from.
        """
        self.policy_net.load_state_dict(torch.load(filename))

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the policy network weights and biases to a dictionary.
        :return: Dictionary containing weights and biases of the policy network layers.
        """
        return self.policy_net.to_dict()

    def from_dict(self, weight_dict: Dict[str, float]):
        """
        Set the policy network weights and biases from a dictionary.
        :param weight_dict: Dictionary containing weights and biases of the policy network layers.
        """
        self.policy_net.from_dict(weight_dict)


class PolicyWithKNN(Policy):
    """
    Policy class that extends the base Policy class and incorporates K-Nearest Neighbors (KNN) for action selection.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, k: int = 5):
        """
        Initializes the Policy with KNN.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param k: Number of neighbors to consider in KNN.
        """
        super(PolicyWithKNN, self).__init__(state_dim, action_dim, hidden_dim)
        self.k = k
        self.state_action_pairs = []

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy for a given state using KNN.
        :param state: Current state of the environment.
        :return: Action selected based on KNN.
        """
        # Check if KNN can be used
        if len(self.state_action_pairs) >= self.k:
            dists = np.linalg.norm(self.state_action_pairs[:, :self.state_dim] - state, axis=1)
            nearest_idxs = np.argsort(dists)[:self.k]
            actions = self.state_action_pairs[nearest_idxs, self.state_dim:]
            return np.mean(actions, axis=0)
        else:
            # Fall back to base policy if not enough state-action pairs
            return super().get_action(state)

    def update(self, state: np.ndarray, action: np.ndarray):
        """
        Update the state-action pairs used for KNN.
        :param state: Current state of the environment.
        :param action: Action taken in the current state.
        """
        self.state_action_pairs.append(np.concatenate((state, action)))


class PolicyWithTransfer(Policy):
    """
    Policy class that extends the base Policy class and incorporates successor features for transfer learning.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, lambda_val: float = 0.5):
        """
        Initializes the Policy with Transfer.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param lambda_val: Weight for the successor features in the policy update.
        """
        super(PolicyWithTransfer, self).__init__(state_dim, action_dim, hidden_dim)
        self.lambda_val = lambda_val
        self.successor_features = None

    def update(self, successor_features: np.ndarray):
        """
        Update the successor features used for transfer learning.
        :param successor_features: Successor features for the current task.
        """
        self.successor_features = successor_features

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy for a given state using successor features for transfer learning.
        :param state: Current state of the environment.
        :return: Action sampled from the Gaussian policy distribution considering successor features.
        """
        mean, log_std = self.policy_net.forward(torch.FloatTensor(state).unsqueeze(0))
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample action from distribution
        action = torch.tanh(x_t).numpy().squeeze(0)  # Tanh squashing and remove batch dimension

        # Incorporate successor features
        action += self.lambda_val * self.successor_features

        # Ensure action is within valid range
        action = np.clip(action, -1, 1)

        return action


class PolicyEnsemble(Policy):
    """
    Policy class that represents an ensemble of multiple policies.
    """

    def __init__(self, policies: list):
        """
        Initializes the Policy Ensemble.
        :param policies: List of Policy objects to include in the ensemble.
        """
        self.policies = policies

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy ensemble for a given state.
        :param state: Current state of the environment.
        :return: Action sampled from one of the policies in the ensemble.
        """
        # Randomly select a policy from the ensemble
        idx = np.random.choice(len(self.policies))
        return self.policies[idx].get_action(state)


class PolicyWithVelocityThreshold(Policy):
    """
    Policy class that extends the base Policy class and incorporates a velocity threshold for action selection.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, threshold: float = 0.1):
        """
        Initializes the Policy with Velocity Threshold.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param threshold: Velocity threshold for action selection.
        """
        super(PolicyWithVelocityThreshold, self).__init__(state_dim, action_dim, hidden_dim)
        self.threshold = threshold

    def get_action(self, state: np.ndarray, velocity: float) -> np.ndarray:
        """
        Get action from the policy for a given state considering the velocity threshold.
        :param state: Current state of the environment.
        :param velocity: Velocity of the agent.
        :return: Action sampled from the Gaussian policy distribution or zero action if velocity is below threshold.
        """
        if np.linalg.norm(velocity) < self.threshold:
            return np.zeros(self.action_dim)
        else:
            return super().get_action(state)


class PolicyWithFlowTheory(Policy):
    """
    Policy class that extends the base Policy class and incorporates Flow Theory for action selection.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, flow_state: np.ndarray = None):
        """
        Initializes the Policy with Flow Theory.
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Number of neurons in the hidden layer.
        :param flow_state: Initial flow state represented as a 2D array.
        """
        super(PolicyWithFlowTheory, self).__init__(state_dim, action_dim, hidden_dim)
        self.flow_state = flow_state if flow_state is not None else np.random.random((2, 2))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from the policy for a given state using Flow Theory.
        :param state: Current state of the environment.
        :return: Action sampled from the Gaussian policy distribution considering Flow Theory.
        """
        # Update flow state using the current state
        self.flow_state = self._update_flow_state(state)

        # Sample action from the policy
        action = super().get_action(state)

        # Incorporate flow state into the action
        action += self.flow_state[0, 0] * state

        return action

    def _update_flow_state(self, state: np.ndarray) -> np.ndarray:
        """
        Update the flow state using the current state based on Flow Theory.
        :param state: Current state of the environment.
        :return: Updated flow state.
        """
        # Define the flow function
        flow_function = lambda x: np.sin(x) if 0 <= x < np.pi/2 else np.cos(x - np.pi/2)

        # Update the flow state using the flow function
        self.flow_state[0, 1] = flow_function(self.flow_state[0, 0])
        self.flow_state[1, 1] = flow_function(self.flow_state[1, 0])

        # Update the flow state using the current state
        self.flow_state[:, 0] += state

        return self.flow_state


class PolicyManager:
    """
    Policy Manager class that provides an interface for selecting and updating policies.
    """

    def __init__(self, policies: Dict[str, Policy]):
        """
        Initializes the Policy Manager.
        :param policies: Dictionary of Policy objects indexed by unique names.
        """
        self.policies = policies

    def get_policy(self, name: str) -> Optional[Policy]:
        """
        Get a policy by its unique name.
        :param name: Unique name of the policy.
        :return: Policy object or None if the policy is not found.
        """
        return self.policies.get(name)

    def set_policy(self, name: str, policy: Policy):
        """
        Set a policy by its unique name.
        :param name: Unique name of the policy.
        :param policy: Policy object to set.
        """
        self.policies[name] = policy

    def remove_policy(self, name: str):
        """
        Remove a policy by its unique name.
        :param name: Unique name of the policy.
        """
        self.policies.pop(name, None)


# Example usage
if __name__ == "__main