import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
from base_model import Model

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# If cuda is available, set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficNetwork(nn.Module):
    """
    Neural network for traffic signal control
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(TrafficNetwork, self).__init__()
        # Three fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Batch normalization layers for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Dropout layers to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(device)

        # If x is a scalar or 1D tensor, add a dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Forward pass through network with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    Replay buffer to store and sample experiences for training
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class AdvancedTrafficAgent(Model):
    """
    Advanced traffic agent using Deep Q-Learning with experience replay and target networks
    """

    def __init__(
        self,
        input_dim=21,  # 4 directions × (1 total + 4 vehicle types) + traffic ratio
        hidden_dim=128,
        output_dim=2,  # N-S green (True) or E-W green (False)
        learning_rate=0.001,
        gamma=0.99,  # Discount factor
        epsilon=1.0,  # Initial exploration rate
        epsilon_decay=0.995,
        min_epsilon=0.05,
        batch_size=64,
        update_target_every=1000,  # Update target network every N steps
        buffer_capacity=100000,  # Experience replay buffer capacity
        min_action_duration=300,  # Minimum duration (in ticks) before allowing signal change
    ):
        super().__init__()

        # Network parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize main network and target network
        self.policy_net = TrafficNetwork(input_dim, hidden_dim, output_dim).to(device)
        self.target_net = TrafficNetwork(input_dim, hidden_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.update_counter = 0
        self.update_target_every = update_target_every

        # Tracking variables
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None
        self.tick_counter = 0
        self.min_action_duration = min_action_duration
        self.training_stats = {"losses": [], "rewards": [], "epsilons": []}

        # Traffic patterns detection
        self.traffic_history = deque(maxlen=1000)  # Store recent traffic patterns
        self.peak_times = {
            "ns": [],  # Store timestamps of N-S traffic peaks
            "ew": [],  # Store timestamps of E-W traffic peaks
        }

    def state_to_tensor(self, state):
        """
        Convert the state dictionary to a tensor for the neural network
        This creates a more detailed representation than the TD learning model
        """
        # Extract vehicle counts for each direction and type
        feature_vector = []

        # Add total vehicles in each direction
        for direction in ["right", "left", "up", "down"]:
            total = sum(state[direction].values())
            feature_vector.append(total)

            # Optionally add counts by vehicle type for more detailed state
            for vehicle_type in ["car", "bus", "truck", "bike"]:
                if vehicle_type in state[direction]:
                    feature_vector.append(state[direction][vehicle_type])
                else:
                    feature_vector.append(0)

        # Calculate N-S vs E-W traffic ratio (more detailed than TD learning)
        ns_traffic = feature_vector[2] + feature_vector[3]  # up + down totals
        ew_traffic = feature_vector[0] + feature_vector[1]  # right + left totals

        # Calculate various traffic metrics
        if ns_traffic == 0 and ew_traffic == 0:
            ratio = 0
        elif ns_traffic == 0:
            ratio = 5  # Strongly favor east-west
        elif ew_traffic == 0:
            ratio = -5  # Strongly favor north-south
        else:
            # Continuous ratio, not discretized
            ratio = (ew_traffic - ns_traffic) / (ew_traffic + ns_traffic)

        feature_vector.append(ratio)

        # Store traffic pattern for future analysis
        self.traffic_history.append((ns_traffic, ew_traffic))

        # Add time-based features (time of day patterns, if simulation supported it)
        # This would help the agent learn traffic patterns based on time

        # Convert to tensor and normalize values
        return torch.FloatTensor(feature_vector).to(device)

    def get_action(self, state, history):
        """
        Choose an action based on the current state using epsilon-greedy policy

        Args:
            state (dict): Dictionary containing vehicle counts for each direction and type
            history (list): List of previous actions

        Returns:
            bool: True for north-south green, False for east-west green
        """
        # Increment tick counter
        self.tick_counter += 1

        # If we haven't reached minimum action duration, stick with previous action
        if (
            self.previous_action is not None
            and self.tick_counter < self.min_action_duration
        ):
            return self.previous_action

        # Convert state to tensor
        state_tensor = self.state_to_tensor(state)

        # Choose action using epsilon-greedy policy
        if random.random() < self.epsilon:
            # Exploration: choose random action
            action_idx = random.randint(0, self.output_dim - 1)
        else:
            # Exploitation: choose best action according to policy network
            with torch.no_grad():
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
                self.policy_net.train()

        # Convert action index to boolean (0 = False for E-W, 1 = True for N-S)
        action = bool(action_idx)

        # If this is not the first action and we have all components for a transition
        if self.previous_state is not None and self.previous_action is not None:
            # Store the transition in replay buffer
            # The reward will be updated in the next call
            # For now use a placeholder reward of 0
            self.replay_buffer.add(
                self.previous_state.detach().cpu().numpy(),
                1 if self.previous_action else 0,
                0,  # Placeholder reward, will be updated
                state_tensor.detach().cpu().numpy(),
                False,  # Not terminal state
            )

        # Reset tick counter if action changed
        if self.previous_action != action:
            self.tick_counter = 0

        # Store current state and action for next iteration
        self.previous_state = state_tensor
        self.previous_action = action

        return action

    def update_reward(self, reward):
        """
        Update the most recent transition in replay buffer with actual reward

        Args:
            reward (float): The reward received from the environment
        """
        if len(self.replay_buffer.buffer) > 0:
            # Get most recent transition
            state, action, _, next_state, done = self.replay_buffer.buffer[-1]

            # Update the reward
            self.replay_buffer.buffer[-1] = (state, action, reward, next_state, done)

            # Track rewards for stats
            self.training_stats["rewards"].append(reward)

    def update_last_transition_as_terminal(self, reward):
        """
        Mark the last transition as terminal with final reward

        Args:
            reward (float): Final reward for the episode
        """
        if len(self.replay_buffer.buffer) > 0:
            # Get most recent transition
            state, action, _, next_state, done = self.replay_buffer.buffer[-1]

            # Update reward and mark as terminal
            self.replay_buffer.buffer[-1] = (state, action, reward, next_state, True)

    def train(self):
        """
        Train the network using sampled mini-batch from replay buffer
        """
        # If replay buffer doesn't have enough samples, skip training
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Track loss for stats
        self.training_stats["losses"].append(loss.item())

        # Update target network if it's time
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Decay the exploration rate
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.training_stats["epsilons"].append(self.epsilon)

    def save_model(self, filename=None):
        """
        Save the model to a file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_agent_{timestamp}.pt"

        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")

        # Save model
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "training_stats": self.training_stats,
            "traffic_history": list(self.traffic_history),
            "peak_times": self.peak_times,
        }

        torch.save(checkpoint, os.path.join("models", filename))
        print(f"Model saved to models/{filename}")

    def load_model(self, filename):
        """
        Load the model from a file
        """
        # Load checkpoint
        checkpoint = torch.load(os.path.join("models", filename))

        # Recreate networks if dimensions don't match
        if (
            self.input_dim != checkpoint["input_dim"]
            or self.hidden_dim != checkpoint["hidden_dim"]
            or self.output_dim != checkpoint["output_dim"]
        ):

            self.input_dim = checkpoint["input_dim"]
            self.hidden_dim = checkpoint["hidden_dim"]
            self.output_dim = checkpoint["output_dim"]

            self.policy_net = TrafficNetwork(
                self.input_dim, self.hidden_dim, self.output_dim
            ).to(device)
            self.target_net = TrafficNetwork(
                self.input_dim, self.hidden_dim, self.output_dim
            ).to(device)

        # Load state dictionaries
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load other parameters
        self.epsilon = checkpoint["epsilon"]
        self.training_stats = checkpoint["training_stats"]
        self.traffic_history = deque(checkpoint["traffic_history"], maxlen=1000)
        self.peak_times = checkpoint["peak_times"]

        print(f"Model loaded from models/{filename}")

    def analyze_traffic_patterns(self):
        """
        Analyze stored traffic patterns to detect peaks and trends
        """
        if len(self.traffic_history) < 100:
            return

        # Extract NS and EW traffic data
        ns_traffic = [t[0] for t in self.traffic_history]
        ew_traffic = [t[1] for t in self.traffic_history]

        # Calculate moving averages
        window_size = 50
        ns_avg = np.convolve(
            ns_traffic, np.ones(window_size) / window_size, mode="valid"
        )
        ew_avg = np.convolve(
            ew_traffic, np.ones(window_size) / window_size, mode="valid"
        )

        # Detect peaks in NS traffic
        for i in range(1, len(ns_avg) - 1):
            if (
                ns_avg[i] > ns_avg[i - 1]
                and ns_avg[i] > ns_avg[i + 1]
                and ns_avg[i] > np.mean(ns_avg) + np.std(ns_avg)
            ):
                self.peak_times["ns"].append(i)

        # Detect peaks in EW traffic
        for i in range(1, len(ew_avg) - 1):
            if (
                ew_avg[i] > ew_avg[i - 1]
                and ew_avg[i] > ew_avg[i + 1]
                and ew_avg[i] > np.mean(ew_avg) + np.std(ew_avg)
            ):
                self.peak_times["ew"].append(i)

    def plot_training_stats(self):
        """
        Plot training statistics
        """
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Plot losses
        if self.training_stats["losses"]:
            axs[0].plot(self.training_stats["losses"])
            axs[0].set_title("Loss over training")
            axs[0].set_xlabel("Training steps")
            axs[0].set_ylabel("Loss")

        # Plot rewards
        if self.training_stats["rewards"]:
            # Use moving average for smoother curve
            window_size = min(50, len(self.training_stats["rewards"]))
            rewards_avg = np.convolve(
                self.training_stats["rewards"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axs[1].plot(rewards_avg)
            axs[1].set_title("Average Reward over training")
            axs[1].set_xlabel("Training steps")
            axs[1].set_ylabel("Average Reward")

        # Plot epsilon
        if self.training_stats["epsilons"]:
            axs[2].plot(self.training_stats["epsilons"])
            axs[2].set_title("Exploration rate (epsilon) over training")
            axs[2].set_xlabel("Training steps")
            axs[2].set_ylabel("Epsilon")

        plt.tight_layout()

        # Create plots directory if it doesn't exist
        if not os.path.exists("plots"):
            os.makedirs("plots")

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"plots/training_stats_{timestamp}.png")
        plt.close()


def train_advanced_agent(
    agent=None, episodes=100, ticks_per_episode=36000, save_interval=10
):
    """
    Train the advanced traffic agent

    Args:
        agent (AdvancedTrafficAgent): The agent to train, creates a new one if None
        episodes (int): Number of training episodes
        ticks_per_episode (int): Number of ticks per episode
        save_interval (int): Save model every this many episodes

    Returns:
        AdvancedTrafficAgent: The trained agent
    """
    from simulation import simulate

    # Create agent if not provided
    if agent is None:
        agent = AdvancedTrafficAgent()

    # Create directories if they don't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Track rewards for evaluation
    rewards_history = []

    # Track overall training progress
    total_start_time = datetime.now()

    print(f"Starting training for {episodes} episodes")
    for episode in range(1, episodes + 1):
        # Calculate overall progress
        overall_progress = ((episode - 1) / episodes) * 100

        # Create overall progress bar
        bar_length = 20
        filled_length = int(bar_length * (episode - 1) // episodes)
        progress_bar = "#" * filled_length + "-" * (bar_length - filled_length)

        print(f"\nOverall progress: |{progress_bar}| {overall_progress:.1f}%")
        print(f"Episode {episode}/{episodes}")
        episode_start_time = datetime.now()

        # Run one episode of simulation
        reward = simulate(
            agent,
            TRAINING=True,
            TICKS_PER_SECOND=float("inf"),
            NO_OF_TICKS=ticks_per_episode,
        )

        # Update final reward and mark last transition as terminal
        agent.update_last_transition_as_terminal(reward)

        # Train the network
        agent.train()

        # Decay exploration rate
        agent.decay_epsilon()

        # Analyze traffic patterns
        agent.analyze_traffic_patterns()

        # Track rewards
        rewards_history.append(reward)

        # Calculate episode statistics
        episode_duration = (datetime.now() - episode_start_time).total_seconds()
        time_remaining = episode_duration * (episodes - episode) / 60  # in minutes

        print(f"Episode {episode} completed with reward: {reward}")
        print(f"Episode duration: {episode_duration:.2f} seconds")
        print(f"Estimated time remaining: {time_remaining:.2f} minutes")
        print(f"Current exploration rate: {agent.epsilon:.4f}")

        # Save model at intervals
        if episode % save_interval == 0:
            agent.save_model(f"advanced_agent_episode_{episode}.pt")
            agent.plot_training_stats()

    # Training complete
    total_duration = (
        datetime.now() - total_start_time
    ).total_seconds() / 60  # in minutes
    print(f"\nTraining completed in {total_duration:.2f} minutes")

    # Save final model
    agent.save_model("advanced_agent_final.pt")

    # Plot final training stats
    agent.plot_training_stats()

    # Plot rewards history
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("plots/rewards_per_episode.png")
    plt.close()

    return agent


def evaluate_advanced_agent(model_path, num_evaluations=5, ticks_per_evaluation=36000):
    """
    Evaluate a trained advanced agent

    Args:
        model_path (str): Path to the trained model
        num_evaluations (int): Number of evaluation runs
        ticks_per_evaluation (int): Number of ticks per evaluation

    Returns:
        float: Average reward across evaluations
    """
    from simulation import simulate

    agent = AdvancedTrafficAgent()
    agent.load_model(model_path)

    # Set to evaluation mode (no exploration)
    agent.epsilon = 0.0

    rewards = []

    print(f"Evaluating agent for {num_evaluations} runs...")
    for i in range(num_evaluations):
        print(f"Evaluation run {i+1}/{num_evaluations}")
        reward = simulate(
            agent,
            TRAINING=False,
            TICKS_PER_SECOND=600,  # Use higher speed for evaluation
            NO_OF_TICKS=ticks_per_evaluation,
        )
        rewards.append(reward)
        print(f"Evaluation {i+1} reward: {reward}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average evaluation reward: {avg_reward}")

    return avg_reward


if __name__ == "__main__":
    # Train a new agent
    # agent = train_advanced_agent(episodes=100)

    # Or load and evaluate an existing agent
    # evaluate_advanced_agent("models/advanced_agent_final.pt")

    # For running simulation with a trained agent
    from simulation import simulate

    agent = AdvancedTrafficAgent()

    # Uncomment to load a trained model
    # agent.load_model("models/advanced_agent_final.pt")

    # For evaluation, no exploration
    agent.epsilon = 0.0

    # Run simulation
    simulate(agent, TRAINING=False, TICKS_PER_SECOND=60)
