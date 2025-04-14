import numpy as np
import random
import pickle
import os
from datetime import datetime
from simulation import simulate
from base_model import Model
from simulation_config import DEFAULT_SCAN_ZONE_CONFIG, vehicleTypes


class TDLearningModel(Model):
    """
    A Temporal Difference (TD) Learning model for traffic signal control.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2,
        exploration_decay=0.999,
        min_exploration_rate=0.01,
    ):
        super().__init__()
        self.learning_rate = learning_rate  # Alpha: Learning rate
        self.discount_factor = (
            discount_factor  # Gamma: Discount factor for future rewards
        )
        self.exploration_rate = (
            exploration_rate  # Epsilon: For exploration-exploitation tradeoff
        )
        self.exploration_decay = exploration_decay  # Decay rate for exploration
        self.min_exploration_rate = min_exploration_rate  # Minimum exploration rate

        # Initialize Q-table as a dictionary for state-action pairs
        self.q_table = {}

        # Track previous state, action and reward for TD learning
        self.previous_state = None
        self.previous_action = None
        self.tick_counter = 0
        self.min_action_duration = 200  # Minimum duration for an action in ticks

    def state_to_key(self, state):
        """
        Convert the state dictionary to a hashable key for the Q-table.
        Simplifies the state to make learning more effective.
        """
        # Create a simplified representation of the state
        simplified_state = []

        # Count total vehicles in each direction
        for direction in ["right", "left", "up", "down"]:
            total = sum(state[direction].values())
            simplified_state.append(total)

        # Add a component that reflects the ratio of north-south vs east-west traffic
        ns_traffic = simplified_state[2] + simplified_state[3]  # up + down
        ew_traffic = simplified_state[0] + simplified_state[1]  # right + left

        # Add a traffic ratio indicator (discretized)
        if ns_traffic == 0 and ew_traffic == 0:
            ratio = 0
        elif ns_traffic == 0:
            ratio = 5  # Strongly favor east-west
        elif ew_traffic == 0:
            ratio = -5  # Strongly favor north-south
        else:
            ratio = int(5 * (ew_traffic - ns_traffic) / (ew_traffic + ns_traffic))

        simplified_state.append(ratio)

        # Convert to tuple for hashing
        return tuple(simplified_state)

    def get_action(self, state, history):
        """
        Choose an action based on the current state using the Q-table.

        Args:
            state (dict): A dictionary containing vehicle counts for each direction and type
            history (list): A list of previous actions taken by the model

        Returns:
            bool: True for north-south green, False for east-west green
        """
        # Increment tick counter
        self.tick_counter += 1

        # If we haven't reached the minimum action duration, stick with the previous action
        if (
            self.previous_action is not None
            and self.tick_counter < self.min_action_duration
        ):
            return self.previous_action

        # Convert state to a hashable key
        state_key = self.state_to_key(state)

        # If this state is not in the Q-table, add it
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                True: 0.0,
                False: 0.0,
            }  # Initialize Q-values for both actions

        # Choose between exploration and exploitation
        if random.random() < self.exploration_rate:
            # Exploration: choose a random action
            action = random.choice([True, False])
        else:
            # Exploitation: choose the best action according to Q-table
            if self.q_table[state_key][True] > self.q_table[state_key][False]:
                action = True
            elif self.q_table[state_key][False] > self.q_table[state_key][True]:
                action = False
            else:
                # If Q-values are equal, choose randomly
                action = random.choice([True, False])

        # Reset tick counter if the action changed
        if self.previous_action != action:
            self.tick_counter = 0

        # Store the current state and action for the next update
        self.previous_state = state_key
        self.previous_action = action

        return action

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a state-action pair using the TD learning formula.

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        Args:
            state: The state key
            action: The action taken
            reward: The reward received
            next_state: The resulting state key
        """
        if state not in self.q_table:
            self.q_table[state] = {True: 0.0, False: 0.0}

        if next_state not in self.q_table:
            self.q_table[next_state] = {True: 0.0, False: 0.0}

        # Calculate the maximum Q-value for the next state
        max_next_q = max(self.q_table[next_state].values())

        # Current Q-value
        current_q = self.q_table[state][action]

        # TD update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update the Q-value
        self.q_table[state][action] = new_q

    def decay_exploration_rate(self):
        """Decay the exploration rate to gradually shift from exploration to exploitation"""
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def save_model(self, filename=None):
        """Save the model to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"td_model_{timestamp}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "exploration_rate": self.exploration_rate,
                    "exploration_decay": self.exploration_decay,
                    "min_exploration_rate": self.min_exploration_rate,
                },
                f,
            )

        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load the model from a file"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.learning_rate = data["learning_rate"]
            self.discount_factor = data["discount_factor"]
            self.exploration_rate = data["exploration_rate"]
            self.exploration_decay = data["exploration_decay"]
            self.min_exploration_rate = data["min_exploration_rate"]

        print(f"Model loaded from {filename}")


def train_model(episodes=100, ticks_per_episode=36000, save_interval=10):
    """
    Train the TD learning model using the traffic simulation.

    Args:
        episodes: Number of training episodes
        ticks_per_episode: Number of ticks per episode
        save_interval: Save the model every this many episodes

    Returns:
        The trained model
    """
    model = TDLearningModel()

    # Create directory for models if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

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
            model,
            TRAINING=True,
            TICKS_PER_SECOND=float("inf"),
            NO_OF_TICKS=ticks_per_episode,
        )
        rewards_history.append(reward)

        episode_duration = (datetime.now() - episode_start_time).total_seconds()

        # Calculate estimated time remaining
        avg_episode_time = (datetime.now() - total_start_time).total_seconds() / episode
        est_time_remaining = avg_episode_time * (episodes - episode)

        print(f"Episode {episode} completed with reward: {reward}")
        print(f"Episode duration: {episode_duration:.2f} seconds")
        print(f"Estimated time remaining: {est_time_remaining/60:.2f} minutes")

        # Decay exploration rate
        model.decay_exploration_rate()
        print(f"Current exploration rate: {model.exploration_rate:.4f}")

        # Save model periodically
        if episode % save_interval == 0:
            model.save_model(f"models/td_model_episode_{episode}.pkl")

    # Print total training time
    total_duration = (datetime.now() - total_start_time).total_seconds()
    print(f"\nTotal training time: {total_duration/60:.2f} minutes")

    # Save the final model
    model.save_model(f"models/td_model_final.pkl")

    # Print training summary
    print("\nTraining completed!")
    print(f"Final exploration rate: {model.exploration_rate:.4f}")
    print(
        f"Average reward over all episodes: {sum(rewards_history) / len(rewards_history):.2f}"
    )
    print(f"Best episode reward: {max(rewards_history):.2f}")

    return model


def evaluate_model(model_path, num_evaluations=5, ticks_per_evaluation=36000):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the saved model
        num_evaluations: Number of evaluation runs
        ticks_per_evaluation: Number of ticks per evaluation

    Returns:
        Average reward over all evaluations
    """
    model = TDLearningModel()
    model.load_model(model_path)

    # Set exploration rate to 0 for deterministic policy
    model.exploration_rate = 0.0

    rewards = []

    print(f"Evaluating model {model_path}")
    for i in range(num_evaluations):
        print(f"Evaluation run {i+1}/{num_evaluations}")
        reward = simulate(
            model, TRAINING=True, TICKS_PER_SECOND=60, NO_OF_TICKS=ticks_per_evaluation
        )
        rewards.append(reward)
        print(f"Evaluation {i+1} completed with reward: {reward}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nEvaluation completed!")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Min reward: {min(rewards):.2f}")
    print(f"Max reward: {max(rewards):.2f}")

    return avg_reward


def run_trained_model(model_path):
    """
    Run a trained model in the simulation for visualization.

    Args:
        model_path: Path to the saved model
    """
    model = TDLearningModel()
    model.load_model(model_path)

    # Set exploration rate to 0 for deterministic policy
    model.exploration_rate = 0.0

    print(f"Running model {model_path} in simulation")
    reward = simulate(model, TRAINING=False, TICKS_PER_SECOND=60)
    print(f"Simulation completed with reward: {reward}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TD Learning for Traffic Signal Control"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "run"],
        help="Mode: train, evaluate, or run",
    )
    parser.add_argument(
        "--model", type=str, help="Path to model file (for evaluate and run modes)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of training episodes"
    )
    parser.add_argument(
        "--ticks", type=int, default=36000, help="Number of ticks per episode"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save interval in episodes"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            episodes=args.episodes,
            ticks_per_episode=args.ticks,
            save_interval=args.save_interval,
        )
    elif args.mode == "evaluate":
        if args.model is None:
            print("Error: Model path required for evaluation mode")
        else:
            evaluate_model(args.model, ticks_per_evaluation=args.ticks)
    elif args.mode == "run":
        if args.model is None:
            print("Error: Model path required for run mode")
        else:
            run_trained_model(args.model)
